"""Phase 2 — Multimodal Contrastive Learning for EEG and Video/Audio Stimuli.

Directly optimizes retrieval ranking using InfoNCE loss (NT-Xent) between
EEG neural recordings and multimodal stimulus representations (ViCLIP + AST).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """Residual 1D convolutional block with batch normalization and GELU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class EEGEncoder(nn.Module):
    """Deep temporal-spatial 1D ConvNet for multichannel EEG.

    Parameters
    ----------
    in_channels : int, default=17
        Number of EEG electrodes.
    proj_dim : int, default=256
        Dimensionality of the joint contrastive latent space.
    dropout : float, default=0.25
        Dropout probability.
    use_subject_adapter : bool, default=False
        Whether to enable per-subject 1x1 spatial adapters.
    num_subjects : int, default=6
        Number of subject adapters to instantiate.
    """

    def __init__(
        self,
        in_channels: int = 17,
        proj_dim: int = 256,
        dropout: float = 0.25,
        use_subject_adapter: bool = False,
        num_subjects: int = 6,
    ):
        super().__init__()
        self.use_subject_adapter = use_subject_adapter
        if use_subject_adapter:
            # 1x1 conv per subject, initialized to zeros so it starts as exact identity
            self.subject_adapters = nn.ModuleList([
                nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
                for _ in range(num_subjects)
            ])
            for adapter in self.subject_adapters:
                nn.init.zeros_(adapter.weight)

        # 1. Temporal filter: kernel size 15 (covering 150 ms at 100 Hz, matching visual ERP wave cycle)
        self.temp_conv = nn.Conv1d(in_channels, 64, kernel_size=15, padding=7, bias=False)
        self.temp_bn = nn.BatchNorm1d(64)
        self.temp_act = nn.GELU()

        # 2. Spatial mixing: 1x1 conv cross-channel linear combination
        self.spat_conv = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.spat_bn = nn.BatchNorm1d(64)
        self.spat_act = nn.GELU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)  # 55 -> 27

        # 3. Residual temporal stages
        self.stage1 = Conv1DBlock(64, 128, kernel_size=7, stride=2, dropout=dropout)   # 27 -> 14
        self.stage2 = Conv1DBlock(128, 128, kernel_size=5, stride=2, dropout=dropout)  # 14 -> 7
        self.stage3 = Conv1DBlock(128, 256, kernel_size=3, stride=2, dropout=dropout)  # 7 -> 4

        # 4. Multiscale temporal pooling: combine mean (sustained ERP) and max (transient spikes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 5. Non-linear projection head to shared embedding space
        self.proj_head = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x: torch.Tensor, sub_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 17, T) tensor — EEG time series.
        sub_ids : (B,) int tensor, optional — 1-based subject numbers.

        Returns
        -------
        z : (B, proj_dim) tensor — L2-normalized EEG embedding.
        """
        if self.use_subject_adapter and sub_ids is not None:
            adapted = torch.zeros_like(x)
            for s_idx, adapter in enumerate(self.subject_adapters):
                mask = (sub_ids == (s_idx + 1))
                if mask.any():
                    adapted[mask] = adapter(x[mask])
            x = x + adapted

        out = self.temp_act(self.temp_bn(self.temp_conv(x)))
        out = self.spat_act(self.spat_bn(self.spat_conv(out)))
        out = self.pool1(out)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        avg_rep = self.avg_pool(out).flatten(1)
        max_rep = self.max_pool(out).flatten(1)
        pooled = torch.cat([avg_rep, max_rep], dim=1)

        z = self.proj_head(pooled)
        return F.normalize(z, p=2, dim=-1)



class StimulusProjector(nn.Module):
    """MLP projector mapping frozen stimulus features (ViCLIP + AST) to shared latent space.

    Parameters
    ----------
    stim_dim : int, default=1536
        Dimension of the concatenated ViCLIP (768-d) + AST (768-d) representations.
    proj_dim : int, default=256
        Dimensionality of the joint contrastive latent space.
    dropout : float, default=0.2
    """

    def __init__(self, stim_dim: int = 1536, proj_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stim_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, stim_dim) tensor

        Returns
        -------
        z : (B, proj_dim) tensor — L2-normalized stimulus embedding.
        """
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class MultimodalContrastiveModel(nn.Module):
    """Dual-encoder contrastive architecture aligning EEG with multimodal stimuli.

    Parameters
    ----------
    stim_dim : int, default=1536
        Multimodal stimulus feature dimension (ViCLIP + AST).
    proj_dim : int, default=256
        Joint embedding dimensionality.
    init_temp : float, default=0.07
        Initial temperature for InfoNCE softmax scaling.
    """

    def __init__(
        self,
        stim_dim: int = 1536,
        proj_dim: int = 256,
        init_temp: float = 0.07,
        dropout: float = 0.25,
        use_subject_adapter: bool = False,
        num_subjects: int = 6,
    ):
        super().__init__()
        self.use_subject_adapter = use_subject_adapter
        self.eeg_encoder = EEGEncoder(
            in_channels=17,
            proj_dim=proj_dim,
            dropout=dropout,
            use_subject_adapter=use_subject_adapter,
            num_subjects=num_subjects,
        )
        self.stim_projector = StimulusProjector(stim_dim=stim_dim, proj_dim=proj_dim, dropout=dropout)

        # Learnable logit scale (CLIP-style clamp between [ln(1/100), ln(1/0.01)])
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / init_temp))

    def forward(
        self,
        eeg: torch.Tensor,
        stim: torch.Tensor,
        stim_ids: torch.Tensor = None,
        sub_ids: torch.Tensor = None,
    ):
        """Forward pass computing normalized embeddings and symmetric InfoNCE loss.

        Parameters
        ----------
        eeg : (B, 17, T)
        stim : (B, stim_dim)
        stim_ids : (B,) int tensor, optional
            Stimulus IDs for identifying cross-subject identical video collisions (false negatives).
        sub_ids : (B,) int tensor, optional
            Subject IDs (1..6) for subject spatial adapter.

        Returns
        -------
        loss : scalar tensor
        logits : (B, B) similarity matrix scaled by temperature
        """
        z_eeg = self.eeg_encoder(eeg, sub_ids=sub_ids)       # (B, proj_dim)
        z_stim = self.stim_projector(stim)  # (B, proj_dim)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # Cosine similarity matrix scaled by temperature
        logits_eeg_to_stim = logit_scale * torch.matmul(z_eeg, z_stim.T)  # (B, B)
        logits_stim_to_eeg = logits_eeg_to_stim.T

        labels = torch.arange(eeg.size(0), device=eeg.device)

        if stim_ids is not None:
            # Mask out false negatives: cross-subject trials that viewed the exact same stimulus ID
            # off-diagonal pairs where stim_ids[i] == stim_ids[j]
            same_stim = (stim_ids[:, None] == stim_ids[None, :]) & ~torch.eye(
                eeg.size(0), dtype=torch.bool, device=eeg.device
            )
            logits_eeg_to_stim = logits_eeg_to_stim.masked_fill(same_stim, -1e9)
            logits_stim_to_eeg = logits_stim_to_eeg.masked_fill(same_stim.T, -1e9)

        loss_e = F.cross_entropy(logits_eeg_to_stim, labels)
        loss_s = F.cross_entropy(logits_stim_to_eeg, labels)
        loss = 0.5 * (loss_e + loss_s)

        return loss, logits_eeg_to_stim

    @torch.no_grad()
    def compute_similarity(
        self,
        eeg: torch.Tensor,
        stim: torch.Tensor,
        sub_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute cosine similarity matrix between EEG trials and test candidate stimuli.

        Parameters
        ----------
        eeg : (N_eeg, 17, T) tensor
        stim : (N_stim, stim_dim) tensor
        sub_ids : (N_eeg,) int tensor, optional

        Returns
        -------
        sim : (N_eeg, N_stim) cosine similarity matrix in [-1, 1].
        """
        self.eval()
        z_eeg = self.eeg_encoder(eeg, sub_ids=sub_ids)
        z_stim = self.stim_projector(stim)
        return torch.matmul(z_eeg, z_stim.T)
