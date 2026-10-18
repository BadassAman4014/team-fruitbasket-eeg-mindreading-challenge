"""
src/models/contrastive_eeg.py - Deep Contrastive EEG-to-Multimodal Retrieval Model
===================================================================================
Implements WP3 and WP10 of the Contrastive Retrieval Plan:
1. SubjectLayer: Learnable per-subject 17x17 spatial calibration matrix (init: Identity).
2. EEGEncoder: Temporal filter banks + spatial mixing + residual dilated temporal convs + dual mean/max pooling.
3. MultiBranchProjector: Per-modality MLPs for DINOv2 and CLAP/AST with learned gating and modality dropout.
4. info_nce: Symmetric temperature-scaled contrastive loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectLayer(nn.Module):
    """
    Per-subject spatial calibration layer:
      y = W_s @ x + b_s
    where W_s is a 17x17 matrix initialized to Identity, and b_s is a 17x1 bias vector.
    Regularized by 1e-3 * ||W_s - I||_F^2 to penalize deviations from identity.
    """
    def __init__(self, num_subjects=6, num_channels=17):
        super().__init__()
        self.num_subjects = num_subjects
        self.num_channels = num_channels
        
        # Initialize W as Identity for each subject
        init_W = torch.stack([torch.eye(num_channels) for _ in range(num_subjects)])
        self.weights = nn.Parameter(init_W)  # (S, 17, 17)
        self.biases = nn.Parameter(torch.zeros(num_subjects, num_channels, 1))  # (S, 17, 1)

    def forward(self, x, sub_idx):
        """
        x: (B, 17, T)
        sub_idx: (B,) tensor of subject indices (0 to 5)
        """
        W = self.weights[sub_idx]  # (B, 17, 17)
        b = self.biases[sub_idx]   # (B, 17, 1)
        return torch.bmm(W, x) + b

    def penalty(self, sub_idx=None):
        """Returns L2 deviation penalty from identity."""
        I = torch.eye(self.num_channels, device=self.weights.device)
        if sub_idx is not None:
            W = self.weights[sub_idx]
            diff = W - I
        else:
            diff = self.weights - I
        return 1e-3 * torch.mean(diff ** 2)


class ResBlock1D(nn.Module):
    """Residual 1D convolutional block with dilation."""
    def __init__(self, channels, kernel_size=7, dilation=1, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class EEGEncoder(nn.Module):
    """
    Deep Spatiotemporal EEG Encoder:
    1. Temporal Conv: learns multi-frequency filter banks across time.
    2. Spatial Depthwise Conv: mixes spatial channels.
    3. Residual Dilated Convs: captures transient ERPs and long-range dynamics.
    4. Dual Pooling (Mean + Max): captures sustained power and sharp evoked peaks.
    5. Projection to 256-d embedding + L2 normalization.
    """
    def __init__(self, in_channels=17, emb_dim=256, dropout=0.3):
        super().__init__()
        # 1. Temporal filter bank
        self.temp_conv = nn.Conv1d(in_channels, 32, kernel_size=15, padding=7)
        self.bn0 = nn.BatchNorm1d(32)

        # 2. Spatial channel mixing
        self.spatial_conv = nn.Conv1d(32, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # 3. Residual stages with dilations 1, 2, 4
        self.stage1 = ResBlock1D(64, kernel_size=7, dilation=1, dropout=dropout)
        self.stage2 = ResBlock1D(64, kernel_size=7, dilation=2, dropout=dropout)
        self.stage3 = ResBlock1D(64, kernel_size=7, dilation=4, dropout=dropout)

        # 4. Final projection: dual pooling (64 + 64 = 128) -> emb_dim
        self.proj = nn.Sequential(
            nn.Linear(128, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, x):
        """
        x: (B, 17, T)
        Returns: (B, emb_dim) L2-normalized embedding
        """
        h = self.act(self.bn0(self.temp_conv(x)))
        h = self.drop(self.act(self.bn1(self.spatial_conv(h))))
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)

        # Dual temporal pooling: mean + max
        h_mean = torch.mean(h, dim=-1)
        h_max, _ = torch.max(h, dim=-1)
        h_pool = torch.cat([h_mean, h_max], dim=-1)  # (B, 128)

        emb = self.proj(h_pool)
        return F.normalize(emb, p=2, dim=-1)


class MultiBranchProjector(nn.Module):
    """
    Multimodal stimulus projection and fusion network:
    - Independent 2-layer MLPs per stimulus block (DINOv2, CLAP, AST).
    - Learnable softmax fusion weights.
    - Modality dropout (0.15) to prevent over-reliance on a single stream.
    - L2-normalized 256-d embedding.
    """
    def __init__(self, block_dims, emb_dim=256, dropout=0.3, modality_dropout=0.15):
        super().__init__()
        self.block_names = sorted(list(block_dims.keys()))
        self.modality_dropout = modality_dropout
        
        # Per-block projection MLPs
        self.branches = nn.ModuleDict()
        for name in self.block_names:
            in_dim = block_dims[name]
            self.branches[name] = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, emb_dim),
                nn.LayerNorm(emb_dim)
            )

        # Learnable fusion logits
        self.fusion_logits = nn.Parameter(torch.zeros(len(self.block_names)))

    def forward(self, stim_dict, training=True):
        """
        stim_dict: dict of {block_name: (B, in_dim)}
        Returns: (B, emb_dim) L2-normalized multimodal stimulus embedding
        """
        projected = []
        for name in self.block_names:
            h = self.branches[name](stim_dict[name])
            projected.append(h)
        # Stack: (num_blocks, B, emb_dim)
        stacked = torch.stack(projected, dim=0)

        # Softmax weights across blocks: (num_blocks, 1, 1)
        weights = F.softmax(self.fusion_logits, dim=0).view(-1, 1, 1)

        # Modality dropout during training (only if multiple blocks present)
        if training and self.training and len(self.block_names) > 1 and self.modality_dropout > 0:
            mask = (torch.rand(len(self.block_names), 1, 1, device=stacked.device) >= self.modality_dropout).float()
            # Ensure at least one modality remains active
            if mask.sum() == 0:
                mask[torch.randint(len(self.block_names), (1,)).item()] = 1.0
            weights = weights * mask
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        # Fused stimulus representation
        fused = torch.sum(weights * stacked, dim=0)
        return F.normalize(fused, p=2, dim=-1)

    def get_fusion_weights(self):
        """Returns dictionary of learned fusion weights."""
        w = F.softmax(self.fusion_logits, dim=0).detach().cpu().numpy()
        return {name: float(w[idx]) for idx, name in enumerate(self.block_names)}


class ContrastiveRetrievalModel(nn.Module):
    """
    End-to-End Multimodal Contrastive Retrieval Model.
    """
    def __init__(self, block_dims, emb_dim=256, num_subjects=6, dropout=0.3):
        super().__init__()
        self.subject_layer = SubjectLayer(num_subjects=num_subjects, num_channels=17)
        self.eeg_encoder = EEGEncoder(in_channels=17, emb_dim=emb_dim, dropout=dropout)
        self.stim_projector = MultiBranchProjector(block_dims=block_dims, emb_dim=emb_dim, dropout=dropout)
        # Learnable log temperature scale (init 0.07 -> log(1/0.07) ~ 2.659)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_eeg(self, eeg, sub_idx):
        """x: (B, 17, T) -> (B, emb_dim)"""
        calibrated_eeg = self.subject_layer(eeg, sub_idx)
        return self.eeg_encoder(calibrated_eeg)

    def encode_stim(self, stim_dict, training=False):
        """stim_dict: dict -> (B, emb_dim)"""
        return self.stim_projector(stim_dict, training=training)

    def forward(self, eeg, sub_idx, stim_dict):
        e_emb = self.encode_eeg(eeg, sub_idx)
        s_emb = self.encode_stim(stim_dict, training=True)
        return e_emb, s_emb


def info_nce(e, s, log_scale, vid_ids=None):
    """
    Symmetric InfoNCE loss:
      scale = exp(log_scale).clamp(max=100.0)
      sim = scale * (e @ s.T)
    """
    scale = log_scale.exp().clamp(max=100.0)
    sim = scale * torch.mm(e, s.t())  # (B, B)

    if vid_ids is not None:
        # If any duplicate video IDs exist in batch, mask non-diagonal matching IDs
        same = (vid_ids.unsqueeze(1) == vid_ids.unsqueeze(0))
        same.fill_diagonal_(False)
        sim = sim.masked_fill(same, float('-inf'))

    targets = torch.arange(e.size(0), device=e.device)
    loss_e2s = F.cross_entropy(sim, targets)
    loss_s2e = F.cross_entropy(sim.t(), targets)
    return 0.5 * (loss_e2s + loss_s2e)
