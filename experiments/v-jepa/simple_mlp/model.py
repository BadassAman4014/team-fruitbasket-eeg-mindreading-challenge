"""Minimal CLIP-style contrastive model for EEG ↔ video/audio retrieval.

This is the deliberately *simple* counterpart to src/models/contrastive_eeg.py.
It implements exactly what the challenge brief suggests — "train a small MLP
using a contrastive (CLIP-style) loss" — and nothing more:

    EEG (flattened 17×T)  ── small MLP ──►  z_eeg   (L2-normalized)
    stim feats (1280-d)   ── small MLP ──►  z_stim  (L2-normalized)
    loss = symmetric InfoNCE over the batch (correct pair = positive)

Deliberately omitted (vs the heavy Phase 2 model), so this stays a clean baseline:
  - no 1D-CNN EEG encoder (both encoders are plain MLPs)
  - no per-subject spatial adapters
  - no data augmentation
  - no multi-seed ensembling / CSLS / Ridge blending

The point of this file is to test whether the literal "small MLP" matches the
much heavier stack. Keep it minimal on purpose.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """A small MLP projecting a flat input vector to the shared latent space.

    One hidden layer, LayerNorm + GELU + dropout, then a linear projection.
    Output is L2-normalized so dot products are cosine similarities.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (flattened EEG, or stimulus feature dim).
    proj_dim : int, default=256
        Shared embedding dimension.
    hidden_dim : int, default=512
        Width of the single hidden layer.
    dropout : float, default=0.3
        Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_dim) tensor

        Returns
        -------
        z : (B, proj_dim) tensor — L2-normalized embedding.
        """
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class SimpleContrastiveModel(nn.Module):
    """Dual small-MLP encoders aligned by a CLIP-style symmetric InfoNCE loss.

    Parameters
    ----------
    eeg_dim : int
        Flattened EEG dimension (channels × window length).
    stim_dim : int, default=1280
        Stimulus feature dimension (e.g. ViCLIP 768 + CLAP 512 = 1280).
    proj_dim : int, default=256
        Shared embedding dimension.
    hidden_dim : int, default=512
        Hidden width for both encoders.
    dropout : float, default=0.3
    init_temp : float, default=0.07
        Initial InfoNCE temperature (learnable, CLIP-style logit scale).
    """

    def __init__(
        self,
        eeg_dim: int,
        stim_dim: int = 1280,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        init_temp: float = 0.07,
    ):
        super().__init__()
        self.eeg_encoder = MLPEncoder(eeg_dim, proj_dim, hidden_dim, dropout)
        self.stim_encoder = MLPEncoder(stim_dim, proj_dim, hidden_dim, dropout)

        # Learnable logit scale (CLIP-style); clamp to <=100 at use time.
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / init_temp))

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """Flatten (B, C, T) → (B, C*T) if needed, then project."""
        if eeg.dim() == 3:
            eeg = eeg.flatten(1)
        return self.eeg_encoder(eeg)

    def encode_stim(self, stim: torch.Tensor) -> torch.Tensor:
        return self.stim_encoder(stim)

    def forward(
        self,
        eeg: torch.Tensor,
        stim: torch.Tensor,
        stim_ids: torch.Tensor = None,
    ):
        """Compute embeddings and the symmetric InfoNCE loss over the batch.

        Parameters
        ----------
        eeg : (B, C, T) or (B, C*T) tensor
        stim : (B, stim_dim) tensor
        stim_ids : (B,) int tensor, optional
            If given, off-diagonal pairs that share the same stimulus ID (e.g.
            two subjects who saw the same video) are masked out of the negatives,
            so they are not wrongly penalized as hard negatives.

        Returns
        -------
        loss : scalar tensor
        logits : (B, B) similarity matrix scaled by temperature
        """
        z_eeg = self.encode_eeg(eeg)    # (B, proj_dim)
        z_stim = self.encode_stim(stim)  # (B, proj_dim)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_e2s = logit_scale * (z_eeg @ z_stim.T)  # (B, B)
        logits_s2e = logits_e2s.T

        labels = torch.arange(eeg.size(0), device=eeg.device)

        if stim_ids is not None:
            same = (stim_ids[:, None] == stim_ids[None, :]) & ~torch.eye(
                eeg.size(0), dtype=torch.bool, device=eeg.device
            )
            logits_e2s = logits_e2s.masked_fill(same, -1e9)
            logits_s2e = logits_s2e.masked_fill(same.T, -1e9)

        loss_e = F.cross_entropy(logits_e2s, labels)
        loss_s = F.cross_entropy(logits_s2e, labels)
        loss = 0.5 * (loss_e + loss_s)
        return loss, logits_e2s

    @torch.no_grad()
    def compute_similarity(self, eeg: torch.Tensor, stim: torch.Tensor) -> torch.Tensor:
        """Cosine similarity matrix between EEG trials and candidate stimuli.

        Parameters
        ----------
        eeg : (N_eeg, C, T) or (N_eeg, C*T) tensor
        stim : (N_stim, stim_dim) tensor

        Returns
        -------
        sim : (N_eeg, N_stim) cosine similarity matrix in [-1, 1].
        """
        self.eval()
        z_eeg = self.encode_eeg(eeg)
        z_stim = self.encode_stim(stim)
        return z_eeg @ z_stim.T
