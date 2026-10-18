"""Faithful port of the exp4 recipe for FRAME-LEVEL V-JEPA2 (N,8,D) + audio.

Mirrors exp4_tuned_mlp_dino_audio/train.py:
  - DinoPool: mean-pool the 8 video sub-clip embeddings → one video vector.
  - Fusion = CONCAT: [pooled_video | audio] → single ProjectionMLP.
  - ProjectionMLP = LayerNorm → Linear → GELU → Dropout → Linear → LayerNorm.
  - MULTIPOSITIVE loss = logsumexp(positives) − logsumexp(all), symmetric.
  - L2-normalize embeddings after projection.
  - Learnable log temperature (init log(1/0.07)).
  - Modality dropout: in CONCAT mode exp4 has no fusion branches, so modality
    dropout is applied by zeroing the pooled-video or audio sub-vector before
    concatenation (per-sample, p split 50/50). Kept for parity with the flag.

Video is pre-extracted as 8 sub-clip embeddings (N,8,1024); audio is CLAP (N,512).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RecipeContrastiveModel(nn.Module):
    """exp4 concat-fusion retriever for frame-level video + audio.

    Parameters
    ----------
    eeg_dim : int            flattened EEG dim (17 * window)
    video_dim : int          per-sub-clip video embedding dim (V-JEPA2 = 1024)
    audio_dim : int          audio embedding dim (CLAP = 512)
    n_frames : int           number of video sub-clips (8)
    proj_dim, hidden_dim, dropout, init_temp, modality_dropout : as exp4
    """

    def __init__(
        self,
        eeg_dim: int,
        video_dim: int,
        audio_dim: int,
        n_frames: int = 8,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        init_temp: float = 0.07,
        modality_dropout: float = 0.15,
    ):
        super().__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.n_frames = n_frames
        self.modality_dropout = modality_dropout

        self.eeg_mlp = ProjectionMLP(eeg_dim, hidden_dim, proj_dim, dropout)
        # CONCAT fusion: single MLP over [pooled_video | audio].
        self.concat_mlp = ProjectionMLP(video_dim + audio_dim, hidden_dim, proj_dim, dropout)
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / init_temp)), dtype=torch.float32))

    def param_groups(self, weight_decay: float):
        """exp4: decay only weight matrices (ndim>=2), not biases/norms/temperature."""
        decayed, no_decay = [], []
        for name, p in self.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if p.ndim < 2 or leaf == "log_scale":
                no_decay.append(p)
            else:
                decayed.append(p)
        groups = []
        if decayed:
            groups.append({"params": decayed, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.dim() == 3:
            eeg = eeg.flatten(1)
        return F.normalize(self.eeg_mlp(eeg), p=2, dim=-1)

    def encode_stim(self, video_frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """video_frames: (B, n_frames, video_dim); audio: (B, audio_dim)."""
        pooled_video = video_frames.mean(dim=1)  # DinoPool mean → (B, video_dim)
        if self.training and self.modality_dropout > 0:
            B = pooled_video.size(0)
            r = torch.rand(B, device=pooled_video.device)
            drop = r < self.modality_dropout
            drop_v = drop & (torch.rand(B, device=pooled_video.device) < 0.5)
            drop_a = drop & ~drop_v
            if drop_v.any():
                pooled_video = pooled_video.clone(); pooled_video[drop_v] = 0.0
            if drop_a.any():
                audio = audio.clone(); audio[drop_a] = 0.0
        fused = torch.cat([pooled_video, audio], dim=1)  # (B, video_dim+audio_dim)
        return F.normalize(self.concat_mlp(fused), p=2, dim=-1)

    def forward(self, eeg, video_frames, audio, stim_ids=None):
        z_eeg = self.encode_eeg(eeg)
        z_stim = self.encode_stim(video_frames, audio)
        sim = self.log_scale.exp().clamp(max=100.0) * (z_eeg @ z_stim.T)
        B = eeg.size(0)
        if stim_ids is not None:
            pos = stim_ids[:, None].eq(stim_ids[None, :])
        else:
            pos = torch.eye(B, dtype=torch.bool, device=eeg.device)

        def direction(scores, mask):
            num = torch.logsumexp(scores.masked_fill(~mask, -torch.inf), dim=1)
            den = torch.logsumexp(scores, dim=1)
            return (den - num).mean()

        loss = 0.5 * (direction(sim, pos) + direction(sim.T, pos.T))
        return loss, sim

    @torch.no_grad()
    def compute_similarity(self, eeg, video_frames, audio):
        self.eval()
        z_eeg = self.encode_eeg(eeg)
        z_stim = self.encode_stim(video_frames, audio)
        return z_eeg @ z_stim.T
