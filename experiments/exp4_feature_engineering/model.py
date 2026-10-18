"""Model architecture for Exp4 + Feature Engineering.

Exact state_dict compatibility with Exp4 TunedMLPRetriever, but generalized to support:
- Arbitrary channel counts (e.g. 17, 12, 8 channels)
- Pre-extracted 2D feature matrices (B, feature_dim) or 3D EEG tensors (B, C, T)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SUBJECTS = 6


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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DinoPool(nn.Module):
    def __init__(self, mode: str, feature_dim: int, frames: int, hidden_dim: int):
        super().__init__()
        self.mode = mode
        self.frames = frames
        if mode == "attention":
            self.content = nn.Linear(feature_dim, hidden_dim)
            self.position = nn.Parameter(torch.zeros(frames, hidden_dim))
            nn.init.normal_(self.position, std=0.02)
            self.score = nn.Linear(hidden_dim, 1, bias=False)

    def weights(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 3 or frames.shape[1] != self.frames:
            raise ValueError(f"Expected DINO (B,{self.frames},D), got {tuple(frames.shape)}")
        if self.mode == "mean":
            return frames.new_full((len(frames), self.frames), 1.0 / self.frames)
        logits = self.score(torch.tanh(self.content(frames) + self.position[None])).squeeze(-1)
        return F.softmax(logits, dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        weights = self.weights(frames)
        return torch.sum(weights.unsqueeze(-1) * frames, dim=1)


class TunedMLPRetriever(nn.Module):
    """Drop-in drop-compatible replacement for Exp4 TunedMLPRetriever."""
    def __init__(
        self,
        eeg_dim: int,
        dino_dim: int,
        audio_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        temperature: float,
        fusion: str,
        modality_dropout: float,
        subject_affine: bool,
        dino_pooling: str,
        dino_frames: int,
        attention_hidden_dim: int,
    ):
        super().__init__()
        self.fusion = fusion
        self.modality_dropout = modality_dropout
        self.subject_affine_enabled = subject_affine
        self.log_scale = nn.Parameter(torch.tensor(1.0 / temperature).log())
        self.dino_pool = DinoPool(
            dino_pooling, dino_dim, dino_frames, attention_hidden_dim
        )
        self.eeg_mlp = ProjectionMLP(eeg_dim, hidden_dim, embedding_dim, dropout)
        if fusion == "concat":
            self.concat_mlp = ProjectionMLP(
                dino_dim + audio_dim, hidden_dim, embedding_dim, dropout
            )
        else:
            self.dino_mlp = ProjectionMLP(dino_dim, hidden_dim, embedding_dim, dropout)
            self.audio_mlp = ProjectionMLP(audio_dim, hidden_dim, embedding_dim, dropout)
            self.fusion_logits = nn.Parameter(torch.zeros(2))
        if subject_affine:
            self.subject_scale = nn.Parameter(torch.ones(N_SUBJECTS, 1, 1))
            self.subject_bias = nn.Parameter(torch.zeros(N_SUBJECTS, 1, 1))

    def encode_eeg(self, eeg: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        if self.subject_affine_enabled and eeg.ndim == 3:
            index = subject_ids.long() - 1
            eeg = eeg * self.subject_scale[index] + self.subject_bias[index]
        flat_eeg = eeg.flatten(start_dim=1) if eeg.ndim > 2 else eeg
        return F.normalize(self.eeg_mlp(flat_eeg), p=2, dim=-1)

    def encode_stimulus(
        self, dino_frames: torch.Tensor, audio: torch.Tensor
    ) -> torch.Tensor:
        pooled_dino = self.dino_pool(dino_frames)
        if self.fusion == "concat":
            return F.normalize(
                self.concat_mlp(torch.cat((pooled_dino, audio), dim=1)), p=2, dim=-1
            )
        projected = torch.stack(
            (self.dino_mlp(pooled_dino), self.audio_mlp(audio)), dim=1
        )
        weights = F.softmax(self.fusion_logits, dim=0).expand(len(projected), -1)
        if self.training and self.modality_dropout > 0:
            keep = torch.rand_like(weights) >= self.modality_dropout
            empty = ~keep.any(dim=1)
            if empty.any():
                replacement = torch.randint(2, (int(empty.sum()),), device=keep.device)
                keep[empty] = False
                keep[empty, replacement] = True
            weights = weights * keep
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = torch.sum(weights.unsqueeze(-1) * projected, dim=1)
        return F.normalize(fused, p=2, dim=-1)

    def forward(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor,
        dino_frames: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_eeg(eeg, subject_ids), self.encode_stimulus(dino_frames, audio)

    def affine_penalty(self) -> torch.Tensor:
        if not self.subject_affine_enabled:
            return self.log_scale.new_zeros(())
        return (self.subject_scale - 1).square().mean() + self.subject_bias.square().mean()

    def fusion_weights(self) -> dict[str, float]:
        if self.fusion == "concat":
            return {"concat": 1.0}
        weights = F.softmax(self.fusion_logits.detach(), dim=0).cpu().numpy()
        return {"dino": float(weights[0]), "audio": float(weights[1])}


def contrastive_loss(
    eeg: torch.Tensor,
    stimulus: torch.Tensor,
    log_scale: torch.Tensor,
    video_ids: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if eeg.shape != stimulus.shape or eeg.ndim != 2:
        raise ValueError("EEG and stimulus embeddings must have equal (B,D) shapes")
    if video_ids.shape != (len(eeg),):
        raise ValueError("video_ids must have shape (B,)")
    similarity = log_scale.exp().clamp(max=100.0) * (eeg @ stimulus.T)
    positives = video_ids[:, None].eq(video_ids[None, :])
    if mode == "standard":
        diagonal = torch.eye(len(eeg), dtype=torch.bool, device=eeg.device)
        similarity = similarity.masked_fill(positives & ~diagonal, -torch.inf)
        targets = torch.arange(len(eeg), device=eeg.device)
        return 0.5 * (
            F.cross_entropy(similarity, targets)
            + F.cross_entropy(similarity.T, targets)
        )
    if mode != "multipositive":
        raise ValueError(f"Unknown loss: {mode}")

    def direction(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = torch.logsumexp(scores.masked_fill(~mask, -torch.inf), dim=1)
        denominator = torch.logsumexp(scores, dim=1)
        return (denominator - numerator).mean()

    return 0.5 * (direction(similarity, positives) + direction(similarity.T, positives.T))
