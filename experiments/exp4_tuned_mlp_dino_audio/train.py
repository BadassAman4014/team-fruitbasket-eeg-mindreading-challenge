"""Tuned pooled-EEG MLP retrieval with frozen frame-level DINO and audio caches.

The experiment is intentionally self-contained under exp4. Model checkpoints are
selected only by IDs 901..1000; optional public test labels are read only after
all checkpoint and CSLS selection has finished.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.signal import decimate


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_path(data_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else data_root / resolved


def load_training_eeg(data_root: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    path = data_root / "trn" / f"sub-{subject:02d}_trn.npz"
    with np.load(path) as archive:
        ids = np.array(sorted((int(k) for k in archive.files)), dtype=np.int64)
        raw = np.stack([archive[str(int(video_id))] for video_id in ids])
    eeg = decimate(raw, 5, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)
    if eeg.ndim != 3 or eeg.shape[1] != 17 or eeg.shape[2] < 60:
        raise ValueError(f"Expected downsampled EEG shape (N,17,>=60), got {eeg.shape}")
    return ids, eeg


def load_test_eeg(data_root: Path, subject: int) -> np.ndarray:
    path = data_root / "tst" / f"sub-{subject:02d}_tst.npy"
    raw = np.load(path)
    eeg = decimate(raw, 5, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)
    if eeg.shape != (102, 17, 300):
        raise ValueError(f"Expected test EEG shape (102,17,300), got {eeg.shape}")
    return eeg


def load_audio_map(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Audio feature cache not found: {path}")
    result: dict[int, np.ndarray] = {}
    with np.load(path) as cache:
        for key in cache.files:
            try:
                video_id = int(key.removeprefix("stim_"))
            except ValueError:
                continue
            result[video_id] = np.asarray(cache[key], dtype=np.float32).reshape(-1)
    if not result:
        raise ValueError(f"No 'stim_<video_id>' entries found in {path}")
    return result


def stack_features(feature_map: dict[int, np.ndarray], ids: np.ndarray, name: str) -> np.ndarray:
    missing = [int(video_id) for video_id in ids if int(video_id) not in feature_map]
    if missing:
        raise KeyError(f"{name} is missing {len(missing)} requested IDs: {missing[:10]}")
    output = np.stack([feature_map[int(video_id)] for video_id in ids]).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f"{name} contains non-finite values")
    return output


def fit_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(features.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    return mean, std


def apply_normalizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def ranked_from_scores(S: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    order = np.argsort(-S, axis=1)
    return np.array(candidate_ids)[order]


def evaluate_ranking(ranked: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    n_trials = len(gt)
    top1 = float((ranked[:, 0] == gt).mean())
    top5 = float((ranked[:, :5] == gt[:, None]).any(axis=1).mean())
    ranks = []
    reciprocal_ranks = []
    for i in range(n_trials):
        match = np.where(ranked[i] == gt[i])[0]
        rank = int(match[0] + 1) if len(match) > 0 else ranked.shape[1]
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
    return {
        "top1": top1,
        "top5": top5,
        "mrr": float(np.mean(reciprocal_ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "ranks": np.array(ranks),
    }

TRAIN_IDS = np.arange(1, 901, dtype=np.int64)
HELDOUT_IDS = np.arange(901, 1001, dtype=np.int64)
TEST_IDS = np.arange(1001, 1103, dtype=np.int64)
PUBLIC_SUBJECTS = frozenset((1, 2, 3, 4))
N_SUBJECTS = 6
N_CHANNELS = 17
WINDOW = slice(5, 60)
_FRAME_KEY = re.compile(r"^(\d+)_f(\d+)$")


@dataclass(frozen=True)
class Config:
    data_root: str
    output_root: str
    run_name: str
    subjects: tuple[int, ...] = (1, 2, 3, 4)
    prediction_subjects: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    seeds: tuple[int, ...] = (42, 123, 999)
    audio_model: str = "clap"
    dino_train_cache: str = "features_dinov2/consolidated_encoder.layer.11.npz"
    dino_test_cache: str = "features_test_dinov2/consolidated_encoder.layer.11.npz"
    hidden_dim: int = 512
    embedding_dim: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    dropout: float = 0.3
    temperature: float = 0.07
    batch_size: int = 128
    epochs: int = 30
    loss: str = "multipositive"
    fusion: str = "separate"
    modality_dropout: float = 0.15
    subject_affine: bool = False
    subject_affine_reg: float = 1e-4
    dino_pooling: str = "mean"
    dino_frames: int = 8
    attention_hidden_dim: int = 64
    anti_hub: str = "csls"
    csls_k: int = 10
    csls_lambda: str = "auto"
    device: str = "auto"
    num_workers: int = 0
    predict_test: bool = False
    report_public_subjects: bool = False
    final_full_fit: bool = False
    smoke: bool = False
    self_test: bool = False


def parse_csv_ints(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "exp4_tuned_mlp_dino_audio" / "runs",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--subjects", type=parse_csv_ints, default=(1, 2, 3, 4))
    parser.add_argument(
        "--prediction-subjects", type=parse_csv_ints, default=(1, 2, 3, 4, 5, 6)
    )
    parser.add_argument("--seeds", type=parse_csv_ints, default=(42, 123, 999))
    parser.add_argument("--audio-model", choices=("clap", "ast"), default="clap")
    parser.add_argument(
        "--dino-train-cache",
        default="features_dinov2/consolidated_encoder.layer.11.npz",
    )
    parser.add_argument(
        "--dino-test-cache",
        default="features_test_dinov2/consolidated_encoder.layer.11.npz",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--loss", choices=("standard", "multipositive"), default="multipositive"
    )
    parser.add_argument("--fusion", choices=("concat", "separate"), default="separate")
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--subject-affine", action="store_true")
    parser.add_argument("--subject-affine-reg", type=float, default=1e-4)
    parser.add_argument("--dino-pooling", choices=("mean", "attention"), default="mean")
    parser.add_argument("--dino-frames", type=int, default=8)
    parser.add_argument("--attention-hidden-dim", type=int, default=64)
    parser.add_argument("--anti-hub", choices=("none", "csls"), default="csls")
    parser.add_argument("--csls-k", type=int, default=10)
    parser.add_argument(
        "--csls-lambda",
        default="auto",
        help="'auto' tests mild values 0.10..0.30 using heldout IDs only.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument(
        "--report-public-subjects",
        action="store_true",
        help="After selection, report labels for prediction subjects 1..4 only.",
    )
    parser.add_argument(
        "--final-full-fit",
        action="store_true",
        help="Select epoch + CSLS on 1-900/heldout, then retrain each seed on ALL "
        "1-1000 stimuli for its selected epoch count, and use those full-fit models "
        "for every test prediction. Matches the simple_mlp final-fit protocol.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic model/loss forward-backward checks; load no data.",
    )
    args = parser.parse_args()

    for name in ("subjects", "prediction_subjects"):
        values = getattr(args, name)
        invalid = [value for value in values if value not in range(1, N_SUBJECTS + 1)]
        if invalid:
            parser.error(f"--{name.replace('_', '-')} values must be 1..6; got {invalid}")
        if len(set(values)) != len(values):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    invalid_seeds = [seed for seed in args.seeds if not 0 <= seed <= 2**32 - 1]
    if invalid_seeds:
        parser.error(f"--seeds values must be in [0, 2**32-1]; got {invalid_seeds}")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for contrastive learning")
    if args.dino_frames != 8:
        parser.error("--dino-frames must be 8 for the repository's fixed eight-frame caches")
    positive = {
        "hidden-dim": args.hidden_dim,
        "embedding-dim": args.embedding_dim,
        "lr": args.lr,
        "temperature": args.temperature,
        "batch-size": args.batch_size,
        "epochs": args.epochs,
        "dino-frames": args.dino_frames,
        "attention-hidden-dim": args.attention_hidden_dim,
        "csls-k": args.csls_k,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.weight_decay < 0 or args.subject_affine_reg < 0:
        parser.error("regularization coefficients must be non-negative")
    if not 0 <= args.dropout < 1 or not 0 <= args.modality_dropout < 1:
        parser.error("dropout values must be in [0, 1)")
    if args.report_public_subjects and not args.predict_test:
        parser.error("--report-public-subjects requires --predict-test")
    if args.csls_lambda != "auto":
        try:
            csls_value = float(args.csls_lambda)
        except ValueError:
            parser.error("--csls-lambda must be 'auto' or a non-negative float")
        if csls_value < 0:
            parser.error("--csls-lambda must be non-negative")

    run_name = args.run_name or (
        f"tuned-mlp-{args.audio_model}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    )
    return Config(
        data_root=str(args.data_root.resolve()),
        output_root=str(args.output_root.resolve()),
        run_name=run_name,
        subjects=args.subjects,
        prediction_subjects=args.prediction_subjects,
        seeds=args.seeds,
        audio_model=args.audio_model,
        dino_train_cache=args.dino_train_cache,
        dino_test_cache=args.dino_test_cache,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        temperature=args.temperature,
        batch_size=args.batch_size,
        epochs=args.epochs,
        loss=args.loss,
        fusion=args.fusion,
        modality_dropout=args.modality_dropout,
        subject_affine=args.subject_affine,
        subject_affine_reg=args.subject_affine_reg,
        dino_pooling=args.dino_pooling,
        dino_frames=args.dino_frames,
        attention_hidden_dim=args.attention_hidden_dim,
        anti_hub=args.anti_hub,
        csls_k=args.csls_k,
        csls_lambda=args.csls_lambda,
        device=args.device,
        num_workers=args.num_workers,
        predict_test=args.predict_test,
        report_public_subjects=args.report_public_subjects,
        final_full_fit=args.final_full_fit,
        smoke=args.smoke,
        self_test=args.self_test,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_trials(eeg: np.ndarray) -> np.ndarray:
    if eeg.ndim != 3 or eeg.shape[1:] != (N_CHANNELS, WINDOW.stop - WINDOW.start):
        raise ValueError(f"Expected EEG (N,17,55), got {eeg.shape}")
    mean = eeg.mean(axis=(1, 2), keepdims=True)
    std = np.maximum(eeg.std(axis=(1, 2), keepdims=True), 1e-6)
    result = ((eeg - mean) / std).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Normalized EEG contains non-finite values")
    return result


def load_pooled_eeg(
    data_root: Path, subjects: tuple[int, ...], first_id: int, last_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    eeg_rows: list[np.ndarray] = []
    id_rows: list[np.ndarray] = []
    subject_rows: list[np.ndarray] = []
    counts: dict[int, int] = {}
    for subject in subjects:
        ids, eeg = load_training_eeg(data_root, subject)
        keep = (ids >= first_id) & (ids <= last_id)
        selected_ids = ids[keep]
        if len(selected_ids) == 0:
            raise ValueError(f"Subject {subject} has no EEG in IDs {first_id}..{last_id}")
        selected = normalize_trials(eeg[keep, :, WINDOW])
        eeg_rows.append(selected)
        id_rows.append(selected_ids)
        subject_rows.append(np.full(len(selected_ids), subject, dtype=np.int64))
        counts[subject] = len(selected_ids)
    return (
        np.concatenate(eeg_rows),
        np.concatenate(id_rows),
        np.concatenate(subject_rows),
        counts,
    )


def load_dino_frames(
    path: Path, required_ids: np.ndarray, frame_count: int
) -> dict[int, np.ndarray]:
    """Load exact f00..fNN frame vectors for every requested video ID."""
    if not path.exists():
        raise FileNotFoundError(
            f"DINO cache not found: {path}. This experiment never extracts DINO online."
        )
    required = set(int(value) for value in required_ids)
    grouped: dict[int, dict[int, np.ndarray]] = {video_id: {} for video_id in required}
    feature_dim: int | None = None
    with np.load(path) as cache:
        for key in cache.files:
            match = _FRAME_KEY.fullmatch(key)
            if match is None:
                continue
            video_id, frame_id = map(int, match.groups())
            if video_id not in required:
                continue
            if frame_id in grouped[video_id]:
                raise ValueError(f"Duplicate DINO frame {video_id}_f{frame_id:02d} in {path}")
            vector = np.asarray(cache[key], dtype=np.float32).reshape(-1)
            if feature_dim is None:
                feature_dim = len(vector)
            if len(vector) != feature_dim:
                raise ValueError(
                    f"DINO dimension mismatch at {key}: {len(vector)} != {feature_dim}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"DINO frame {key} contains non-finite values")
            grouped[video_id][frame_id] = vector
    expected = set(range(frame_count))
    problems: list[str] = []
    for video_id in sorted(required):
        actual = set(grouped[video_id])
        if actual != expected:
            problems.append(
                f"{video_id}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    if problems:
        raise ValueError(
            f"DINO cache {path} does not have exactly frames 0..{frame_count - 1} "
            f"for {len(problems)} required IDs; examples: {problems[:5]}"
        )
    return {
        video_id: np.stack([grouped[video_id][i] for i in range(frame_count)]).astype(
            np.float32
        )
        for video_id in sorted(required)
    }


def stack_dino_frames(
    feature_map: dict[int, np.ndarray], ids: np.ndarray, name: str
) -> np.ndarray:
    missing = [int(value) for value in ids if int(value) not in feature_map]
    if missing:
        raise KeyError(f"{name} missing requested IDs: {missing[:10]}")
    result = np.stack([feature_map[int(value)] for value in ids]).astype(np.float32)
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite (N,frames,features), got {result.shape}")
    return result


def index_by_ids(values: np.ndarray, source_ids: np.ndarray, requested: np.ndarray) -> np.ndarray:
    index = {int(video_id): row for row, video_id in enumerate(source_ids)}
    missing = [int(video_id) for video_id in requested if int(video_id) not in index]
    if missing:
        raise KeyError(f"Feature matrix missing IDs: {missing[:10]}")
    return np.stack([values[index[int(video_id)]] for video_id in requested]).astype(np.float32)


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
            self.subject_scale = nn.Parameter(torch.ones(N_SUBJECTS, N_CHANNELS, 1))
            self.subject_bias = nn.Parameter(torch.zeros(N_SUBJECTS, N_CHANNELS, 1))
        self.log_scale = nn.Parameter(
            torch.tensor(float(np.log(1.0 / temperature)), dtype=torch.float32)
        )

    def encode_eeg(self, eeg: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != N_CHANNELS:
            raise ValueError(f"Expected EEG (B,17,T), got {tuple(eeg.shape)}")
        if subject_ids.shape != (len(eeg),):
            raise ValueError("subject_ids must have shape (B,)")
        if torch.any((subject_ids < 1) | (subject_ids > N_SUBJECTS)):
            raise ValueError("subject_ids must be 1..6")
        if self.subject_affine_enabled:
            index = subject_ids.long() - 1
            eeg = eeg * self.subject_scale[index] + self.subject_bias[index]
        return F.normalize(self.eeg_mlp(eeg.flatten(start_dim=1)), p=2, dim=-1)

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


def make_model(config: Config, dino_dim: int, audio_dim: int, device: torch.device):
    return TunedMLPRetriever(
        eeg_dim=N_CHANNELS * (WINDOW.stop - WINDOW.start),
        dino_dim=dino_dim,
        audio_dim=audio_dim,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        temperature=config.temperature,
        fusion=config.fusion,
        modality_dropout=config.modality_dropout,
        subject_affine=config.subject_affine,
        dino_pooling=config.dino_pooling,
        dino_frames=config.dino_frames,
        attention_hidden_dim=config.attention_hidden_dim,
    ).to(device)


def encode_eeg(
    model: TunedMLPRetriever,
    eeg: np.ndarray,
    subjects: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if len(eeg) != len(subjects):
        raise ValueError("EEG and subject arrays must align")
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(eeg), batch_size):
            x = torch.from_numpy(eeg[start : start + batch_size]).to(device)
            s = torch.from_numpy(subjects[start : start + batch_size]).to(device)
            output.append(model.encode_eeg(x, s).cpu().numpy())
    return np.concatenate(output)


def encode_stimulus(
    model: TunedMLPRetriever,
    dino: np.ndarray,
    audio: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if len(dino) != len(audio):
        raise ValueError("DINO and audio arrays must align")
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dino), batch_size):
            d = torch.from_numpy(dino[start : start + batch_size]).to(device)
            a = torch.from_numpy(audio[start : start + batch_size]).to(device)
            output.append(model.encode_stimulus(d, a).cpu().numpy())
    return np.concatenate(output)


def raw_scores(
    model: TunedMLPRetriever,
    eeg: np.ndarray,
    subjects: np.ndarray,
    candidate_dino: np.ndarray,
    candidate_audio: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    query = encode_eeg(model, eeg, subjects, device, batch_size)
    candidates = encode_stimulus(
        model, candidate_dino, candidate_audio, device, batch_size
    )
    scores = (query @ candidates.T).astype(np.float32)
    if scores.shape != (len(eeg), len(candidate_dino)):
        raise AssertionError("Unexpected score shape")
    return scores


def mild_csls(scores: np.ndarray, k: int, strength: float) -> np.ndarray:
    if scores.ndim != 2 or min(scores.shape) == 0 or k <= 0:
        raise ValueError("CSLS requires a non-empty score matrix and positive k")
    effective_k = min(k, *scores.shape)
    query_density = np.partition(scores, -effective_k, axis=1)[:, -effective_k:].mean(
        axis=1, keepdims=True
    )
    candidate_density = np.partition(scores, -effective_k, axis=0)[
        -effective_k:, :
    ].mean(axis=0, keepdims=True)
    return (scores - strength * (query_density + candidate_density)).astype(np.float32)


def metrics_from_scores(
    scores: np.ndarray, candidate_ids: np.ndarray, query_ids: np.ndarray
) -> dict[str, Any]:
    if scores.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError("Scores, query IDs, and candidate IDs do not align")
    result = evaluate_ranking(ranked_from_scores(scores, candidate_ids), query_ids)
    result["ranks"] = result["ranks"].tolist()
    return result


def choose_csls_lambda(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    query_ids: np.ndarray,
    k: int,
    requested: str,
) -> tuple[float, dict[str, Any]]:
    values = [float(requested)] if requested != "auto" else [0.10, 0.15, 0.20, 0.25, 0.30]
    choices: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for value in values:
        metrics = metrics_from_scores(mild_csls(scores, k, value), candidate_ids, query_ids)
        key = (
            float(metrics["top1"]),
            float(metrics["top5"]),
            float(metrics["mrr"]),
            -float(metrics["median_rank"]),
        )
        choices.append((key, value, metrics))
    _, value, metrics = max(choices, key=lambda item: item[0])
    return value, metrics


def attention_summary(
    model: TunedMLPRetriever, dino: np.ndarray, device: torch.device, batch_size: int
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "mode": model.dino_pool.mode,
        "frame_count": model.dino_pool.frames,
        "frame_ids": list(range(model.dino_pool.frames)),
        "cache_features_frozen": True,
    }
    if model.dino_pool.mode == "attention":
        rows: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(dino), batch_size):
                batch = torch.from_numpy(dino[start : start + batch_size]).to(device)
                rows.append(model.dino_pool.weights(batch).cpu().numpy())
        all_weights = np.concatenate(rows)
        metadata["mean_heldout_frame_weights"] = all_weights.mean(axis=0).tolist()
        metadata["mean_attention_entropy"] = float(
            np.mean(-np.sum(all_weights * np.log(np.maximum(all_weights, 1e-12)), axis=1))
        )
    return metadata


def train_seed(
    config: Config,
    seed: int,
    train_eeg: np.ndarray,
    train_subjects: np.ndarray,
    train_ids: np.ndarray,
    train_dino: np.ndarray,
    train_audio: np.ndarray,
    heldout_eeg: np.ndarray,
    heldout_subjects: np.ndarray,
    heldout_query_ids: np.ndarray,
    heldout_dino: np.ndarray,
    heldout_audio: np.ndarray,
    device: torch.device,
    fixed_epochs: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    # fixed_epochs set => FINAL full-fit mode: train exactly that many epochs on the
    # supplied data with NO heldout evaluation or per-epoch selection, return the last
    # epoch. Otherwise, select the best epoch on the heldout signal as usual.
    seed_everything(seed)
    model = make_model(config, train_dino.shape[2], train_audio.shape[1], device)
    dataset = TensorDataset(
        torch.from_numpy(train_eeg),
        torch.from_numpy(train_subjects),
        torch.from_numpy(train_ids),
        torch.from_numpy(train_dino),
        torch.from_numpy(train_audio),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    # Apply AdamW decay only to matrix weights. Biases, normalization vectors,
    # temperature/fusion calibration, frame positions, and subject calibration
    # stay in the no-decay group.
    no_decay_names = {"log_scale", "fusion_logits", "subject_scale", "subject_bias"}
    decayed: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        leaf_name = name.rsplit(".", 1)[-1]
        if parameter.ndim < 2 or leaf_name in no_decay_names or "position" in name:
            no_decay.append(parameter)
        else:
            decayed.append(parameter)
    parameter_groups: list[dict[str, Any]] = []
    if decayed:
        parameter_groups.append({"params": decayed, "weight_decay": config.weight_decay})
    if no_decay:
        parameter_groups.append({"params": no_decay, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(parameter_groups, lr=config.lr)
    final_fit = fixed_epochs is not None
    if final_fit:
        epochs = min(fixed_epochs, 2) if config.smoke else fixed_epochs
    else:
        epochs = min(config.epochs, 2) if config.smoke else config.epochs
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_key: tuple[float, float, float] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        contrastive_sum = 0.0
        count = 0
        for eeg, subjects, video_ids, dino, audio in loader:
            eeg = eeg.to(device, non_blocking=True)
            subjects = subjects.to(device, non_blocking=True)
            video_ids = video_ids.to(device, non_blocking=True)
            dino = dino.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            eeg_embedding, stimulus_embedding = model(eeg, subjects, dino, audio)
            retrieval_loss = contrastive_loss(
                eeg_embedding, stimulus_embedding, model.log_scale, video_ids, config.loss
            )
            loss = retrieval_loss + config.subject_affine_reg * model.affine_penalty()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(eeg)
            contrastive_sum += float(retrieval_loss.item()) * len(eeg)
            count += len(eeg)

        # FINAL full-fit: no heldout evaluation or selection; keep the last epoch.
        if final_fit:
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            if epoch % 5 == 0 or epoch == epochs:
                print(
                    f"seed={seed} full-fit epoch={epoch:02d}/{epochs} "
                    f"loss={loss_sum / count:.4f}"
                )
            continue

        scores = raw_scores(
            model,
            heldout_eeg,
            heldout_subjects,
            heldout_dino,
            heldout_audio,
            device,
            config.batch_size,
        )
        metrics = metrics_from_scores(scores, HELDOUT_IDS, heldout_query_ids)
        record = {
            "epoch": epoch,
            "train_total_loss": loss_sum / count,
            "train_contrastive_loss": contrastive_sum / count,
            "heldout_top1": metrics["top1"],
            "heldout_top5": metrics["top5"],
            "heldout_mrr": metrics["mrr"],
            "heldout_median_rank": metrics["median_rank"],
            "temperature": float(1.0 / model.log_scale.exp().clamp(max=100).item()),
            "fusion_weights": model.fusion_weights(),
        }
        history.append(record)
        key = (float(metrics["top1"]), float(metrics["top5"]), float(metrics["mrr"]))
        marker = ""
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            marker = " [best]"
        print(
            f"seed={seed} epoch={epoch:02d}/{epochs} loss={record['train_total_loss']:.4f} "
            f"heldout top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} "
            f"median={metrics['median_rank']:.1f}{marker}"
        )
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    return best_state, history, best_epoch


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def synthetic_forward_backward_test() -> None:
    seed_everything(7)
    device = torch.device("cpu")
    for fusion, pooling, affine, loss_mode in (
        ("concat", "mean", False, "standard"),
        ("separate", "attention", True, "multipositive"),
    ):
        model = TunedMLPRetriever(
            eeg_dim=N_CHANNELS * 55,
            dino_dim=32,
            audio_dim=24,
            hidden_dim=48,
            embedding_dim=16,
            dropout=0.1,
            temperature=0.07,
            fusion=fusion,
            modality_dropout=0.2,
            subject_affine=affine,
            dino_pooling=pooling,
            dino_frames=8,
            attention_hidden_dim=12,
        ).to(device)
        batch = 7
        eeg = torch.randn(batch, N_CHANNELS, 55)
        subjects = torch.tensor([1, 2, 3, 4, 5, 6, 1])
        dino = torch.randn(batch, 8, 32)
        audio = torch.randn(batch, 24)
        video_ids = torch.tensor([1, 1, 2, 3, 3, 4, 5])
        eeg_embedding, stimulus_embedding = model(eeg, subjects, dino, audio)
        assert eeg_embedding.shape == stimulus_embedding.shape == (batch, 16)
        loss = contrastive_loss(
            eeg_embedding, stimulus_embedding, model.log_scale, video_ids, loss_mode
        ) + 1e-4 * model.affine_penalty()
        assert torch.isfinite(loss)
        loss.backward()
        missing = [name for name, value in model.named_parameters() if value.requires_grad and value.grad is None]
        if missing:
            raise AssertionError(f"Parameters without gradients: {missing}")
        if not all(
            torch.isfinite(value.grad).all()
            for value in model.parameters()
            if value.grad is not None
        ):
            raise AssertionError("Synthetic test produced non-finite gradients")
    singleton = torch.randn(1, 8)
    singleton = F.normalize(singleton, dim=1)
    singleton_loss = contrastive_loss(
        singleton, singleton, torch.tensor(0.0), torch.tensor([9]), "multipositive"
    )
    assert torch.isfinite(singleton_loss) and float(singleton_loss) == 0.0
    print("Synthetic forward/backward tests passed (concat/standard and separate/multipositive).")


def main() -> None:
    config = parse_args()
    if config.self_test:
        synthetic_forward_backward_test()
        return

    data_root = Path(config.data_root)
    output_dir = Path(config.output_root) / config.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    device = resolve_device(config.device)

    all_train_ids = np.concatenate((TRAIN_IDS, HELDOUT_IDS))
    dino_map = load_dino_frames(
        resolve_path(data_root, config.dino_train_cache), all_train_ids, config.dino_frames
    )
    audio_map = load_audio_map(
        data_root / "features_audio" / f"train_{config.audio_model}_features.npz"
    )
    dino_test_map: dict[int, np.ndarray] | None = None
    audio_test_map: dict[int, np.ndarray] | None = None
    if config.predict_test:
        dino_test_map = load_dino_frames(
            resolve_path(data_root, config.dino_test_cache), TEST_IDS, config.dino_frames
        )
        audio_test_map = load_audio_map(
            data_root / "features_audio" / f"test_{config.audio_model}_features.npz"
        )
        # Validate all candidate IDs and prediction-subject EEG files before
        # creating a run directory or spending time training.
        stack_dino_frames(dino_test_map, TEST_IDS, "test DINO preflight")
        stack_features(audio_test_map, TEST_IDS, "test audio preflight")
        missing_test_eeg = [
            subject
            for subject in config.prediction_subjects
            if not (data_root / "tst" / f"sub-{subject:02d}_tst.npy").exists()
        ]
        if missing_test_eeg:
            raise FileNotFoundError(
                f"Missing test EEG for prediction subjects: {missing_test_eeg}"
            )
    train_dino_raw = stack_dino_frames(dino_map, TRAIN_IDS, "training DINO")
    heldout_dino_raw = stack_dino_frames(dino_map, HELDOUT_IDS, "heldout DINO")
    train_audio_raw = stack_features(audio_map, TRAIN_IDS, "training audio")
    heldout_audio_raw = stack_features(audio_map, HELDOUT_IDS, "heldout audio")
    dino_mean, dino_std = fit_normalizer(train_dino_raw.mean(axis=1))
    audio_mean, audio_std = fit_normalizer(train_audio_raw)
    train_dino_candidates = apply_normalizer(train_dino_raw, dino_mean[:, None, :], dino_std[:, None, :])
    heldout_dino_candidates = apply_normalizer(
        heldout_dino_raw, dino_mean[:, None, :], dino_std[:, None, :]
    )
    train_audio_candidates = apply_normalizer(train_audio_raw, audio_mean, audio_std)
    heldout_audio_candidates = apply_normalizer(heldout_audio_raw, audio_mean, audio_std)

    train_eeg, train_query_ids, train_subjects, train_counts = load_pooled_eeg(
        data_root, config.subjects, 1, 900
    )
    heldout_eeg, heldout_query_ids, heldout_subjects, heldout_counts = load_pooled_eeg(
        data_root, config.subjects, 901, 1000
    )
    train_dino = index_by_ids(train_dino_candidates, TRAIN_IDS, train_query_ids)
    train_audio = index_by_ids(train_audio_candidates, TRAIN_IDS, train_query_ids)
    if config.smoke:
        train_keep = np.concatenate(
            [np.flatnonzero(train_subjects == subject)[:64] for subject in config.subjects]
        )
        heldout_keep = np.concatenate(
            [np.flatnonzero(heldout_subjects == subject)[:32] for subject in config.subjects]
        )
        train_eeg, train_query_ids, train_subjects = (
            train_eeg[train_keep], train_query_ids[train_keep], train_subjects[train_keep]
        )
        train_dino, train_audio = train_dino[train_keep], train_audio[train_keep]
        heldout_eeg, heldout_query_ids, heldout_subjects = (
            heldout_eeg[heldout_keep],
            heldout_query_ids[heldout_keep],
            heldout_subjects[heldout_keep],
        )

    if train_eeg.shape[1:] != (17, 55) or train_dino.shape[1] != config.dino_frames:
        raise AssertionError("Protocol shape assertion failed")
    if heldout_dino_candidates.shape[0] != len(HELDOUT_IDS):
        raise AssertionError("Heldout candidate bank must contain exactly IDs 901..1000")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(json_ready(asdict(config)), indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "normalization.npz",
        dino_mean=dino_mean,
        dino_std=dino_std,
        audio_mean=audio_mean,
        audio_std=audio_std,
        fitted_stimulus_ids=TRAIN_IDS,
    )
    np.savez_compressed(
        output_dir / "split.npz",
        train_candidate_ids=TRAIN_IDS,
        heldout_candidate_ids=HELDOUT_IDS,
        train_query_ids=train_query_ids,
        train_query_subjects=train_subjects,
        heldout_query_ids=heldout_query_ids,
        heldout_query_subjects=heldout_subjects,
        eeg_sample_indices=np.arange(WINDOW.start, WINDOW.stop, dtype=np.int64),
    )

    print("=" * 78)
    print("Tuned MLP: frozen frame-level DINO + frozen audio cache")
    print(
        f"subjects={config.subjects} seeds={config.seeds} audio={config.audio_model} "
        f"loss={config.loss} fusion={config.fusion} pooling={config.dino_pooling} device={device}"
    )
    print(f"training counts={train_counts}; heldout counts={heldout_counts}")
    print("=" * 78)

    states: dict[int, dict[str, torch.Tensor]] = {}
    histories: dict[int, list[dict[str, Any]]] = {}
    best_epochs: dict[int, int] = {}
    heldout_seed_scores: list[np.ndarray] = []
    seed_metadata: dict[int, dict[str, Any]] = {}
    dino_dim, audio_dim = train_dino.shape[2], train_audio.shape[1]
    for seed in config.seeds:
        state, history, best_epoch = train_seed(
            config,
            seed,
            train_eeg,
            train_subjects,
            train_query_ids,
            train_dino,
            train_audio,
            heldout_eeg,
            heldout_subjects,
            heldout_query_ids,
            heldout_dino_candidates,
            heldout_audio_candidates,
            device,
        )
        states[seed], histories[seed], best_epochs[seed] = state, history, best_epoch
        model = make_model(config, dino_dim, audio_dim, device)
        model.load_state_dict(state)
        heldout_seed_scores.append(
            raw_scores(
                model,
                heldout_eeg,
                heldout_subjects,
                heldout_dino_candidates,
                heldout_audio_candidates,
                device,
                config.batch_size,
            )
        )
        seed_metadata[seed] = {
            "fusion_weights": model.fusion_weights(),
            "attention": attention_summary(
                model, heldout_dino_candidates, device, config.batch_size
            ),
        }
        torch.save(
            {
                "model_state_dict": state,
                "seed": seed,
                "best_epoch": best_epoch,
                "config": asdict(config),
                "dino_dim": dino_dim,
                "audio_dim": audio_dim,
                "normalization_file": "normalization.npz",
                "split_file": "split.npz",
                "cache_inputs_are_frozen": True,
            },
            output_dir / f"model_seed-{seed}.pt",
        )

    ensemble_raw = np.mean(heldout_seed_scores, axis=0).astype(np.float32)
    if config.anti_hub == "csls":
        selected_lambda, heldout_metrics = choose_csls_lambda(
            ensemble_raw,
            HELDOUT_IDS,
            heldout_query_ids,
            config.csls_k,
            config.csls_lambda,
        )
        ensemble_adjusted = mild_csls(ensemble_raw, config.csls_k, selected_lambda)
    else:
        selected_lambda = 0.0
        ensemble_adjusted = ensemble_raw.copy()
        heldout_metrics = metrics_from_scores(
            ensemble_adjusted, HELDOUT_IDS, heldout_query_ids
        )
    heldout_rankings = ranked_from_scores(ensemble_adjusted, HELDOUT_IDS)
    np.savez_compressed(
        output_dir / "heldout_outputs.npz",
        query_ids=heldout_query_ids,
        query_subjects=heldout_subjects,
        candidate_ids=HELDOUT_IDS,
        raw_scores=ensemble_raw,
        adjusted_scores=ensemble_adjusted,
        rankings=heldout_rankings,
        selected_csls_lambda=np.float32(selected_lambda),
    )

    # FINAL FULL FIT: epoch + CSLS lambda are now fixed from the 1-900/heldout signal.
    # Retrain each seed on ALL 1-1000 stimuli for its selected epoch count and use those
    # models for every test prediction. Heldout metrics above remain the honest estimate.
    if config.final_full_fit:
        full_eeg = np.concatenate((train_eeg, heldout_eeg), axis=0)
        full_subjects = np.concatenate((train_subjects, heldout_subjects), axis=0)
        full_ids = np.concatenate((train_query_ids, heldout_query_ids), axis=0)
        full_dino = np.concatenate((train_dino, heldout_dino_candidates[
            [int(np.where(HELDOUT_IDS == vid)[0][0]) for vid in heldout_query_ids]
        ]), axis=0)
        full_audio = np.concatenate((train_audio, heldout_audio_candidates[
            [int(np.where(HELDOUT_IDS == vid)[0][0]) for vid in heldout_query_ids]
        ]), axis=0)
        print("=" * 78)
        print(
            "FINAL FULL FIT: retraining each seed on all 1-1000 stimuli "
            f"({len(full_eeg)} trials) for its selected epoch count {best_epochs}"
        )
        print("=" * 78)
        for seed in config.seeds:
            full_state, _, _ = train_seed(
                config,
                seed,
                full_eeg,
                full_subjects,
                full_ids,
                full_dino,
                full_audio,
                heldout_eeg,
                heldout_subjects,
                heldout_query_ids,
                heldout_dino_candidates,
                heldout_audio_candidates,
                device,
                fixed_epochs=max(1, int(best_epochs[seed])),
            )
            states[seed] = full_state
            torch.save(
                {
                    "model_state_dict": full_state,
                    "seed": seed,
                    "fixed_epochs": int(best_epochs[seed]),
                    "config": asdict(config),
                    "dino_dim": dino_dim,
                    "audio_dim": audio_dim,
                    "fit_on": "stimuli 1-1000",
                },
                output_dir / f"model_seed-{seed}_fullfit.pt",
            )

    public_metrics: dict[int, dict[str, Any]] = {}
    test_summaries: dict[int, dict[str, Any]] = {}
    if config.predict_test:
        if dino_test_map is None or audio_test_map is None:
            raise AssertionError("Test feature preflight did not run")
        test_dino_raw = stack_dino_frames(dino_test_map, TEST_IDS, "test DINO")
        test_audio_raw = stack_features(audio_test_map, TEST_IDS, "test audio")
        test_dino = apply_normalizer(
            test_dino_raw, dino_mean[:, None, :], dino_std[:, None, :]
        )
        test_audio = apply_normalizer(test_audio_raw, audio_mean, audio_std)
        subject_eeg: dict[int, np.ndarray] = {}
        for subject in config.prediction_subjects:
            rows = load_test_eeg(data_root, subject)[:, :, WINDOW]
            subject_eeg[subject] = normalize_trials(rows)
        score_sums = {
            subject: np.zeros((len(subject_eeg[subject]), len(TEST_IDS)), dtype=np.float32)
            for subject in config.prediction_subjects
        }
        for seed in config.seeds:
            model = make_model(config, dino_dim, audio_dim, device)
            model.load_state_dict(states[seed])
            candidate_embeddings = encode_stimulus(
                model, test_dino, test_audio, device, config.batch_size
            )
            for subject in config.prediction_subjects:
                subject_ids = np.full(len(subject_eeg[subject]), subject, dtype=np.int64)
                query_embeddings = encode_eeg(
                    model,
                    subject_eeg[subject],
                    subject_ids,
                    device,
                    config.batch_size,
                )
                score_sums[subject] += query_embeddings @ candidate_embeddings.T

        submissions = output_dir / "submissions"
        submissions.mkdir()
        for subject in config.prediction_subjects:
            raw = (score_sums[subject] / len(config.seeds)).astype(np.float32)
            adjusted = (
                mild_csls(raw, config.csls_k, selected_lambda)
                if config.anti_hub == "csls"
                else raw.copy()
            )
            rankings = ranked_from_scores(adjusted, TEST_IDS)
            np.savez_compressed(
                output_dir / f"sub-{subject:02d}_test_outputs.npz",
                test_row_indices=np.arange(len(raw), dtype=np.int64),
                candidate_ids=TEST_IDS,
                raw_scores=raw,
                adjusted_scores=adjusted,
                rankings=rankings,
                selected_csls_lambda=np.float32(selected_lambda),
            )
            descriptive_ranking_path = (
                submissions
                / f"tuned_mlp_dino_{config.audio_model}_sub-{subject:02d}_ranked_candidate_list.npy"
            )
            np.save(descriptive_ranking_path, rankings)
            if subject in (5, 6):
                np.save(
                    submissions / f"sub-{subject:02d}_ranked_candidate_list.npy",
                    rankings,
                )
            # This block executes after all model/CSLS selection and can never read 5/6 labels.
            if config.report_public_subjects and subject in PUBLIC_SUBJECTS:
                truth_path = data_root / "tst_stim" / f"sub-{subject:02d}_tst_stim.npy"
                if truth_path.exists():
                    truth = np.asarray(np.load(truth_path), dtype=np.int64)
                    if truth.shape != (len(raw),):
                        raise ValueError(f"Unexpected public label shape at {truth_path}: {truth.shape}")
                    if len(np.unique(truth)) != len(TEST_IDS) or set(truth.tolist()) != set(TEST_IDS.tolist()):
                        raise ValueError(
                            f"Public labels at {truth_path} must be a permutation of IDs 1001..1102"
                        )
                    public_metrics[subject] = metrics_from_scores(adjusted, TEST_IDS, truth)
            test_summaries[subject] = {
                "scores_shape": list(raw.shape),
                "rankings_shape": list(rankings.shape),
                "original_test_row_order_preserved": True,
                "subject_ids_passed_to_model": True,
            }

    mean_fusion: dict[str, float] = {}
    fusion_keys = next(iter(seed_metadata.values()))["fusion_weights"].keys()
    for key in fusion_keys:
        mean_fusion[key] = float(
            np.mean([metadata["fusion_weights"][key] for metadata in seed_metadata.values()])
        )
    model_metadata = {
        "per_seed": seed_metadata,
        "mean_fusion_weights": mean_fusion,
        "attention": {
            "pooling": config.dino_pooling,
            "frame_count": config.dino_frames,
            "required_frame_ids": list(range(config.dino_frames)),
            "strict_cache_validation": True,
        },
    }
    safeguards = {
        "dino_cache_frozen": True,
        "audio_cache_frozen": True,
        "feature_normalizers_fit_on_ids": "1..900 only",
        "checkpoint_selection_uses": "raw cosine on IDs 901..1000 only",
        "csls_selected_on": "IDs 901..1000 only" if config.anti_hub == "csls" else None,
        "seed_ensemble_combines": "raw cosine scores",
        "public_labels_loaded_during_selection": False,
        "subject_05_06_labels_loaded": False,
        "public_label_reporting_after_all_selection": bool(config.report_public_subjects),
        "existing_run_overwritten": False,
        "test_original_row_order_preserved": True,
    }
    metrics = {
        "heldout": heldout_metrics,
        "public_validation_reporting_only": public_metrics,
        "best_epochs": best_epochs,
        "final_full_fit": config.final_full_fit,
        "test_models_fit_on": "stimuli 1-1000" if config.final_full_fit else "stimuli 1-900",
        "selected_csls_lambda": selected_lambda,
        "anti_hub": config.anti_hub,
        "csls_k": config.csls_k,
        "fusion_weights": mean_fusion,
        "train_counts_before_smoke_cap": train_counts,
        "heldout_counts_before_smoke_cap": heldout_counts,
        "test": test_summaries,
        "safeguards": safeguards,
    }
    (output_dir / "history.json").write_text(
        json.dumps(json_ready(histories), indent=2), encoding="utf-8"
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(json_ready(model_metadata), indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2), encoding="utf-8"
    )
    (output_dir / "safeguards.json").write_text(
        json.dumps(safeguards, indent=2), encoding="utf-8"
    )

    print("\nENSEMBLE HELDOUT RESULT")
    print(
        f"top1={heldout_metrics['top1']:.4f} top5={heldout_metrics['top5']:.4f} "
        f"median={heldout_metrics['median_rank']:.1f} mrr={heldout_metrics['mrr']:.4f}"
    )
    print(f"CSLS lambda={selected_lambda:.2f}; fusion weights={mean_fusion}")
    if public_metrics:
        print("PUBLIC SUBJECT RESULTS (reporting only; never used for selection)")
        for subject, values in public_metrics.items():
            print(
                f"sub-{subject:02d}: top1={values['top1']:.4f} "
                f"top5={values['top5']:.4f} median={values['median_rank']:.1f}"
            )
    print(f"Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
