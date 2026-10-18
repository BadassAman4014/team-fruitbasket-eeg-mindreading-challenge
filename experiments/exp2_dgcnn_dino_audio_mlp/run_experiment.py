"""DGCNN EEG retrieval against frozen DINOv2 + audio features through an MLP.

DINOv2 and audio foundation models are never loaded or updated here. The runner
consumes their existing cached embeddings, trains a DGCNN EEG query encoder and
the repository's multimodal MLP projector, then ranks candidates by cosine score.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
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

from experiments.exp1_two_layer_gcn.run_experiment import (  # noqa: E402
    DynamicEdgeConvolution,
    fit_eeg_normalizer,
    load_test_eeg,
    load_training_eeg,
    normalize_eeg,
    resolve_device,
    select_rows,
    split_video_ids,
)
from src.metrics import compute_csls, evaluate_ranking, ranked_from_scores  # noqa: E402
from src.models.contrastive_eeg import MultiBranchProjector, info_nce  # noqa: E402


@dataclass
class Config:
    data_root: str
    output_root: str
    run_name: str
    subject: int = 1
    audio_model: str = "clap"
    dino_train_cache: str = "features_dinov2/consolidated_encoder.layer.11.npz"
    dino_test_cache: str = "features_test_dinov2/consolidated_encoder.layer.11.npz"
    seed: int = 42
    val_videos: int = 102
    window_start: int = 5
    window_end: int = 60
    graph_k: int = 4
    hidden_dim: int = 64
    graph_embedding_dim: int = 32
    retrieval_dim: int = 256
    dropout: float = 0.3
    modality_dropout: float = 0.15
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    temperature: float = 0.07
    max_epochs: int = 100
    patience: int = 20
    clip_sd: float = 5.0
    hubness_mode: str = "none"
    device: str = "auto"
    num_workers: int = 0
    smoke: bool = False
    predict_test: bool = False


class DGCNNMultimodalRanker(nn.Module):
    """DGCNN EEG encoder and existing multimodal stimulus projection MLP."""

    def __init__(
        self,
        input_dim: int,
        block_dims: dict[str, int],
        graph_k: int = 4,
        hidden_dim: int = 64,
        graph_embedding_dim: int = 32,
        retrieval_dim: int = 256,
        dropout: float = 0.3,
        modality_dropout: float = 0.15,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if not 1 <= graph_k < 17:
            raise ValueError("graph_k must be between 1 and 16")
        self.edge_conv1 = DynamicEdgeConvolution(input_dim, hidden_dim, graph_k)
        self.edge_conv2 = DynamicEdgeConvolution(
            hidden_dim, graph_embedding_dim, graph_k
        )
        self.eeg_dropout = nn.Dropout(dropout)
        self.eeg_projector = nn.Sequential(
            nn.Linear(graph_embedding_dim, retrieval_dim),
            nn.LayerNorm(retrieval_dim),
        )
        # Reuse the repository's per-modality MLPs and learned softmax fusion.
        self.stim_projector = MultiBranchProjector(
            block_dims=block_dims,
            emb_dim=retrieval_dim,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.edge_conv1(eeg))
        hidden = self.eeg_dropout(hidden)
        hidden = torch.relu(self.edge_conv2(hidden))
        pooled = hidden.mean(dim=1)
        return F.normalize(self.eeg_projector(pooled), p=2, dim=-1)

    def encode_stim(
        self, stimulus: dict[str, torch.Tensor], training: bool = False
    ) -> torch.Tensor:
        return self.stim_projector(stimulus, training=training)

    def forward(
        self, eeg: torch.Tensor, stimulus: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_eeg(eeg), self.encode_stim(stimulus, training=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "exp2_dgcnn_dino_audio_mlp" / "runs",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--subject", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--audio-model", choices=("clap", "ast", "beats"), default="clap")
    parser.add_argument(
        "--dino-train-cache",
        default="features_dinov2/consolidated_encoder.layer.11.npz",
    )
    parser.add_argument(
        "--dino-test-cache",
        default="features_test_dinov2/consolidated_encoder.layer.11.npz",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-videos", type=int, default=102)
    parser.add_argument("--graph-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hubness-mode", choices=("none", "csls"), default="none")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or (
        f"dgcnn-dino-{args.audio_model}-sub-{args.subject:02d}-"
        f"{'smoke' if args.smoke else 'full'}-seed-{args.seed}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    return Config(
        data_root=str(args.data_root.resolve()),
        output_root=str(args.output_root.resolve()),
        run_name=run_name,
        subject=args.subject,
        audio_model=args.audio_model,
        dino_train_cache=args.dino_train_cache,
        dino_test_cache=args.dino_test_cache,
        seed=args.seed,
        val_videos=args.val_videos,
        graph_k=args.graph_k,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
        temperature=args.temperature,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hubness_mode=args.hubness_mode,
        device=args.device,
        num_workers=args.num_workers,
        smoke=args.smoke,
        predict_test=args.predict_test,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(data_root: Path, path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else data_root / resolved


def load_dino_map(path: Path) -> dict[int, np.ndarray]:
    """Mean-pool existing frame-level DINO vectors into one vector per video ID."""
    if not path.exists():
        raise FileNotFoundError(
            f"DINO cache not found: {path}. Generate/copy the existing consolidated "
            "DINO train and test caches before running this experiment."
        )
    grouped: dict[int, list[np.ndarray]] = {}
    with np.load(path) as cache:
        for key in cache.files:
            try:
                video_id = int(key.split("_")[0])
            except ValueError:
                continue
            grouped.setdefault(video_id, []).append(
                np.asarray(cache[key], dtype=np.float32).reshape(-1)
            )
    if not grouped:
        raise ValueError(f"No '<video_id>_<frame>' DINO entries found in {path}")
    return {
        video_id: np.mean(np.stack(vectors), axis=0).astype(np.float32)
        for video_id, vectors in grouped.items()
    }


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


def adjusted_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    return compute_csls(scores, k=10) if mode == "csls" else scores


def encode_eeg(
    model: DGCNNMultimodalRanker,
    eeg: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(eeg), batch_size):
            batch = torch.from_numpy(eeg[start : start + batch_size]).to(device)
            outputs.append(model.encode_eeg(batch).cpu().numpy())
    return np.concatenate(outputs)


def encode_stimulus(
    model: DGCNNMultimodalRanker,
    blocks: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    count = len(next(iter(blocks.values())))
    with torch.no_grad():
        for start in range(0, count, batch_size):
            batch = {
                name: torch.from_numpy(values[start : start + batch_size]).to(device)
                for name, values in blocks.items()
            }
            outputs.append(model.encode_stim(batch, training=False).cpu().numpy())
    return np.concatenate(outputs)


def retrieval(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    hubness_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_scores = (query_embeddings @ candidate_embeddings.T).astype(np.float32)
    scores = adjusted_scores(raw_scores, hubness_mode).astype(np.float32)
    rankings = ranked_from_scores(scores, candidate_ids)
    metrics = evaluate_ranking(rankings, query_ids)
    metrics["ranks"] = metrics["ranks"].tolist()
    return scores, rankings, metrics


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.99) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def evaluate_validation(
    model: DGCNNMultimodalRanker,
    val_eeg: np.ndarray,
    val_query_ids: np.ndarray,
    val_blocks: dict[str, np.ndarray],
    val_candidate_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    hubness_mode: str,
) -> tuple[float, dict[str, Any]]:
    eeg_embeddings = encode_eeg(model, val_eeg, device, batch_size)
    candidate_embeddings = encode_stimulus(model, val_blocks, device, batch_size)
    _, _, metrics = retrieval(
        eeg_embeddings,
        candidate_embeddings,
        val_query_ids,
        val_candidate_ids,
        hubness_mode,
    )
    candidate_index = {int(video_id): i for i, video_id in enumerate(val_candidate_ids)}
    matching = candidate_embeddings[
        [candidate_index[int(video_id)] for video_id in val_query_ids]
    ]
    with torch.no_grad():
        loss = info_nce(
            torch.from_numpy(eeg_embeddings).to(device),
            torch.from_numpy(matching).to(device),
            model.log_scale,
            torch.from_numpy(val_query_ids).to(device),
        )
    return float(loss.item()), metrics


def train_model(
    config: Config,
    model: DGCNNMultimodalRanker,
    train_eeg: np.ndarray,
    train_ids: np.ndarray,
    train_blocks: dict[str, np.ndarray],
    val_eeg: np.ndarray,
    val_query_ids: np.ndarray,
    val_blocks: dict[str, np.ndarray],
    val_candidate_ids: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    audio_name = next(name for name in train_blocks if name.startswith("aud_"))
    dataset = TensorDataset(
        torch.from_numpy(train_eeg),
        torch.from_numpy(train_blocks["vis_dino"]),
        torch.from_numpy(train_blocks[audio_name]),
        torch.from_numpy(train_ids),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    max_epochs = min(config.max_epochs, 2) if config.smoke else config.max_epochs
    patience = min(config.patience, 2) if config.smoke else config.patience
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for eeg, dino, audio, video_ids in loader:
            eeg = eeg.to(device, non_blocking=True)
            stimulus = {
                "vis_dino": dino.to(device, non_blocking=True),
                audio_name: audio.to(device, non_blocking=True),
            }
            video_ids = video_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            eeg_embedding, stimulus_embedding = model(eeg, stimulus)
            loss = info_nce(
                eeg_embedding, stimulus_embedding, model.log_scale, video_ids
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema_model, model)
            total_loss += float(loss.item()) * len(eeg)
            total_examples += len(eeg)

        val_loss, val_metrics = evaluate_validation(
            ema_model,
            val_eeg,
            val_query_ids,
            val_blocks,
            val_candidate_ids,
            device,
            config.batch_size,
            config.hubness_mode,
        )
        record = {
            "epoch": epoch,
            "train_infonce": total_loss / max(total_examples, 1),
            "validation_infonce": val_loss,
            "validation_top1": val_metrics["top1"],
            "validation_top5": val_metrics["top5"],
            "validation_mrr": val_metrics["mrr"],
            "validation_median_rank": val_metrics["median_rank"],
            "temperature": float(1.0 / ema_model.log_scale.exp().clamp(max=100).item()),
            "fusion_weights": ema_model.stim_projector.get_fusion_weights(),
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_nce={record['train_infonce']:.4f} "
            f"val_nce={val_loss:.4f} val_top1={val_metrics['top1']:.4f} "
            f"val_top5={val_metrics['top5']:.4f} val_mrr={val_metrics['mrr']:.4f} "
            f"val_med={val_metrics['median_rank']:.1f} "
            f"fusion={record['fusion_weights']}"
        )

        key = (
            float(val_metrics["mrr"]),
            float(val_metrics["top5"]),
            -float(val_metrics["median_rank"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in ema_model.state_dict().items()
            }
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    return best_state, history, best_epoch


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data_root = Path(config.data_root)
    output_dir = Path(config.output_root) / config.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")

    dino_train = load_dino_map(resolve_path(data_root, config.dino_train_cache))
    dino_test = load_dino_map(resolve_path(data_root, config.dino_test_cache))
    audio_train_path = data_root / "features_audio" / f"train_{config.audio_model}_features.npz"
    audio_test_path = data_root / "features_audio" / f"test_{config.audio_model}_features.npz"
    audio_train = load_audio_map(audio_train_path)
    audio_test = load_audio_map(audio_test_path)
    audio_name = f"aud_{config.audio_model}"

    all_ids, all_eeg = load_training_eeg(data_root, config.subject)
    available = np.array(
        [int(video_id) in dino_train and int(video_id) in audio_train for video_id in all_ids]
    )
    if not available.all():
        print(f"Filtering {(~available).sum()} training IDs missing a frozen feature block.")
    all_ids, all_eeg = all_ids[available], all_eeg[available]
    train_ids, val_ids = split_video_ids(all_ids, config.val_videos, config.seed)
    train_rows = select_rows(all_ids, train_ids)
    val_rows = select_rows(all_ids, val_ids)
    if config.smoke:
        train_rows = train_rows[: min(128, len(train_rows))]
        train_ids = all_ids[train_rows]
        val_rows = val_rows[: min(32, len(val_rows))]
    val_query_ids = all_ids[val_rows]

    raw_train_eeg = all_eeg[train_rows, :, config.window_start : config.window_end]
    raw_val_eeg = all_eeg[val_rows, :, config.window_start : config.window_end]
    eeg_mean, eeg_std = fit_eeg_normalizer(raw_train_eeg)
    train_eeg = normalize_eeg(raw_train_eeg, eeg_mean, eeg_std, config.clip_sd)
    val_eeg = normalize_eeg(raw_val_eeg, eeg_mean, eeg_std, config.clip_sd)

    train_dino_raw = stack_features(dino_train, train_ids, "training DINO")
    val_dino_raw = stack_features(dino_train, val_ids, "validation DINO")
    train_audio_raw = stack_features(audio_train, train_ids, f"training {audio_name}")
    val_audio_raw = stack_features(audio_train, val_ids, f"validation {audio_name}")
    dino_mean, dino_std = fit_normalizer(train_dino_raw)
    audio_mean, audio_std = fit_normalizer(train_audio_raw)
    train_blocks = {
        "vis_dino": apply_normalizer(train_dino_raw, dino_mean, dino_std),
        audio_name: apply_normalizer(train_audio_raw, audio_mean, audio_std),
    }
    val_blocks = {
        "vis_dino": apply_normalizer(val_dino_raw, dino_mean, dino_std),
        audio_name: apply_normalizer(val_audio_raw, audio_mean, audio_std),
    }
    block_dims = {name: values.shape[1] for name, values in train_blocks.items()}

    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "normalization.npz",
        eeg_mean=eeg_mean,
        eeg_std=eeg_std,
        dino_mean=dino_mean,
        dino_std=dino_std,
        audio_mean=audio_mean,
        audio_std=audio_std,
    )
    np.savez_compressed(
        output_dir / "split.npz",
        train_video_ids=train_ids,
        validation_video_ids=val_ids,
        validation_query_ids=val_query_ids,
    )

    shapes = {
        "train_eeg": list(train_eeg.shape),
        "validation_eeg": list(val_eeg.shape),
        "train_dino": list(train_blocks["vis_dino"].shape),
        "train_audio": list(train_blocks[audio_name].shape),
        "validation_candidates": len(val_ids),
    }
    print(f"device={device}; audio={config.audio_model}; shapes={shapes}")
    assert train_eeg.shape[1:] == (17, 55)
    assert len(val_ids) == config.val_videos

    model = DGCNNMultimodalRanker(
        input_dim=55,
        block_dims=block_dims,
        graph_k=config.graph_k,
        hidden_dim=config.hidden_dim,
        graph_embedding_dim=config.graph_embedding_dim,
        retrieval_dim=config.retrieval_dim,
        dropout=config.dropout,
        modality_dropout=config.modality_dropout,
        temperature=config.temperature,
    ).to(device)
    with torch.no_grad():
        eeg_shape = model.encode_eeg(torch.from_numpy(train_eeg[:2]).to(device))
        stim_shape = model.encode_stim(
            {
                name: torch.from_numpy(values[:2]).to(device)
                for name, values in train_blocks.items()
            }
        )
    assert eeg_shape.shape == stim_shape.shape == (min(2, len(train_eeg)), 256)

    best_state, history, best_epoch = train_model(
        config,
        model,
        train_eeg,
        train_ids,
        train_blocks,
        val_eeg,
        val_query_ids,
        val_blocks,
        val_ids,
        device,
    )
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(config),
            "block_dims": block_dims,
            "best_epoch": best_epoch,
        },
        output_dir / "model.pt",
    )
    (output_dir / "history.json").write_text(
        json.dumps(json_ready(history), indent=2), encoding="utf-8"
    )

    train_eeg_embeddings = encode_eeg(model, train_eeg, device, config.batch_size)
    train_candidate_embeddings = encode_stimulus(
        model, train_blocks, device, config.batch_size
    )
    train_scores, train_rankings, train_metrics = retrieval(
        train_eeg_embeddings,
        train_candidate_embeddings,
        train_ids,
        train_ids,
        config.hubness_mode,
    )
    val_eeg_embeddings = encode_eeg(model, val_eeg, device, config.batch_size)
    val_candidate_embeddings = encode_stimulus(model, val_blocks, device, config.batch_size)
    val_scores, val_rankings, val_metrics = retrieval(
        val_eeg_embeddings,
        val_candidate_embeddings,
        val_query_ids,
        val_ids,
        config.hubness_mode,
    )
    np.savez_compressed(
        output_dir / "retrieval_outputs.npz",
        train_query_ids=train_ids,
        train_candidate_ids=train_ids,
        train_scores=train_scores,
        train_rankings=train_rankings,
        train_eeg_embeddings=train_eeg_embeddings,
        train_candidate_embeddings=train_candidate_embeddings,
        validation_query_ids=val_query_ids,
        validation_candidate_ids=val_ids,
        validation_scores=val_scores,
        validation_rankings=val_rankings,
        validation_eeg_embeddings=val_eeg_embeddings,
        validation_candidate_embeddings=val_candidate_embeddings,
    )

    test_summary = None
    if config.predict_test:
        test_ids = np.arange(1001, 1103, dtype=np.int64)
        raw_test_eeg = load_test_eeg(data_root, config.subject)
        test_eeg = normalize_eeg(
            raw_test_eeg[:, :, config.window_start : config.window_end],
            eeg_mean,
            eeg_std,
            config.clip_sd,
        )
        test_blocks = {
            "vis_dino": apply_normalizer(
                stack_features(dino_test, test_ids, "test DINO"), dino_mean, dino_std
            ),
            audio_name: apply_normalizer(
                stack_features(audio_test, test_ids, f"test {audio_name}"),
                audio_mean,
                audio_std,
            ),
        }
        test_eeg_embeddings = encode_eeg(model, test_eeg, device, config.batch_size)
        test_candidate_embeddings = encode_stimulus(
            model, test_blocks, device, config.batch_size
        )
        raw_test_scores = test_eeg_embeddings @ test_candidate_embeddings.T
        test_scores = adjusted_scores(raw_test_scores, config.hubness_mode).astype(np.float32)
        test_rankings = ranked_from_scores(test_scores, test_ids)
        np.savez_compressed(
            output_dir / "test_outputs.npz",
            test_row_indices=np.arange(102, dtype=np.int64),
            candidate_ids=test_ids,
            scores=test_scores,
            rankings=test_rankings,
            eeg_embeddings=test_eeg_embeddings,
            candidate_embeddings=test_candidate_embeddings,
        )
        ranking_name = (
            f"sub-{config.subject:02d}_dgcnn_dino_{config.audio_model}_mlp_"
            "ranked_candidate_list.npy"
        )
        np.save(output_dir / ranking_name, test_rankings)
        test_summary = {
            "scores_shape": list(test_scores.shape),
            "rankings_shape": list(test_rankings.shape),
            "candidate_ids": [1001, 1102],
            "original_test_row_order_preserved": True,
        }

    metrics = {
        "subject": config.subject,
        "audio_model": config.audio_model,
        "best_epoch": best_epoch,
        "train": train_metrics,
        "validation": val_metrics,
        "fusion_weights": model.stim_projector.get_fusion_weights(),
        "temperature": float(1.0 / model.log_scale.exp().clamp(max=100).item()),
        "shapes": shapes,
        "test": test_summary,
        "safeguards": {
            "dino_foundation_model_frozen": True,
            "audio_foundation_model_frozen": True,
            "feature_normalization_fit_on_training_split_only": True,
            "test_labels_loaded": False,
            "existing_outputs_modified": False,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2), encoding="utf-8"
    )
    print("\nFINAL RETRIEVAL")
    print(
        f"train      top1={train_metrics['top1']:.4f} top5={train_metrics['top5']:.4f} "
        f"median={train_metrics['median_rank']:.1f} mrr={train_metrics['mrr']:.4f}"
    )
    print(
        f"validation top1={val_metrics['top1']:.4f} top5={val_metrics['top5']:.4f} "
        f"median={val_metrics['median_rank']:.1f} mrr={val_metrics['mrr']:.4f}"
    )
    print(f"fusion weights: {metrics['fusion_weights']}")
    print(f"Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
