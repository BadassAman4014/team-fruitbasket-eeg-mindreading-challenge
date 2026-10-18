"""Fixed-graph GCN and dynamic EdgeConv DGCNN for EEG-to-video retrieval.

The script is intentionally isolated from the original prediction/submission pipeline.
It uses only training EEG labels for splitting, fitting, early stopping, and metrics.
Use ``--architecture dgcnn`` to rebuild a per-trial k-NN graph at each layer.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.signal import decimate
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import evaluate_ranking, ranked_from_scores  # noqa: E402


@dataclass
class Config:
    data_root: str
    output_root: str
    run_name: str
    architecture: str = "gcn"
    subject: int = 1
    stim_cache: str = "cache/stim/vis_alexnet_k256.npz"
    seed: int = 42
    val_videos: int = 102
    window_start: int = 5
    window_end: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.5
    hidden_dim: int = 64
    embedding_dim: int = 32
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-6
    graph: str = "correlation_knn"
    graph_k: int = 4
    clip_sd: float = 5.0
    device: str = "auto"
    num_workers: int = 0
    smoke: bool = False
    predict_test: bool = False


class GraphConvolution(nn.Module):
    """A graph-convolution layer: A_norm @ X @ W + b."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("ij,bjf->bif", adjacency, x)
        return self.linear(aggregated)


class TwoLayerGCN(nn.Module):
    """17-node fixed-graph GCN with a pooled 32-D EEG embedding."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        adjacency: np.ndarray,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(embedding_dim, target_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x, self.adjacency))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, self.adjacency))
        embedding = x.mean(dim=1)
        prediction = self.regression_head(embedding)
        return prediction, embedding


class DynamicEdgeConvolution(nn.Module):
    """EdgeConv using a k-NN graph recomputed from every trial and layer input."""

    def __init__(self, in_features: int, out_features: int, k: int) -> None:
        super().__init__()
        self.k = k
        # Edge feature for i <- j is [x_i, x_j - x_i].
        self.message = nn.Linear(2 * in_features, out_features)

    def knn_indices(self, x: torch.Tensor) -> torch.Tensor:
        if self.k >= x.shape[1]:
            raise ValueError(f"DGCNN k={self.k} must be smaller than {x.shape[1]} nodes")
        distances = torch.cdist(x, x, p=2)
        diagonal = torch.eye(x.shape[1], device=x.device, dtype=torch.bool).unsqueeze(0)
        distances = distances.masked_fill(diagonal, torch.inf)
        return distances.topk(self.k, dim=-1, largest=False, sorted=True).indices

    def forward(
        self, x: torch.Tensor, return_neighbors: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        neighbors_index = self.knn_indices(x)
        batch_index = torch.arange(x.shape[0], device=x.device)[:, None, None]
        neighbors = x[batch_index, neighbors_index]
        centers = x.unsqueeze(2).expand(-1, -1, self.k, -1)
        edges = torch.cat((centers, neighbors - centers), dim=-1)
        output = self.message(edges).amax(dim=2)
        if return_neighbors:
            return output, neighbors_index
        return output


class TwoLayerDGCNN(nn.Module):
    """Two dynamic EdgeConv layers with per-layer, per-trial k-NN graphs."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        k: int = 4,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if not 1 <= k < 17:
            raise ValueError("DGCNN graph_k must be between 1 and 16")
        self.edge_conv1 = DynamicEdgeConvolution(input_dim, hidden_dim, k)
        self.edge_conv2 = DynamicEdgeConvolution(hidden_dim, embedding_dim, k)
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(embedding_dim, target_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.edge_conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.edge_conv2(x))
        embedding = x.mean(dim=1)
        prediction = self.regression_head(embedding)
        return prediction, embedding

    def neighbor_indices(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both evaluation-mode dynamic graphs for auditing."""
        first_hidden, first_neighbors = self.edge_conv1(x, return_neighbors=True)
        first_hidden = torch.relu(first_hidden)
        second_neighbors = self.edge_conv2.knn_indices(first_hidden)
        return first_neighbors, second_neighbors


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "exp1_two_layer_gcn" / "runs",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--architecture",
        choices=("gcn", "dgcnn"),
        default="gcn",
        help="Use 'dgcnn' for per-trial dynamic EdgeConv graphs.",
    )
    parser.add_argument("--subject", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--stim-cache", default="cache/stim/vis_alexnet_k256.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-videos", type=int, default=102)
    parser.add_argument("--batch-size", type=int, default=32, choices=range(16, 33))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--graph",
        choices=("correlation_knn", "identity"),
        default="correlation_knn",
        help="Fixed adjacency fallback used only by --architecture gcn.",
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=4,
        help="Neighbors per node (dynamic per layer for DGCNN).",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or (
        f"{args.architecture}_sub-{args.subject:02d}_"
        f"{'smoke' if args.smoke else 'full'}_seed-{args.seed}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    return Config(
        data_root=str(args.data_root.resolve()),
        output_root=str(args.output_root.resolve()),
        run_name=run_name,
        architecture=args.architecture,
        subject=args.subject,
        stim_cache=args.stim_cache,
        seed=args.seed,
        val_videos=args.val_videos,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        patience=args.patience,
        graph=args.graph,
        graph_k=args.graph_k,
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


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def load_stimulus_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    with np.load(path) as cache:
        if not {"ids", "X"}.issubset(cache.files):
            raise ValueError(f"{path} must contain arrays named 'ids' and 'X'")
        ids = np.asarray(cache["ids"], dtype=np.int64)
        features = np.asarray(cache["X"], dtype=np.float32)
    if len(ids) != len(features) or len(np.unique(ids)) != len(ids):
        raise ValueError("Stimulus cache IDs must be unique and aligned one-to-one with X")
    return ids, features, {int(video_id): i for i, video_id in enumerate(ids)}


def features_for_ids(
    ids: np.ndarray, features: np.ndarray, index: dict[int, int], requested_ids: np.ndarray
) -> np.ndarray:
    del ids  # Explicit index lookup prevents accidental row-position alignment.
    missing = [int(v) for v in requested_ids if int(v) not in index]
    if missing:
        raise KeyError(f"Stimulus cache is missing video IDs: {missing[:10]}")
    return np.stack([features[index[int(v)]] for v in requested_ids]).astype(np.float32)


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


def split_video_ids(ids: np.ndarray, val_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = np.unique(ids)
    if val_count <= 1 or val_count >= len(unique_ids):
        raise ValueError(f"val_videos must be in [2, {len(unique_ids) - 1}]")
    shuffled = np.random.default_rng(seed).permutation(unique_ids)
    val_set = set(int(v) for v in shuffled[:val_count])
    # Filtering the original ID array preserves its trial order.
    train_ids = ids[np.array([int(v) not in val_set for v in ids])]
    val_ids = ids[np.array([int(v) in val_set for v in ids])]
    return train_ids, val_ids


def select_rows(all_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    selected = set(int(v) for v in selected_ids)
    return np.flatnonzero(np.array([int(v) in selected for v in all_ids]))


def fit_eeg_normalizer(train_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_window.mean(axis=(0, 2), keepdims=True)
    std = train_window.std(axis=(0, 2), keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def normalize_eeg(
    eeg_window: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_sd: float
) -> np.ndarray:
    return np.clip((eeg_window - mean) / std, -clip_sd, clip_sd).astype(np.float32)


def build_graph(train_eeg: np.ndarray, mode: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_channels = train_eeg.shape[1]
    if mode == "identity":
        adjacency = np.zeros((n_channels, n_channels), dtype=np.float32)
    else:
        if not 1 <= k < n_channels:
            raise ValueError(f"graph_k must be between 1 and {n_channels - 1}")
        flattened = train_eeg.transpose(1, 0, 2).reshape(n_channels, -1)
        correlations = np.abs(np.corrcoef(flattened)).astype(np.float32)
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(correlations, 0.0)
        adjacency = np.zeros_like(correlations)
        for channel in range(n_channels):
            neighbors = np.argsort(correlations[channel])[-k:]
            adjacency[channel, neighbors] = 1.0
        adjacency = np.maximum(adjacency, adjacency.T)

    with_loops = adjacency + np.eye(n_channels, dtype=np.float32)
    degree = with_loops.sum(axis=1)
    inv_sqrt_degree = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = inv_sqrt_degree[:, None] * with_loops * inv_sqrt_degree[None, :]
    return adjacency, normalized.astype(np.float32)


def fit_feature_normalizer(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def row_standardized_scores(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Repository-compatible row-wise standardized correlation score matrix."""
    queries = np.asarray(queries, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    qz = (queries - queries.mean(axis=1, keepdims=True)) / (
        queries.std(axis=1, keepdims=True) + 1e-8
    )
    cz = (candidates - candidates.mean(axis=1, keepdims=True)) / (
        candidates.std(axis=1, keepdims=True) + 1e-8
    )
    return (qz @ cz.T / queries.shape[1]).astype(np.float32)


def predict_model(
    model: nn.Module, eeg: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    loader = DataLoader(torch.from_numpy(eeg), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            prediction, embedding = model(batch.to(device))
            predictions.append(prediction.cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embeddings)


def collect_dynamic_neighbors(
    model: TwoLayerDGCNN, eeg: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collect both learned per-trial graph neighborhoods in original row order."""
    model.eval()
    first_graphs: list[np.ndarray] = []
    second_graphs: list[np.ndarray] = []
    loader = DataLoader(torch.from_numpy(eeg), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            first, second = model.neighbor_indices(batch.to(device))
            first_graphs.append(first.cpu().numpy())
            second_graphs.append(second.cpu().numpy())
    return np.concatenate(first_graphs), np.concatenate(second_graphs)


def retrieval_result(
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scores = row_standardized_scores(query_vectors, candidate_vectors)
    ranked = ranked_from_scores(scores, candidate_ids)
    metrics = evaluate_ranking(ranked, query_ids)
    metrics["ranks"] = metrics["ranks"].tolist()
    return scores, ranked, metrics


def train_gcn(
    config: Config,
    model: nn.Module,
    train_eeg: np.ndarray,
    train_targets: np.ndarray,
    val_eeg: np.ndarray,
    val_query_ids: np.ndarray,
    val_candidates: np.ndarray,
    val_candidate_ids: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    generator = torch.Generator().manual_seed(config.seed)
    dataset = TensorDataset(torch.from_numpy(train_eeg), torch.from_numpy(train_targets))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    max_epochs = min(config.max_epochs, 2) if config.smoke else config.max_epochs
    patience = min(config.patience, 2) if config.smoke else config.patience

    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for eeg_batch, target_batch in loader:
            eeg_batch = eeg_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(eeg_batch)
            loss = criterion(prediction, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(eeg_batch)
            total_examples += len(eeg_batch)

        val_predictions, _ = predict_model(model, val_eeg, device, config.batch_size)
        val_loss = float(np.mean((val_predictions - val_candidates[[
            int(np.where(val_candidate_ids == video_id)[0][0]) for video_id in val_query_ids
        ]]) ** 2))
        _, _, val_metrics = retrieval_result(
            val_predictions, val_candidates, val_query_ids, val_candidate_ids
        )
        train_loss = total_loss / max(total_examples, 1)
        record = {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss,
            "validation_top1": val_metrics["top1"],
            "validation_top5": val_metrics["top5"],
            "validation_mrr": val_metrics["mrr"],
            "validation_median_rank": val_metrics["median_rank"],
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.5f} val_mse={val_loss:.5f} "
            f"val_top1={val_metrics['top1']:.4f} val_top5={val_metrics['top5']:.4f} "
            f"val_mrr={val_metrics['mrr']:.4f} val_med={val_metrics['median_rank']:.1f}"
        )

        key = (
            float(val_metrics["mrr"]),
            float(val_metrics["top5"]),
            -float(val_metrics["median_rank"]),
        )
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    return best_state, history, best_epoch


def fit_pca_ridge_baseline(
    train_video_features: np.ndarray, train_eeg: np.ndarray
) -> dict[str, Any]:
    first_scaler = StandardScaler().fit(train_video_features)
    scaled = first_scaler.transform(train_video_features)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=0).fit(scaled)
    projected = pca.transform(scaled)
    second_scaler = StandardScaler().fit(projected)
    model_input = second_scaler.transform(projected)
    ridge = RidgeCV(alphas=np.logspace(-3, 6, 20)).fit(
        model_input, train_eeg.reshape(len(train_eeg), -1)
    )
    return {"sc": first_scaler, "pca": pca, "sc2": second_scaler, "model": ridge}


def baseline_predict(bundle: dict[str, Any], features: np.ndarray, n_time: int) -> np.ndarray:
    x = bundle["sc"].transform(features)
    x = bundle["pca"].transform(x)
    x = bundle["sc2"].transform(x)
    return bundle["model"].predict(x).reshape(len(features), 17, n_time).astype(np.float32)


def baseline_retrieval(
    measured_eeg: np.ndarray,
    candidate_predicted_eeg: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return retrieval_result(
        measured_eeg.reshape(len(measured_eeg), -1),
        candidate_predicted_eeg.reshape(len(candidate_predicted_eeg), -1),
        query_ids,
        candidate_ids,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
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
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {output_dir}\n"
            "Choose a different --run-name."
        )
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    stim_path = Path(config.stim_cache)
    if not stim_path.is_absolute():
        stim_path = data_root / stim_path
    stim_ids, all_features, stim_index = load_stimulus_cache(stim_path)
    all_ids, all_eeg = load_training_eeg(data_root, config.subject)
    available = np.array([int(v) in stim_index for v in all_ids])
    all_ids, all_eeg = all_ids[available], all_eeg[available]

    train_ids, val_ids = split_video_ids(all_ids, config.val_videos, config.seed)
    train_rows = select_rows(all_ids, train_ids)
    val_rows = select_rows(all_ids, val_ids)
    if config.smoke:
        train_rows = train_rows[: min(128, len(train_rows))]
        train_ids = all_ids[train_rows]
        val_rows = val_rows[: min(32, len(val_rows))]

    raw_train_window = all_eeg[train_rows, :, config.window_start : config.window_end]
    raw_val_window = all_eeg[val_rows, :, config.window_start : config.window_end]
    eeg_mean, eeg_std = fit_eeg_normalizer(raw_train_window)
    train_eeg = normalize_eeg(raw_train_window, eeg_mean, eeg_std, config.clip_sd)
    val_eeg = normalize_eeg(raw_val_window, eeg_mean, eeg_std, config.clip_sd)
    val_query_ids = all_ids[val_rows]

    train_features_raw = features_for_ids(stim_ids, all_features, stim_index, train_ids)
    val_features_raw = features_for_ids(stim_ids, all_features, stim_index, val_ids)
    feature_mean, feature_std = fit_feature_normalizer(train_features_raw)
    train_targets = ((train_features_raw - feature_mean) / feature_std).astype(np.float32)
    val_candidates = ((val_features_raw - feature_mean) / feature_std).astype(np.float32)

    expected_shapes = {
        "train_eeg": list(train_eeg.shape),
        "validation_eeg": list(val_eeg.shape),
        "train_video_targets": list(train_targets.shape),
        "validation_candidates": list(val_candidates.shape),
    }
    if config.architecture == "gcn":
        adjacency, normalized_adjacency = build_graph(
            train_eeg, mode=config.graph, k=config.graph_k
        )
        np.save(output_dir / "adjacency.npy", adjacency)
        np.save(output_dir / "adjacency_normalized.npy", normalized_adjacency)
        expected_shapes["adjacency"] = list(normalized_adjacency.shape)
        model: nn.Module = TwoLayerGCN(
            input_dim=train_eeg.shape[-1],
            target_dim=train_targets.shape[-1],
            adjacency=normalized_adjacency,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
        )
    else:
        expected_shapes["dynamic_neighbors_per_node"] = config.graph_k
        model = TwoLayerDGCNN(
            input_dim=train_eeg.shape[-1],
            target_dim=train_targets.shape[-1],
            k=config.graph_k,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
        )

    np.savez_compressed(
        output_dir / "normalization.npz",
        eeg_mean=eeg_mean,
        eeg_std=eeg_std,
        video_feature_mean=feature_mean,
        video_feature_std=feature_std,
    )
    np.savez_compressed(
        output_dir / "split.npz",
        train_video_ids=train_ids,
        validation_video_ids=val_ids,
        train_query_ids=train_ids,
        validation_query_ids=val_query_ids,
    )

    print(f"architecture={config.architecture}; device={device}; shapes={expected_shapes}")
    assert train_eeg.shape[1:] == (17, 55)
    assert len(val_ids) == config.val_videos

    model = model.to(device)
    with torch.no_grad():
        shape_prediction, shape_embedding = model(torch.from_numpy(train_eeg[:2]).to(device))
    assert shape_prediction.shape == (min(2, len(train_eeg)), train_targets.shape[1])
    assert shape_embedding.shape == (min(2, len(train_eeg)), config.embedding_dim)

    best_state, history, best_epoch = train_gcn(
        config,
        model,
        train_eeg,
        train_targets,
        val_eeg,
        val_query_ids,
        val_candidates,
        val_ids,
        device,
    )
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(config),
            "best_epoch": best_epoch,
            "input_dim": train_eeg.shape[-1],
            "target_dim": train_targets.shape[-1],
        },
        output_dir / "model.pt",
    )
    (output_dir / "history.json").write_text(
        json.dumps(json_ready(history), indent=2), encoding="utf-8"
    )

    model_tag = config.architecture
    train_predictions, train_embeddings = predict_model(
        model, train_eeg, device, config.batch_size
    )
    val_predictions, val_embeddings = predict_model(model, val_eeg, device, config.batch_size)
    model_train_scores, model_train_ranked, model_train_metrics = retrieval_result(
        train_predictions, train_targets, train_ids, train_ids
    )
    model_val_scores, model_val_ranked, model_val_metrics = retrieval_result(
        val_predictions, val_candidates, val_query_ids, val_ids
    )
    np.savez_compressed(
        output_dir / f"{model_tag}_outputs.npz",
        train_query_ids=train_ids,
        train_candidate_ids=train_ids,
        train_scores=model_train_scores,
        train_rankings=model_train_ranked,
        train_embeddings=train_embeddings,
        validation_query_ids=val_query_ids,
        validation_candidate_ids=val_ids,
        validation_scores=model_val_scores,
        validation_rankings=model_val_ranked,
        validation_embeddings=val_embeddings,
    )
    if isinstance(model, TwoLayerDGCNN):
        val_graph1, val_graph2 = collect_dynamic_neighbors(
            model, val_eeg, device, config.batch_size
        )
        np.savez_compressed(
            output_dir / "dgcnn_validation_graphs.npz",
            validation_query_ids=val_query_ids,
            layer1_neighbors=val_graph1,
            layer2_neighbors=val_graph2,
        )

    print("Fitting PCA + Ridge baseline on the identical fitting trials...")
    baseline = fit_pca_ridge_baseline(train_features_raw, train_eeg)
    joblib.dump(baseline, output_dir / "pca_ridge_baseline.joblib", compress=3)
    baseline_train_candidates = baseline_predict(
        baseline, train_features_raw, train_eeg.shape[-1]
    )
    baseline_val_candidates = baseline_predict(
        baseline, val_features_raw, val_eeg.shape[-1]
    )
    baseline_train_scores, baseline_train_ranked, baseline_train_metrics = baseline_retrieval(
        train_eeg, baseline_train_candidates, train_ids, train_ids
    )
    baseline_val_scores, baseline_val_ranked, baseline_val_metrics = baseline_retrieval(
        val_eeg, baseline_val_candidates, val_query_ids, val_ids
    )
    np.savez_compressed(
        output_dir / "pca_ridge_outputs.npz",
        train_query_ids=train_ids,
        train_candidate_ids=train_ids,
        train_scores=baseline_train_scores,
        train_rankings=baseline_train_ranked,
        validation_query_ids=val_query_ids,
        validation_candidate_ids=val_ids,
        validation_scores=baseline_val_scores,
        validation_rankings=baseline_val_ranked,
    )

    test_output = None
    if config.predict_test:
        raw_test = load_test_eeg(data_root, config.subject)
        test_eeg = normalize_eeg(
            raw_test[:, :, config.window_start : config.window_end],
            eeg_mean,
            eeg_std,
            config.clip_sd,
        )
        test_candidate_ids = np.arange(1001, 1103, dtype=np.int64)
        test_features_raw = features_for_ids(
            stim_ids, all_features, stim_index, test_candidate_ids
        )
        test_candidates = ((test_features_raw - feature_mean) / feature_std).astype(np.float32)
        test_predictions, test_embeddings = predict_model(
            model, test_eeg, device, config.batch_size
        )
        test_scores = row_standardized_scores(test_predictions, test_candidates)
        test_rankings = ranked_from_scores(test_scores, test_candidate_ids)
        np.savez_compressed(
            output_dir / f"{model_tag}_test_outputs.npz",
            test_row_indices=np.arange(len(test_eeg), dtype=np.int64),
            candidate_ids=test_candidate_ids,
            scores=test_scores,
            rankings=test_rankings,
            embeddings=test_embeddings,
        )
        # Kept inside the experiment directory; does not overwrite official submissions.
        np.save(
            output_dir / f"sub-{config.subject:02d}_{model_tag}_ranked_candidate_list.npy",
            test_rankings,
        )
        test_output = {
            "eeg_shape": list(test_eeg.shape),
            "scores_shape": list(test_scores.shape),
            "rankings_shape": list(test_rankings.shape),
            "candidate_id_min": int(test_candidate_ids.min()),
            "candidate_id_max": int(test_candidate_ids.max()),
            "original_test_row_order_preserved": True,
        }

    final_train_prediction, _ = predict_model(model, train_eeg, device, config.batch_size)
    final_val_prediction, _ = predict_model(model, val_eeg, device, config.batch_size)
    metrics = {
        "architecture": config.architecture,
        "subject": config.subject,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "selection_metric": "validation MRR, then Top-5, then median rank",
        "shapes": expected_shapes,
        model_tag: {
            "train": model_train_metrics,
            "validation": model_val_metrics,
            "train_mse": float(np.mean((final_train_prediction - train_targets) ** 2)),
            "validation_mse": float(
                np.mean((final_val_prediction - val_candidates[[
                    int(np.where(val_ids == video_id)[0][0]) for video_id in val_query_ids
                ]]) ** 2)
            ),
        },
        "pca_ridge": {
            "train": baseline_train_metrics,
            "validation": baseline_val_metrics,
            "alpha": float(baseline["model"].alpha_),
            "pca_components": int(baseline["pca"].n_components_),
        },
        "test_output": test_output,
        "safeguards": {
            "split_from_training_data_only": True,
            "normalization_fit_on_training_split_only": True,
            "graph_policy": (
                "train-only fixed graph"
                if config.architecture == "gcn"
                else "label-free per-trial dynamic graphs"
            ),
            "test_labels_loaded": False,
            "frozen_video_cache_modified": False,
            "existing_prediction_or_submission_files_modified": False,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2), encoding="utf-8"
    )

    def line(name: str, values: dict[str, Any]) -> str:
        return (
            f"{name:<18} top1={values['top1']:.4f} top5={values['top5']:.4f} "
            f"median_rank={values['median_rank']:.1f} mrr={values['mrr']:.4f}"
        )

    architecture_label = config.architecture.upper()
    print("\nFINAL COMPARISON")
    print(line(f"{architecture_label} train", model_train_metrics))
    print(line(f"{architecture_label} validation", model_val_metrics))
    print(line("Ridge train", baseline_train_metrics))
    print(line("Ridge validation", baseline_val_metrics))
    print(f"Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
