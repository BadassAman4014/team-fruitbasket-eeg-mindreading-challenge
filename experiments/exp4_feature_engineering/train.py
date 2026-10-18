"""Exp4 + Feature Engineering & Channel Selection Training Pipeline.

Integrates:
1. Channel subset selection (--channels core8 / top12 / all17 / custom list)
2. EEG Feature Engineering (--eeg-features raw / bandpower / both / erp_stats / rich / full)
3. Tuned MLP architecture matching Exp4 (DINOv2 + CLAP audio fusion)
4. Full-fit training on all 1-1000 stimuli across subjects 1..6
5. Mild-CSLS anti-hubness with automatic lambda search on held-out validation
6. Optional --auto-importance to run channel/feature attribution immediately after training
"""

from __future__ import annotations

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp4_feature_engineering.feature_extractor import (
    ALL_CHANNELS, CHANNEL_TO_IDX, PREDEFINED_CHANNEL_SETS,
    compute_feature_dim, extract_features, resolve_channel_indices,
)
from experiments.exp4_feature_engineering.model import (
    DinoPool, ProjectionMLP, TunedMLPRetriever, contrastive_loss
)
from experiments.exp4_tuned_mlp_dino_audio.train import (
    attention_summary, choose_csls_lambda,
    load_dino_frames, load_pooled_eeg, load_test_eeg, mild_csls,
    metrics_from_scores, normalize_trials, seed_everything,
    stack_dino_frames, evaluate_ranking, ranked_from_scores,
    WINDOW, N_SUBJECTS
)
from experiments.exp2_dgcnn_dino_audio_mlp.run_experiment import (
    apply_normalizer, fit_normalizer, load_audio_map, resolve_path, stack_features
)

TRAIN_IDS = np.arange(1, 901, dtype=np.int64)
HELDOUT_IDS = np.arange(901, 1001, dtype=np.int64)
TEST_IDS = np.arange(1001, 1103, dtype=np.int64)
PUBLIC_SUBJECTS = frozenset((1, 2, 3, 4))
_FRAME_KEY = re.compile(r"^(\d+)_f(\d+)$")


@dataclass(frozen=True)
class Config:
    data_root: str
    output_root: str
    run_name: str
    channels: str = "all17"
    eeg_features: str = "raw"
    subjects: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
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
    fusion: str = "concat"
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
    predict_test: bool = True
    report_public_subjects: bool = True
    final_full_fit: bool = True
    auto_importance: bool = False
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
        default=REPO_ROOT / "experiments" / "exp4_feature_engineering" / "runs",
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--channels",
        default="all17",
        help="EEG channels: 'all17', 'core8', 'top12', 'occipital', or comma-separated channel names (e.g. Oz,P8,PO8,O2...)",
    )
    parser.add_argument(
        "--eeg-features",
        default="raw",
        choices=("raw", "bandpower", "both", "erp_stats", "rich", "full"),
        help="Feature engineering representation: 'raw' (Exp4 baseline), 'bandpower' (Delta/Theta/Alpha/Beta), "
             "'both' (raw+bandpower), 'erp_stats' (raw+P100/N170/P300 window stats), 'rich', 'full'",
    )
    parser.add_argument("--subjects", type=parse_csv_ints, default=(1, 2, 3, 4, 5, 6))
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
    parser.add_argument("--fusion", choices=("concat", "separate"), default="concat")
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
        help="Retrain each seed on ALL 1-1000 stimuli for selected epoch count.",
    )
    parser.add_argument(
        "--auto-importance",
        action="store_true",
        help="Automatically run electrode importance analysis after training.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    for name in ("subjects", "prediction_subjects"):
        values = getattr(args, name)
        invalid = [value for value in values if value not in range(1, N_SUBJECTS + 1)]
        if invalid:
            parser.error(f"--{name.replace('_', '-')} values must be 1..6; got {invalid}")
        if len(set(values)) != len(values):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")

    if args.csls_lambda != "auto":
        try:
            value = float(args.csls_lambda)
            if value < 0:
                raise ValueError()
        except ValueError:
            parser.error("--csls-lambda must be 'auto' or non-negative float")

    return Config(
        data_root=str(args.data_root),
        output_root=str(args.output_root),
        run_name=args.run_name
        or f"exp4-fe-{args.channels}-{args.eeg_features}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        channels=args.channels,
        eeg_features=args.eeg_features,
        subjects=tuple(args.subjects),
        prediction_subjects=tuple(args.prediction_subjects),
        seeds=tuple(args.seeds),
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
        epochs=2 if args.smoke else args.epochs,
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
        auto_importance=args.auto_importance,
        smoke=args.smoke,
        self_test=args.self_test,
    )


def encode_features_eeg(
    model: TunedMLPRetriever,
    eeg_features: np.ndarray,
    subjects: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(eeg_features), batch_size):
            x = torch.from_numpy(eeg_features[start : start + batch_size]).to(device)
            s = torch.from_numpy(subjects[start : start + batch_size]).to(device)
            output.append(model.encode_eeg(x, s).cpu().numpy())
    return np.concatenate(output)


def compute_raw_scores(
    model: TunedMLPRetriever,
    eeg_features: np.ndarray,
    subjects: np.ndarray,
    candidate_dino: np.ndarray,
    candidate_audio: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    query = encode_features_eeg(model, eeg_features, subjects, device, batch_size)
    model.eval()
    cand_d = torch.from_numpy(candidate_dino).to(device)
    cand_a = torch.from_numpy(candidate_audio).to(device)
    with torch.no_grad():
        candidates = model.encode_stimulus(cand_d, cand_a).cpu().numpy()
    return (query @ candidates.T).astype(np.float32)


def train_seed(
    config: Config,
    seed: int,
    train_feats: np.ndarray,
    train_subjects: np.ndarray,
    train_ids: np.ndarray,
    train_dino: np.ndarray,
    train_audio: np.ndarray,
    heldout_feats: np.ndarray,
    heldout_subjects: np.ndarray,
    heldout_query_ids: np.ndarray,
    heldout_dino_cands: np.ndarray,
    heldout_audio_cands: np.ndarray,
    device: torch.device,
    fixed_epochs: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    seed_everything(seed)
    eeg_dim = train_feats.shape[1]
    model = TunedMLPRetriever(
        eeg_dim=eeg_dim,
        dino_dim=train_dino.shape[2],
        audio_dim=train_audio.shape[1],
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

    dataset = TensorDataset(
        torch.from_numpy(train_feats),
        torch.from_numpy(train_subjects),
        torch.from_numpy(train_ids),
        torch.from_numpy(train_dino),
        torch.from_numpy(train_audio),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    epochs = fixed_epochs if fixed_epochs is not None else config.epochs

    best_key = None
    best_state = None
    best_epoch = epochs
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        contrastive_sum = 0.0
        count = 0
        for eeg_b, sub_b, ids_b, dino_b, audio_b in loader:
            optimizer.zero_grad(set_to_none=True)
            eeg_b = eeg_b.to(device)
            sub_b = sub_b.to(device)
            ids_b = ids_b.to(device)
            dino_b = dino_b.to(device)
            audio_b = audio_b.to(device)
            eeg_embed, stim_embed = model(eeg_b, sub_b, dino_b, audio_b)
            c_loss = contrastive_loss(
                eeg_embed, stim_embed, model.log_scale, ids_b, config.loss
            )
            total_loss = c_loss + config.subject_affine_reg * model.affine_penalty()
            total_loss.backward()
            optimizer.step()
            loss_sum += float(total_loss.item()) * len(eeg_b)
            contrastive_sum += float(c_loss.item()) * len(eeg_b)
            count += len(eeg_b)

        if fixed_epochs is not None:
            if epoch % 5 == 0 or epoch == epochs:
                print(f"seed={seed} full-fit epoch={epoch:02d}/{epochs} loss={loss_sum / count:.4f}")
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue

        # Heldout validation
        scores = compute_raw_scores(
            model, heldout_feats, heldout_subjects, heldout_dino_cands, heldout_audio_cands, device, config.batch_size
        )
        metrics = metrics_from_scores(scores, HELDOUT_IDS, heldout_query_ids)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "heldout_top1": metrics["top1"],
            "heldout_top5": metrics["top5"],
            "heldout_mrr": metrics["mrr"],
            "median_rank": metrics["median_rank"],
        }
        history.append(record)
        key = (float(metrics["top1"]), float(metrics["top5"]), float(metrics["mrr"]))
        marker = ""
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            marker = " [best]"
        print(
            f"seed={seed} epoch={epoch:02d}/{epochs} loss={record['train_loss']:.4f} "
            f"heldout top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} "
            f"median={metrics['median_rank']:.1f}{marker}"
        )

    assert best_state is not None
    return best_state, history, best_epoch


def main() -> None:
    config = parse_args()

    if config.self_test:
        from experiments.exp4_feature_engineering.feature_importance import synthetic_self_test
        synthetic_self_test()
        print("Exp4 feature engineering self-test passed successfully!")
        return

    data_root = Path(config.data_root).resolve()
    output_dir = Path(config.output_root).resolve() / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (config.device == "auto" and torch.cuda.is_available()) or config.device.startswith("cuda") else "cpu"
    )

    channel_indices, channel_names = resolve_channel_indices(config.channels)
    n_ch = len(channel_indices)
    t_len = WINDOW.stop - WINDOW.start
    eeg_dim = compute_feature_dim(n_ch, t_len, config.eeg_features)

    print("=" * 80)
    print(f"EXP4 + FEATURE ENGINEERING: {config.run_name}")
    print(f"Channels: {n_ch}/17 electrodes {channel_names}")
    print(f"EEG Feature Mode: '{config.eeg_features}' -> Feature Dim: {eeg_dim}")
    print(f"Subjects: {config.subjects} | Seeds: {config.seeds} | Audio: {config.audio_model} | Fusion: {config.fusion}")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load EEG
    train_eeg_raw, train_query_ids, train_subjects, train_counts = load_pooled_eeg(
        data_root, config.subjects, 1, 900
    )
    heldout_eeg_raw, heldout_query_ids, heldout_subjects, heldout_counts = load_pooled_eeg(
        data_root, config.subjects, 901, 1000
    )

    # Extract engineered EEG features
    train_feats = extract_features(train_eeg_raw, config.eeg_features, channel_indices=channel_indices)
    heldout_feats = extract_features(heldout_eeg_raw, config.eeg_features, channel_indices=channel_indices)

    # 2. Load Stimulus Features
    dino_cache = resolve_path(data_root, config.dino_train_cache)
    train_dino_map = load_dino_frames(dino_cache, np.arange(1, 1001), config.dino_frames)
    audio_cache = data_root / "features_audio" / f"train_{config.audio_model}_features.npz"
    train_audio_map = load_audio_map(audio_cache)

    train_dino_raw = stack_dino_frames(train_dino_map, np.arange(1, 1001), "train DINO")
    train_audio_raw = stack_features(train_audio_map, np.arange(1, 1001), "train audio")

    # Fit normalizers
    dino_mean, dino_std = fit_normalizer(train_dino_raw.mean(axis=1))
    audio_mean, audio_std = fit_normalizer(train_audio_raw)

    train_dino_norm = apply_normalizer(train_dino_raw, dino_mean[:, None, :], dino_std[:, None, :])
    train_audio_norm = apply_normalizer(train_audio_raw, audio_mean, audio_std)

    train_dino = train_dino_norm[train_query_ids - 1]
    train_audio = train_audio_norm[train_query_ids - 1]

    heldout_dino_cands = train_dino_norm[HELDOUT_IDS - 1]
    heldout_audio_cands = train_audio_norm[HELDOUT_IDS - 1]

    np.savez_compressed(
        output_dir / "normalization.npz",
        dino_mean=dino_mean,
        dino_std=dino_std,
        audio_mean=audio_mean,
        audio_std=audio_std,
    )

    # Save run configuration
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    # Train each seed on 1-900 with held-out selection
    states: dict[int, dict[str, torch.Tensor]] = {}
    best_epochs: dict[int, int] = {}
    seed_histories: dict[int, list[dict[str, Any]]] = {}

    for seed in config.seeds:
        print(f"\n--- Training Seed {seed} ---")
        state, history, best_ep = train_seed(
            config, seed, train_feats, train_subjects, train_query_ids,
            train_dino, train_audio,
            heldout_feats, heldout_subjects, heldout_query_ids,
            heldout_dino_cands, heldout_audio_cands,
            device
        )
        states[seed] = state
        best_epochs[seed] = best_ep
        seed_histories[seed] = history
        torch.save(
            {"model_state_dict": state, "seed": seed, "best_epoch": best_ep, "config": asdict(config)},
            output_dir / f"model_seed-{seed}.pt"
        )

    # Selection ensemble evaluation on heldout
    score_sum = np.zeros((len(heldout_feats), len(HELDOUT_IDS)), dtype=np.float32)
    for seed in config.seeds:
        m = TunedMLPRetriever(
            eeg_dim=eeg_dim, dino_dim=train_dino.shape[2], audio_dim=train_audio.shape[1],
            hidden_dim=config.hidden_dim, embedding_dim=config.embedding_dim,
            dropout=config.dropout, temperature=config.temperature, fusion=config.fusion,
            modality_dropout=0.0, subject_affine=config.subject_affine,
            dino_pooling=config.dino_pooling, dino_frames=config.dino_frames,
            attention_hidden_dim=config.attention_hidden_dim,
        ).to(device)
        m.load_state_dict(states[seed])
        score_sum += compute_raw_scores(
            m, heldout_feats, heldout_subjects, heldout_dino_cands, heldout_audio_cands, device, config.batch_size
        )
    heldout_raw = score_sum / len(config.seeds)
    selected_lambda, heldout_metrics = choose_csls_lambda(
        heldout_raw, HELDOUT_IDS, heldout_query_ids, config.csls_k, config.csls_lambda
    )
    print("\n" + "=" * 70)
    print("ENSEMBLE HELDOUT EVALUATION (Selection fit on 1-900)")
    print(f"Top-1={heldout_metrics['top1']:.4f} | Top-5={heldout_metrics['top5']:.4f} | MRR={heldout_metrics['mrr']:.4f} | Med={heldout_metrics['median_rank']:.1f}")
    print(f"Selected CSLS lambda: {selected_lambda}")
    print("=" * 70)

    # Save heldout outputs
    np.savez_compressed(
        output_dir / "heldout_outputs.npz",
        raw_scores=heldout_raw,
        query_ids=heldout_query_ids,
        query_subjects=heldout_subjects,
        candidate_ids=HELDOUT_IDS,
        selected_csls_lambda=np.float32(selected_lambda),
    )

    # Final Full Fit
    if config.final_full_fit:
        full_eeg_raw = np.concatenate((train_eeg_raw, heldout_eeg_raw), axis=0)
        full_feats = extract_features(full_eeg_raw, config.eeg_features, channel_indices=channel_indices)
        full_subjects = np.concatenate((train_subjects, heldout_subjects), axis=0)
        full_ids = np.concatenate((train_query_ids, heldout_query_ids), axis=0)
        full_dino = np.concatenate((train_dino, heldout_dino_cands[heldout_query_ids - 901]), axis=0)
        full_audio = np.concatenate((train_audio, heldout_audio_cands[heldout_query_ids - 901]), axis=0)

        print("\n" + "=" * 70)
        print(f"FINAL FULL FIT: Retraining each seed on ALL 1-1000 stimuli ({len(full_feats)} trials) for selected epochs: {best_epochs}")
        print("=" * 70)
        for seed in config.seeds:
            full_state, _, _ = train_seed(
                config, seed, full_feats, full_subjects, full_ids,
                full_dino, full_audio,
                heldout_feats, heldout_subjects, heldout_query_ids,
                heldout_dino_cands, heldout_audio_cands,
                device, fixed_epochs=max(1, int(best_epochs[seed]))
            )
            states[seed] = full_state
            torch.save(
                {"model_state_dict": full_state, "seed": seed, "fixed_epochs": int(best_epochs[seed]), "config": asdict(config)},
                output_dir / f"model_seed-{seed}_fullfit.pt"
            )

    # Test Prediction & Public Reporting
    if config.predict_test:
        test_dino_cache = resolve_path(data_root, config.dino_test_cache)
        test_dino_map = load_dino_frames(test_dino_cache, TEST_IDS, config.dino_frames)
        test_audio_cache = data_root / "features_audio" / f"test_{config.audio_model}_features.npz"
        test_audio_map = load_audio_map(test_audio_cache)
        test_dino = apply_normalizer(
            stack_dino_frames(test_dino_map, TEST_IDS, "test DINO"), dino_mean[:, None, :], dino_std[:, None, :]
        )
        test_audio = apply_normalizer(stack_features(test_audio_map, TEST_IDS, "test audio"), audio_mean, audio_std)

        submissions_dir = output_dir / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        public_metrics = {}

        for subject in config.prediction_subjects:
            raw_test_eeg = load_test_eeg(data_root, subject)[:, :, WINDOW]
            norm_test_eeg = normalize_trials(raw_test_eeg)
            test_feats = extract_features(norm_test_eeg, config.eeg_features, channel_indices=channel_indices)
            sub_ids = np.full(len(test_feats), subject, dtype=np.int64)

            sub_score_sum = np.zeros((len(test_feats), len(TEST_IDS)), dtype=np.float32)
            for seed in config.seeds:
                m = TunedMLPRetriever(
                    eeg_dim=eeg_dim, dino_dim=test_dino.shape[2], audio_dim=test_audio.shape[1],
                    hidden_dim=config.hidden_dim, embedding_dim=config.embedding_dim,
                    dropout=config.dropout, temperature=config.temperature, fusion=config.fusion,
                    modality_dropout=0.0, subject_affine=config.subject_affine,
                    dino_pooling=config.dino_pooling, dino_frames=config.dino_frames,
                    attention_hidden_dim=config.attention_hidden_dim,
                ).to(device)
                m.load_state_dict(states[seed])
                sub_score_sum += compute_raw_scores(
                    m, test_feats, sub_ids, test_dino, test_audio, device, config.batch_size
                )
            raw_scores = sub_score_sum / len(config.seeds)
            adj_scores = mild_csls(raw_scores, config.csls_k, selected_lambda) if config.anti_hub == "csls" else raw_scores
            rankings = ranked_from_scores(adj_scores, TEST_IDS)

            np.savez_compressed(
                output_dir / f"sub-{subject:02d}_test_outputs.npz",
                test_row_indices=np.arange(len(raw_scores), dtype=np.int64),
                candidate_ids=TEST_IDS,
                raw_scores=raw_scores,
                adjusted_scores=adj_scores,
                rankings=rankings,
                selected_csls_lambda=np.float32(selected_lambda),
            )
            np.save(submissions_dir / f"exp4_fe_{config.channels}_sub-{subject:02d}_ranked_candidate_list.npy", rankings)
            if subject in (5, 6):
                np.save(submissions_dir / f"sub-{subject:02d}_ranked_candidate_list.npy", rankings)

            if config.report_public_subjects and subject in PUBLIC_SUBJECTS:
                truth_file = data_root / "tst_stim" / f"sub-{subject:02d}_tst_stim.npy"
                if truth_file.is_file():
                    truth = np.asarray(np.load(truth_file), dtype=np.int64)
                    public_metrics[subject] = metrics_from_scores(adj_scores, TEST_IDS, truth)

        if public_metrics:
            print("\nPUBLIC SUBJECT RESULTS (reporting only; never used for selection):")
            all_ranks = []
            for s in sorted(public_metrics):
                m = public_metrics[s]
                ranks_arr = np.asarray(m["ranks"])
                top1_c = int(np.sum(ranks_arr == 1))
                top5_c = int(np.sum(ranks_arr <= 5))
                print(f"sub-{s:02d}: Top1 {top1_c:3d}/102 ({m['top1']:.4f}), Top5 {top5_c:3d}/102 ({m['top5']:.4f}), median {m['median_rank']:.1f}, MRR {m['mrr']:.4f}")
                all_ranks.extend(m["ranks"])
            pooled_ranks = np.asarray(all_ranks)
            p_top1 = float(np.mean(pooled_ranks == 1))
            p_top5 = float(np.mean(pooled_ranks <= 5))
            p_mrr = float(np.mean(1.0 / pooled_ranks))
            p_med = float(np.median(pooled_ranks))
            print(f"POOLED: Top1 {int(np.sum(pooled_ranks == 1)):3d}/{len(pooled_ranks)} ({p_top1:.4f}), Top5 {int(np.sum(pooled_ranks <= 5)):3d}/{len(pooled_ranks)} ({p_top5:.4f}), median {p_med:.1f}, MRR {p_mrr:.4f}")

    if config.auto_importance:
        from experiments.exp4_feature_engineering.feature_importance import (
            print_leaderboard, run_importance_analysis
        )
        print("\n" + "=" * 80)
        print("RUNNING AUTOMATIC ELECTRODE & FEATURE IMPORTANCE ANALYSIS...")
        print("=" * 80)
        m = TunedMLPRetriever(
            eeg_dim=eeg_dim, dino_dim=train_dino.shape[2], audio_dim=train_audio.shape[1],
            hidden_dim=config.hidden_dim, embedding_dim=config.embedding_dim,
            dropout=config.dropout, temperature=config.temperature, fusion=config.fusion,
            modality_dropout=0.0, subject_affine=config.subject_affine,
            dino_pooling=config.dino_pooling, dino_frames=config.dino_frames,
            attention_hidden_dim=config.attention_hidden_dim,
        ).to(device)
        m.load_state_dict(states[config.seeds[0]])
        report = run_importance_analysis(
            m, heldout_eeg_raw, heldout_subjects, heldout_query_ids,
            heldout_dino_cands, heldout_audio_cands,
            channel_indices, channel_names, config.eeg_features, device,
            csls_lambda=selected_lambda, n_perms=3
        )
        print_leaderboard(report)
        (output_dir / "feature_importance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nArtifacts successfully saved under: {output_dir}")


if __name__ == "__main__":
    main()
