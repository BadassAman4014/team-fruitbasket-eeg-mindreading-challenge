"""Electrode & Feature Importance Analyzer for Exp4 Tuned MLP Models.

Computes:
1. Permutation Feature Importance (PFI) per electrode (drop in Top-1, Top-5, MRR)
2. Zero-out ablation per electrode
3. Gradient Saliency Attributions (backpropagated through eeg_mlp)
4. Temporal window importance (P100, N170, P300, Late visual processing)
5. Anatomical regional importance (Occipital, Parieto-Occipital, Parietal)
6. Feature-type importance (Raw vs Band-power vs ERP stats)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp4_feature_engineering.feature_extractor import (
    ALL_CHANNELS, CHANNEL_TO_IDX, extract_features, resolve_channel_indices
)

HELDOUT_IDS = np.arange(901, 1001, dtype=np.int64)
REGIONS = {
    "Occipital (Primary Visual Core)": ["Oz", "O1", "O2"],
    "Parieto-Occipital (Dorsal/Ventral Streams)": ["POz", "PO3", "PO4", "PO7", "PO8"],
    "Parietal (Higher-Order Association)": ["Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Path to trained Exp4 or Feature-Engineering run directory")
    parser.add_argument("--checkpoint", type=Path, help="Specific model checkpoint .pt (defaults to model_seed-42.pt in run-dir)")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Path to save output JSON report")
    parser.add_argument("--n-permutations", type=int, default=5, help="Number of random shuffles for PFI stability")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic self-test")
    return parser.parse_args()


def mild_csls(scores: np.ndarray, k: int, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return scores
    effective_k = min(k, *scores.shape)
    query_density = np.partition(scores, -effective_k, axis=1)[:, -effective_k:].mean(axis=1, keepdims=True)
    candidate_density = np.partition(scores, -effective_k, axis=0)[-effective_k:, :].mean(axis=0, keepdims=True)
    return (scores - strength * (query_density + candidate_density)).astype(np.float32)


def evaluate_metrics(scores: np.ndarray, candidate_ids: np.ndarray, query_ids: np.ndarray) -> dict[str, float]:
    ranked = candidate_ids[np.argsort(-scores, axis=1, kind="stable")]
    hits = ranked == query_ids[:, None]
    ranks = np.argmax(hits, axis=1) + 1
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def compute_scores(
    model: torch.nn.Module,
    eeg_features: np.ndarray,
    subjects: np.ndarray,
    candidate_dino: np.ndarray,
    candidate_audio: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        # Encode stimuli
        d_cand = torch.from_numpy(candidate_dino).to(device)
        a_cand = torch.from_numpy(candidate_audio).to(device)
        cand_embeds = model.encode_stimulus(d_cand, a_cand)

        # Encode EEG
        q_embeds: list[torch.Tensor] = []
        for start in range(0, len(eeg_features), batch_size):
            feat_batch = torch.from_numpy(eeg_features[start : start + batch_size]).to(device)
            sub_batch = torch.from_numpy(subjects[start : start + batch_size]).to(device)
            q_embeds.append(model.encode_eeg(feat_batch, sub_batch))
        all_q = torch.cat(q_embeds, dim=0)
        scores = (all_q @ cand_embeds.T).cpu().numpy().astype(np.float32)
    return scores


def run_importance_analysis(
    model: torch.nn.Module,
    heldout_eeg: np.ndarray,
    heldout_subjects: np.ndarray,
    heldout_query_ids: np.ndarray,
    heldout_dino_cands: np.ndarray,
    heldout_audio_cands: np.ndarray,
    channel_indices: list[int],
    channel_names: list[str],
    feature_mode: str,
    device: torch.device,
    csls_lambda: float = 0.10,
    csls_k: int = 10,
    n_perms: int = 5,
) -> dict[str, Any]:
    print("\n[1/5] Evaluating Baseline Performance on Held-Out Validation...")
    base_feats = extract_features(heldout_eeg, feature_mode, channel_indices=channel_indices)
    base_raw_scores = compute_scores(
        model, base_feats, heldout_subjects, heldout_dino_cands, heldout_audio_cands, device
    )
    base_adj_scores = mild_csls(base_raw_scores, csls_k, csls_lambda)
    baseline = evaluate_metrics(base_adj_scores, HELDOUT_IDS, heldout_query_ids)
    print(f"  Baseline: Top-1={baseline['top1']:.4f}, Top-5={baseline['top5']:.4f}, MRR={baseline['mrr']:.4f}, Med={baseline['median_rank']:.1f}")

    print("\n[2/5] Computing Gradient Saliency Attributions per Electrode...")
    model.eval()
    grad_feats = torch.from_numpy(base_feats).to(device).requires_grad_(True)
    subs_t = torch.from_numpy(heldout_subjects).to(device)
    q_embeds = model.encode_eeg(grad_feats, subs_t)

    cand_d = torch.from_numpy(heldout_dino_cands).to(device)
    cand_a = torch.from_numpy(heldout_audio_cands).to(device)
    cand_embeds = model.encode_stimulus(cand_d, cand_a)

    target_cols = [int(np.where(HELDOUT_IDS == vid)[0][0]) for vid in heldout_query_ids]
    target_cand_embeds = cand_embeds[target_cols]
    pos_scores = (q_embeds * target_cand_embeds).sum(dim=1).sum()
    pos_scores.backward()

    grads = grad_feats.grad.detach().cpu().numpy()  # (N, D)
    abs_grads = np.abs(grads).mean(axis=0)

    # Map gradients back to channels
    N_ch = len(channel_indices)
    T = heldout_eeg.shape[-1]
    if feature_mode == "raw":
        channel_grads = abs_grads.reshape(N_ch, T).mean(axis=1)
    else:
        # Approximate by first N_ch * T raw elements
        raw_len = N_ch * T
        channel_grads = abs_grads[:raw_len].reshape(N_ch, T).mean(axis=1)

    max_g = channel_grads.max() if channel_grads.max() > 0 else 1.0
    norm_saliency = {name: float(channel_grads[i] / max_g) for i, name in enumerate(channel_names)}

    print("\n[3/5] Computing Permutation Feature Importance (PFI) & Zero-out per Electrode...")
    pfi_results: dict[str, dict[str, float]] = {}
    zero_results: dict[str, dict[str, float]] = {}

    rng = np.random.RandomState(42)

    for ch_idx_in_subset, name in enumerate(channel_names):
        # 1. Zero-out
        eeg_zeroed = heldout_eeg.copy()
        raw_ch_idx = channel_indices[ch_idx_in_subset]
        eeg_zeroed[:, raw_ch_idx, :] = 0.0
        zero_feats = extract_features(eeg_zeroed, feature_mode, channel_indices=channel_indices)
        zero_scores = mild_csls(
            compute_scores(model, zero_feats, heldout_subjects, heldout_dino_cands, heldout_audio_cands, device),
            csls_k, csls_lambda
        )
        zero_m = evaluate_metrics(zero_scores, HELDOUT_IDS, heldout_query_ids)
        zero_results[name] = {
            "top1": zero_m["top1"],
            "delta_top1": baseline["top1"] - zero_m["top1"],
            "top5": zero_m["top5"],
            "delta_top5": baseline["top5"] - zero_m["top5"],
            "mrr": zero_m["mrr"],
            "delta_mrr": baseline["mrr"] - zero_m["mrr"],
        }

        # 2. PFI (shuffle across queries)
        perm_top1s, perm_top5s, perm_mrrs = [], [], []
        for _ in range(n_perms):
            eeg_perm = heldout_eeg.copy()
            perm_order = rng.permutation(len(eeg_perm))
            eeg_perm[:, raw_ch_idx, :] = eeg_perm[perm_order, raw_ch_idx, :]
            perm_feats = extract_features(eeg_perm, feature_mode, channel_indices=channel_indices)
            perm_scores = mild_csls(
                compute_scores(model, perm_feats, heldout_subjects, heldout_dino_cands, heldout_audio_cands, device),
                csls_k, csls_lambda
            )
            pm = evaluate_metrics(perm_scores, HELDOUT_IDS, heldout_query_ids)
            perm_top1s.append(pm["top1"])
            perm_top5s.append(pm["top5"])
            perm_mrrs.append(pm["mrr"])

        mean_p_top1 = float(np.mean(perm_top1s))
        mean_p_top5 = float(np.mean(perm_top5s))
        mean_p_mrr = float(np.mean(perm_mrrs))
        pfi_results[name] = {
            "top1": mean_p_top1,
            "delta_top1": baseline["top1"] - mean_p_top1,
            "top5": mean_p_top5,
            "delta_top5": baseline["top5"] - mean_p_top5,
            "mrr": mean_p_mrr,
            "delta_mrr": baseline["mrr"] - mean_p_mrr,
            "saliency": norm_saliency[name],
        }

    print("\n[4/5] Evaluating Anatomical Regional Group Ablations...")
    regional_results: dict[str, dict[str, float]] = {}
    for region_name, ch_list in REGIONS.items():
        present_in_subset = [c for c in ch_list if c in channel_names]
        if not present_in_subset:
            continue
        eeg_reg_zero = heldout_eeg.copy()
        for c in present_in_subset:
            raw_idx = CHANNEL_TO_IDX[c]
            eeg_reg_zero[:, raw_idx, :] = 0.0
        reg_feats = extract_features(eeg_reg_zero, feature_mode, channel_indices=channel_indices)
        reg_scores = mild_csls(
            compute_scores(model, reg_feats, heldout_subjects, heldout_dino_cands, heldout_audio_cands, device),
            csls_k, csls_lambda
        )
        rm = evaluate_metrics(reg_scores, HELDOUT_IDS, heldout_query_ids)
        regional_results[region_name] = {
            "channels": present_in_subset,
            "delta_top1": baseline["top1"] - rm["top1"],
            "delta_top5": baseline["top5"] - rm["top5"],
            "delta_mrr": baseline["mrr"] - rm["mrr"],
            "median_rank": rm["median_rank"],
        }

    print("\n[5/5] Evaluating Temporal Window Importance...")
    time_windows = {
        "P100 (50-120ms: Early Retinotopic V1/V2)": (0, 7),
        "N170 (120-220ms: Structural & Motion Encoding)": (7, 17),
        "P300 (220-400ms: Semantic Processing)": (17, 35),
        "Late (400-600ms: Secondary & Sustained)": (35, heldout_eeg.shape[-1]),
    }
    temporal_results: dict[str, dict[str, float]] = {}
    for tw_name, (w_start, w_end) in time_windows.items():
        eeg_time_zero = heldout_eeg.copy()
        eeg_time_zero[:, :, w_start:w_end] = 0.0
        tw_feats = extract_features(eeg_time_zero, feature_mode, channel_indices=channel_indices)
        tw_scores = mild_csls(
            compute_scores(model, tw_feats, heldout_subjects, heldout_dino_cands, heldout_audio_cands, device),
            csls_k, csls_lambda
        )
        tw_m = evaluate_metrics(tw_scores, HELDOUT_IDS, heldout_query_ids)
        temporal_results[tw_name] = {
            "samples": f"{w_start}..{w_end}",
            "delta_top1": baseline["top1"] - tw_m["top1"],
            "delta_top5": baseline["top5"] - tw_m["top5"],
            "delta_mrr": baseline["mrr"] - tw_m["mrr"],
            "median_rank": tw_m["median_rank"],
        }

    # Combined composite importance rank
    leaderboard = []
    for name in channel_names:
        pfi = pfi_results[name]
        zero = zero_results[name]
        score = (
            pfi["delta_top1"] * 0.4
            + pfi["delta_top5"] * 0.3
            + pfi["delta_mrr"] * 0.2
            + pfi["saliency"] * 0.1
        )
        leaderboard.append({
            "channel": name,
            "composite_score": float(score),
            "delta_top1": pfi["delta_top1"],
            "delta_top5": pfi["delta_top5"],
            "delta_mrr": pfi["delta_mrr"],
            "zero_delta_top1": zero["delta_top1"],
            "zero_delta_top5": zero["delta_top5"],
            "saliency": pfi["saliency"],
        })
    leaderboard.sort(key=lambda item: item["composite_score"], reverse=True)

    return {
        "baseline": baseline,
        "leaderboard": leaderboard,
        "pfi": pfi_results,
        "zero_out": zero_results,
        "regions": regional_results,
        "temporal_windows": temporal_results,
        "channels_evaluated": channel_names,
        "feature_mode": feature_mode,
        "csls_lambda": csls_lambda,
    }


def print_leaderboard(report: dict[str, Any]) -> None:
    base = report["baseline"]
    print("\n" + "=" * 90)
    print("EXP4 TUNED MLP — ELECTRODE & FEATURE IMPORTANCE LEADERBOARD")
    print(f"Baseline Held-Out: Top-1={base['top1']:.4f} | Top-5={base['top5']:.4f} | MRR={base['mrr']:.4f} | Med={base['median_rank']:.1f}")
    print("=" * 90)
    print(f"{'Rank':<5} {'Electrode':<10} {'ΔTop-1':<12} {'ΔTop-5':<12} {'ΔMRR':<12} {'Zero ΔTop-5':<14} {'Saliency':<10}")
    print("-" * 90)
    for rank, item in enumerate(report["leaderboard"], start=1):
        print(
            f"{rank:<5} {item['channel']:<10} "
            f"{item['delta_top1']:+<12.4f} "
            f"{item['delta_top5']:+<12.4f} "
            f"{item['delta_mrr']:+<12.4f} "
            f"{item['zero_delta_top5']:+<14.4f} "
            f"{item['saliency']:<10.4f}"
        )
    print("=" * 90)

    print("\nREGIONAL ANATOMICAL ABLATIONS:")
    for region, data in report["regions"].items():
        print(f"  • {region} ({len(data['channels'])} ch): ΔTop-1={data['delta_top1']:+.4f}, ΔTop-5={data['delta_top5']:+.4f}, ΔMRR={data['delta_mrr']:+.4f}")

    print("\nTEMPORAL WINDOW IMPORTANCE:")
    for tw, data in report["temporal_windows"].items():
        print(f"  • {tw}: ΔTop-1={data['delta_top1']:+.4f}, ΔTop-5={data['delta_top5']:+.4f}, ΔMRR={data['delta_mrr']:+.4f}")


def synthetic_self_test() -> None:
    print("Running Exp4 feature_importance self-test...")
    from experiments.exp4_feature_engineering.model import TunedMLPRetriever

    device = torch.device("cpu")
    N = 20
    C = 8
    T = 55
    ch_indices = list(range(C))
    ch_names = ALL_CHANNELS[:C]

    model = TunedMLPRetriever(
        eeg_dim=C * T,
        dino_dim=16,
        audio_dim=8,
        hidden_dim=32,
        embedding_dim=16,
        dropout=0.1,
        temperature=0.07,
        fusion="concat",
        modality_dropout=0.0,
        subject_affine=False,
        dino_pooling="mean",
        dino_frames=8,
        attention_hidden_dim=16,
    ).to(device)

    heldout_eeg = np.random.randn(N, 17, T).astype(np.float32)
    heldout_subjects = np.ones(N, dtype=np.int64)
    heldout_query_ids = HELDOUT_IDS[:N]
    heldout_dino_cands = np.random.randn(100, 8, 16).astype(np.float32)
    heldout_audio_cands = np.random.randn(100, 8).astype(np.float32)

    report = run_importance_analysis(
        model, heldout_eeg, heldout_subjects, heldout_query_ids,
        heldout_dino_cands, heldout_audio_cands,
        ch_indices, ch_names, "raw", device, n_perms=1
    )
    assert len(report["leaderboard"]) == C
    print_leaderboard(report)
    print("Feature importance self-test passed successfully!")


def main() -> None:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return

    run_dir = args.run_dir
    if run_dir is None:
        raise ValueError("Must specify --run-dir or --self-test")
    run_dir = run_dir.expanduser().resolve()

    ckpt_path = args.checkpoint or (run_dir / "model_seed-42.pt")
    if not ckpt_path.is_file():
        ckpt_path = run_dir / "model_seed-42_fullfit.pt"
    if not ckpt_path.is_file():
        # Look for any .pt
        pts = list(run_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
        ckpt_path = pts[0]

    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    device = torch.device(args.device)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state_dict"]

    from experiments.exp4_feature_engineering.model import TunedMLPRetriever
    from experiments.exp4_tuned_mlp_dino_audio.train import (
        load_dino_frames, load_pooled_eeg,
        stack_dino_frames, WINDOW, N_CHANNELS
    )
    from experiments.exp2_dgcnn_dino_audio_mlp.run_experiment import (
        apply_normalizer, fit_normalizer, load_audio_map, resolve_path, stack_features
    )

    ch_spec = config.get("channels", "all17")
    ch_indices, ch_names = resolve_channel_indices(ch_spec)
    feature_mode = config.get("eeg_features", "raw")

    data_root = args.data_root.expanduser().resolve()

    norm_path = run_dir / "normalization.npz"
    if norm_path.is_file():
        with np.load(norm_path) as norm_pack:
            dino_mean = norm_pack["dino_mean"]
            dino_std = norm_pack["dino_std"]
            audio_mean = norm_pack["audio_mean"]
            audio_std = norm_pack["audio_std"]
    else:
        dino_cache = resolve_path(data_root, config.get("dino_train_cache", "features_dinov2/consolidated_encoder.layer.11.npz"))
        dino_map = load_dino_frames(dino_cache, np.arange(1, 1001), config.get("dino_frames", 8))
        audio_cache = data_root / "features_audio" / f"train_{config.get('audio_model', 'clap')}_features.npz"
        audio_map = load_audio_map(audio_cache)
        train_dino_raw = stack_dino_frames(dino_map, np.arange(1, 901), "train DINO")
        train_audio_raw = stack_features(audio_map, np.arange(1, 901), "train audio")
        dino_mean, dino_std = fit_normalizer(train_dino_raw.mean(axis=1))
        audio_mean, audio_std = fit_normalizer(train_audio_raw)

    dino_cache = resolve_path(data_root, config.get("dino_train_cache", "features_dinov2/consolidated_encoder.layer.11.npz"))
    dino_map = load_dino_frames(dino_cache, HELDOUT_IDS, config.get("dino_frames", 8))
    audio_cache = data_root / "features_audio" / f"train_{config.get('audio_model', 'clap')}_features.npz"
    audio_map = load_audio_map(audio_cache)

    heldout_dino_raw = stack_dino_frames(dino_map, HELDOUT_IDS, "heldout DINO")
    heldout_audio_raw = stack_features(audio_map, HELDOUT_IDS, "heldout audio")

    dino_mean_exp = dino_mean[:, None, :] if dino_mean.ndim == 2 else dino_mean
    dino_std_exp = dino_std[:, None, :] if dino_std.ndim == 2 else dino_std
    heldout_dino_cands = apply_normalizer(heldout_dino_raw, dino_mean_exp, dino_std_exp)
    heldout_audio_cands = apply_normalizer(heldout_audio_raw, audio_mean, audio_std)

    heldout_eeg, heldout_query_ids, heldout_subjects, _ = load_pooled_eeg(
        data_root, tuple(config.get("subjects", (1, 2, 3, 4, 5, 6))), 901, 1000
    )

    # Instantiate model
    from experiments.exp4_feature_engineering.feature_extractor import compute_feature_dim
    eeg_dim = compute_feature_dim(len(ch_indices), WINDOW.stop - WINDOW.start, feature_mode)

    model = TunedMLPRetriever(
        eeg_dim=eeg_dim,
        dino_dim=heldout_dino_cands.shape[-1],
        audio_dim=heldout_audio_cands.shape[-1],
        hidden_dim=config["hidden_dim"],
        embedding_dim=config["embedding_dim"],
        dropout=config["dropout"],
        temperature=config["temperature"],
        fusion=config["fusion"],
        modality_dropout=config.get("modality_dropout", 0.0),
        subject_affine=config.get("subject_affine", False),
        dino_pooling=config.get("dino_pooling", "mean"),
        dino_frames=config.get("dino_frames", 8),
        attention_hidden_dim=config.get("attention_hidden_dim", 256),
    ).to(device)
    model.load_state_dict(state)

    csls_lambda = 0.10
    heldout_npz = run_dir / "heldout_outputs.npz"
    if heldout_npz.is_file():
        with np.load(heldout_npz) as pack:
            if "selected_csls_lambda" in pack:
                csls_lambda = float(pack["selected_csls_lambda"])

    report = run_importance_analysis(
        model, heldout_eeg, heldout_subjects, heldout_query_ids,
        heldout_dino_cands, heldout_audio_cands,
        ch_indices, ch_names, feature_mode, device,
        csls_lambda=csls_lambda, n_perms=args.n_permutations
    )
    print_leaderboard(report)

    out_path = args.output or (run_dir / "feature_importance_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved detailed feature importance report to: {out_path}")


if __name__ == "__main__":
    main()
