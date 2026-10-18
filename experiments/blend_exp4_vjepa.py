"""Simple, robust score-level blend of Exp4 (DINOv2), V-JEPA 2 (8 sub-clips), and optional Exp13 (ViEEG).

Supports:
- 2-Way Blend: Exp4 (DINOv2 + CLAP) + V-JEPA 2 (8 sub-clips + CLAP)
- 3-Way Tri-Blend: Exp4 (DINOv2) + V-JEPA 2 (Motion) + Exp13 (ViEEG STC + Attention)

Protocol:
1. Loads heldout scores (901-1000) for all models (582 queries across all 6 subjects).
2. Sweeps weights on a simplex grid and CSLS lambda on heldout to pick the best combination honestly.
3. Aligns and blends test score matrices (1001-1102) for all subjects.
4. Evaluates retrieval on public test subjects (sub-01..04) with ground truth.
5. Saves verified submission files for held-out subjects sub-05 and sub-06.

Usage:
    # 2-way blend
    python experiments/blend_exp4_vjepa.py \\
        --exp4-run experiments/exp4_tuned_mlp_dino_audio/runs/exp4_dino_clap_run \\
        --vjepa-dir experiments/v-jepa/submissions \\
        --output-dir submissions/blend_exp4_vjepa

    # 3-way Tri-Blend with ViEEG
    python experiments/blend_exp4_vjepa.py \\
        --exp4-run experiments/exp4_tuned_mlp_dino_audio/runs/exp4_dino_clap_run \\
        --vjepa-dir experiments/v-jepa/submissions \\
        --vieeg-run experiments/exp13_vieeg_dino_audio/runs/exp13_vieeg_run \\
        --output-dir submissions/tri_blend_exp4_vjepa_vieeg
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

TEST_IDS = np.arange(1001, 1103, dtype=np.int64)
HELDOUT_IDS = np.arange(901, 1001, dtype=np.int64)
EVAL_SUBJECTS = [1, 2, 3, 4]
PRED_SUBJECTS = [5, 6]
ALL_SUBJECTS = [1, 2, 3, 4, 5, 6]


def parse_args():
    p = argparse.ArgumentParser(description="Blend Exp4 (DINOv2) + V-JEPA2 + optional ViEEG")
    p.add_argument(
        "--exp4-run",
        type=str,
        required=True,
        help="Path to exp4 run directory containing heldout_outputs.npz and sub-XX_test_outputs.npz",
    )
    p.add_argument(
        "--vjepa-dir",
        type=str,
        default="experiments/v-jepa/submissions",
        help="Path to v-jepa submissions directory containing recipe_vjepa2_frames_* files",
    )
    p.add_argument(
        "--vieeg-run",
        type=str,
        default=None,
        help="Optional path to Exp13 ViEEG run directory containing heldout_outputs.npz and sub-XX_test_outputs.npz",
    )
    p.add_argument(
        "--gt-dir",
        type=str,
        default="./tst_stim",
        help="Path to ground truth labels for sub-01..04",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="submissions/blend_exp4_vjepa",
        help="Directory to save blended similarity matrices and submission files",
    )
    p.add_argument(
        "--csls-k",
        type=int,
        default=10,
        help="Neighborhood size for CSLS",
    )
    return p.parse_args()


def zscore_matrix(S: np.ndarray) -> np.ndarray:
    """Z-standardize matrix to bring cosine similarity scores to a common scale."""
    std = S.std()
    return (S - S.mean()) / (std if std > 1e-8 else 1.0)


def apply_csls(S: np.ndarray, k: int = 10, hub_penalty: float = 0.0) -> np.ndarray:
    """Cross-Domain Similarity Local Scaling (candidate-side debiasing)."""
    if hub_penalty <= 0.0:
        return S
    k_val = min(k, S.shape[0])
    topk_queries = np.sort(S, axis=0)[-k_val:, :]
    r_cand = topk_queries.mean(axis=0, keepdims=True)
    return S - hub_penalty * r_cand


def ranks_from_scores(S: np.ndarray, cand_ids: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
    """Compute 1-indexed rank of correct candidate for each query."""
    order = np.argsort(-S, axis=1)
    ranked = cand_ids[order]
    hit = (ranked == query_ids[:, None])
    return hit.argmax(axis=1) + 1


def compute_eval_metrics(ranks: np.ndarray) -> dict:
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "median": float(np.median(ranks)),
        "mean": float(ranks.mean()),
    }


def resolve_path(p: str, fallbacks: list) -> str:
    if p is None:
        return None
    if os.path.exists(p):
        return p
    for fb in fallbacks:
        if os.path.exists(fb):
            return fb
    return p


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    # Path resolution fallbacks
    gt_dir = resolve_path(args.gt_dir, [str(repo_root / "tst_stim"), "./tst_stim", "../../tst_stim"])
    vjepa_dir = resolve_path(
        args.vjepa_dir,
        [
            str(here / "v-jepa" / "submissions"),
            str(repo_root / "experiments" / "v-jepa" / "submissions"),
            "./submissions",
        ],
    )
    exp4_dir = resolve_path(
        args.exp4_run,
        [
            str(here / "exp4_tuned_mlp_dino_audio" / "runs" / os.path.basename(args.exp4_run)),
            str(repo_root / args.exp4_run),
        ],
    )
    vieeg_dir = None
    if args.vieeg_run:
        vieeg_dir = resolve_path(
            args.vieeg_run,
            [
                str(here / "exp13_vieeg_dino_audio" / "runs" / os.path.basename(args.vieeg_run)),
                str(repo_root / args.vieeg_run),
            ],
        )

    is_tri_blend = vieeg_dir is not None
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    mode_name = "TRI-BLEND (Exp4 + V-JEPA2 + ViEEG)" if is_tri_blend else "2-WAY BLEND (Exp4 + V-JEPA2)"
    print(f"  {mode_name}")
    print(f"  Exp4 run dir  : {exp4_dir}")
    print(f"  V-JEPA2 dir   : {vjepa_dir}")
    if is_tri_blend:
        print(f"  ViEEG run dir : {vieeg_dir}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Load Held-Out Scores (901-1000) for Honest Tuning
    # ─────────────────────────────────────────────────────────────────────────────
    exp4_heldout_p = os.path.join(exp4_dir, "heldout_outputs.npz")
    vjepa_heldout_p = os.path.join(vjepa_dir, "recipe_vjepa2_frames_heldout_scores.npz")

    if not os.path.exists(exp4_heldout_p):
        raise FileNotFoundError(f"Missing Exp4 heldout file: {exp4_heldout_p}")
    if not os.path.exists(vjepa_heldout_p):
        raise FileNotFoundError(f"Missing V-JEPA2 heldout file: {vjepa_heldout_p}")

    with np.load(exp4_heldout_p) as pack:
        exp4_ho_raw = pack["raw_scores"].astype(np.float32)  # (582, 100)
        exp4_qids = pack["query_ids"].astype(np.int64)
        exp4_cids = pack["candidate_ids"].astype(np.int64)

    with np.load(vjepa_heldout_p) as pack:
        vjepa_ho_raw = pack["scores"].astype(np.float32)     # (582, 100)
        vjepa_qids = pack["query_ids"].astype(np.int64)
        vjepa_cids = pack["cand_ids"].astype(np.int64)

    if not np.array_equal(exp4_cids, vjepa_cids):
        raise ValueError("Heldout candidate IDs do not match between Exp4 and V-JEPA2!")
    if not np.array_equal(exp4_qids, vjepa_qids):
        raise ValueError("Heldout query IDs do not align between Exp4 and V-JEPA2!")

    ho_cids = exp4_cids
    ho_qids = exp4_qids

    vieeg_ho_raw = None
    if is_tri_blend:
        vieeg_heldout_p = os.path.join(vieeg_dir, "heldout_outputs.npz")
        if not os.path.exists(vieeg_heldout_p):
            raise FileNotFoundError(
                f"Missing ViEEG heldout file: {vieeg_heldout_p}\n"
                f"Make sure to train Exp13 first with --final-full-fit --predict-test!"
            )
        with np.load(vieeg_heldout_p) as pack:
            vieeg_ho_raw = pack["raw_scores"].astype(np.float32)
            vieeg_qids = pack["query_ids"].astype(np.int64)
            vieeg_cids = pack["candidate_ids"].astype(np.int64)
        if not np.array_equal(vieeg_cids, ho_cids):
            raise ValueError("Heldout candidate IDs do not match for ViEEG!")
        if not np.array_equal(vieeg_qids, ho_qids):
            raise ValueError("Heldout query IDs do not align for ViEEG!")

    print(f"Loaded heldout matrices: {exp4_ho_raw.shape} trials over {len(ho_cids)} candidates.")

    # Standardize heldout score matrices
    exp4_ho_z = zscore_matrix(exp4_ho_raw)
    vjepa_ho_z = zscore_matrix(vjepa_ho_raw)
    vieeg_ho_z = zscore_matrix(vieeg_ho_raw) if is_tri_blend else None

    # Print baseline single-model heldout performances
    m_exp4_ho = compute_eval_metrics(ranks_from_scores(exp4_ho_raw, ho_cids, ho_qids))
    m_vjepa_ho = compute_eval_metrics(ranks_from_scores(vjepa_ho_raw, ho_cids, ho_qids))
    print(f"  Exp4   heldout top-1: {m_exp4_ho['top1']*100:5.2f}% | top-5: {m_exp4_ho['top5']*100:5.2f}%")
    print(f"  V-JEPA heldout top-1: {m_vjepa_ho['top1']*100:5.2f}% | top-5: {m_vjepa_ho['top5']*100:5.2f}%")
    if is_tri_blend:
        m_vieeg_ho = compute_eval_metrics(ranks_from_scores(vieeg_ho_raw, ho_cids, ho_qids))
        print(f"  ViEEG  heldout top-1: {m_vieeg_ho['top1']*100:5.2f}% | top-5: {m_vieeg_ho['top5']*100:5.2f}%")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Grid Sweep on Held-out Set (Alpha / Weights & CSLS Lambda)
    # ─────────────────────────────────────────────────────────────────────────────
    lam_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    best_t1 = -1.0
    best_lam = 0.0

    if not is_tri_blend:
        # 2-way blend
        alpha_grid = np.round(np.linspace(0.0, 1.0, 21), 2)
        best_alpha = 0.5
        print("\nSweeping alpha (Exp4 weight) and CSLS lambda on held-out 901-1000 split:")
        for a in alpha_grid:
            S_blend = a * exp4_ho_z + (1.0 - a) * vjepa_ho_z
            for lam in lam_grid:
                S_csls = apply_csls(S_blend, k=args.csls_k, hub_penalty=lam) if lam > 0 else S_blend
                r = ranks_from_scores(S_csls, ho_cids, ho_qids)
                t1 = float((r == 1).mean())
                if t1 > best_t1:
                    best_t1, best_alpha, best_lam = t1, a, lam

        best_weights = (best_alpha, round(1.0 - best_alpha, 2), 0.0)
        print("-" * 80)
        print(f"  BEST HELDOUT CONFIG: Exp4={best_weights[0]*100:.1f}%, V-JEPA2={best_weights[1]*100:.1f}%, "
              f"CSLS lambda={best_lam:.2f}")
        print(f"  Held-out Top-1: {best_t1*100:5.2f}%")
        print("-" * 80)
    else:
        # 3-way Tri-Blend simplex sweep
        print("\nSweeping 3-way simplex weights (Exp4, V-JEPA2, ViEEG) and CSLS lambda on held-out split:")
        best_weights = (0.33, 0.33, 0.34)
        for w1 in np.linspace(0.0, 1.0, 21):
            for w2 in np.linspace(0.0, 1.0 - w1, int(round((1.0 - w1) * 20)) + 1):
                w3 = round(1.0 - w1 - w2, 4)
                if w3 < -1e-5:
                    continue
                w3 = max(0.0, w3)
                S_blend = w1 * exp4_ho_z + w2 * vjepa_ho_z + w3 * vieeg_ho_z
                for lam in lam_grid:
                    S_csls = apply_csls(S_blend, k=args.csls_k, hub_penalty=lam) if lam > 0 else S_blend
                    r = ranks_from_scores(S_csls, ho_cids, ho_qids)
                    t1 = float((r == 1).mean())
                    if t1 > best_t1:
                        best_t1 = t1
                        best_weights = (round(w1, 2), round(w2, 2), round(w3, 2))
                        best_lam = lam

        print("-" * 80)
        print(f"  BEST HELDOUT CONFIG (3-WAY): Exp4={best_weights[0]*100:.1f}%, V-JEPA2={best_weights[1]*100:.1f}%, "
              f"ViEEG={best_weights[2]*100:.1f}%, CSLS lambda={best_lam:.2f}")
        print(f"  Held-out Top-1: {best_t1*100:5.2f}%")
        print("-" * 80)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Load & Blend Test Matrices (1001-1102)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\nBlending Test Matrices for Subjects 1..6:")
    blended_sims: Dict[int, np.ndarray] = {}

    for sub in ALL_SUBJECTS:
        exp4_test_f = os.path.join(exp4_dir, f"sub-{sub:02d}_test_outputs.npz")
        vjepa_test_f = os.path.join(vjepa_dir, f"recipe_vjepa2_frames_sub-{sub:02d}_similarity.npy")

        if not os.path.exists(exp4_test_f):
            raise FileNotFoundError(f"Missing Exp4 test file: {exp4_test_f}")
        if not os.path.exists(vjepa_test_f):
            raise FileNotFoundError(f"Missing V-JEPA2 test file: {vjepa_test_f}")

        with np.load(exp4_test_f) as pack:
            S_exp4 = pack["raw_scores"].astype(np.float32)

        S_vjepa = np.load(vjepa_test_f).astype(np.float32)

        S_vieeg = None
        if is_tri_blend:
            vieeg_test_f = os.path.join(vieeg_dir, f"sub-{sub:02d}_test_outputs.npz")
            if not os.path.exists(vieeg_test_f):
                raise FileNotFoundError(f"Missing ViEEG test file: {vieeg_test_f}")
            with np.load(vieeg_test_f) as pack:
                S_vieeg = pack["raw_scores"].astype(np.float32)

        # Row Alignment:
        # In v-jepa, sub-01..04 test matrices were saved in label-sorted order.
        # We sort Exp4 and ViEEG rows by ground-truth labels so all models are label-sorted on sub-01..04.
        if sub in EVAL_SUBJECTS:
            gt_path = os.path.join(gt_dir, f"sub-{sub:02d}_tst_stim.npy")
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Missing ground truth for sub-{sub:02d}: {gt_path}")
            gt_stim = np.load(gt_path)
            order = np.argsort(gt_stim)
            S_exp4_aligned = S_exp4[order]
            S_vjepa_aligned = S_vjepa
            S_vieeg_aligned = S_vieeg[order] if is_tri_blend else None
        else:
            # sub-05 and sub-06 are all in original test recording order
            S_exp4_aligned = S_exp4
            S_vjepa_aligned = S_vjepa
            S_vieeg_aligned = S_vieeg if is_tri_blend else None

        z_exp4 = zscore_matrix(S_exp4_aligned)
        z_vjepa = zscore_matrix(S_vjepa_aligned)

        if not is_tri_blend:
            S_blend = best_weights[0] * z_exp4 + best_weights[1] * z_vjepa
        else:
            z_vieeg = zscore_matrix(S_vieeg_aligned)
            S_blend = best_weights[0] * z_exp4 + best_weights[1] * z_vjepa + best_weights[2] * z_vieeg

        if best_lam > 0:
            S_blend = apply_csls(S_blend, k=args.csls_k, hub_penalty=best_lam)

        blended_sims[sub] = S_blend
        np.save(os.path.join(args.output_dir, f"blend_sub-{sub:02d}_similarity.npy"), S_blend)

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Evaluate Public Subjects (sub-01..04) Against Ground Truth
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    tag_str = (
        f"Exp4: {best_weights[0]*100:.0f}%, V-JEPA2: {best_weights[1]*100:.0f}%, ViEEG: {best_weights[2]*100:.0f}%"
        if is_tri_blend else f"Exp4: {best_weights[0]*100:.0f}%, V-JEPA2: {best_weights[1]*100:.0f}%"
    )
    print(f"  BLENDED TEST RESULTS on sub-01..04 ({tag_str}, CSLS λ={best_lam:.2f})")
    print("=" * 80)
    print(f"  {'Subject':<9} | {'Top-1':<18} | {'Top-5':<18} | Median Rank")
    print("-" * 80)

    top1_list, top5_list, med_list = [], [], []
    for sub in EVAL_SUBJECTS:
        S = blended_sims[sub]
        ranks = (-S).argsort(axis=1).argsort(axis=1)[np.arange(len(S)), np.arange(len(S))] + 1
        m = compute_eval_metrics(ranks)
        top1_list.append(m["top1"])
        top5_list.append(m["top5"])
        med_list.append(m["median"])
        print(f"  sub-{sub:02d}    | {m['top1']*100:5.2f}% ({m['top1']*102:2.0f}/102)   | "
              f"{m['top5']*100:5.2f}% ({m['top5']*102:2.0f}/102)   | {m['median']:.1f}")

    avg_t1 = float(np.mean(top1_list))
    avg_t5 = float(np.mean(top5_list))
    avg_med = float(np.mean(med_list))
    print("-" * 80)
    print(f"  {'AVERAGE':<9} | {avg_t1*100:5.2f}% ({avg_t1*408:3.0f}/408)  | {avg_t5*100:5.2f}% ({avg_t5*408:3.0f}/408)  | {avg_med:.1f}")
    print(f"  {'CHANCE':<9} |  0.98% (  4/408)  |  4.90% ( 20/408)  | 51.5")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. Generate & Verify Submissions for Held-Out Subjects (sub-05, sub-06)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\nGenerated Competition Submissions (Original Test Row Order):")
    for sub in PRED_SUBJECTS:
        S = blended_sims[sub]
        ranked = TEST_IDS[np.argsort(-S, axis=1)]

        assert ranked.shape == (102, 102), f"Bad shape: {ranked.shape}"
        assert ranked.min() == 1001 and ranked.max() == 1102, "Invalid candidate IDs"
        for i in range(102):
            assert len(np.unique(ranked[i])) == 102, f"Row {i} contains duplicates"

        out_path = os.path.join(args.output_dir, f"sub-{sub:02d}_ranked_candidate_list.npy")
        np.save(out_path, ranked)
        print(f"  [OK] sub-{sub:02d}: {out_path} (shape: {ranked.shape})")

    print("\n[Done] Blending complete!")


if __name__ == "__main__":
    main()
