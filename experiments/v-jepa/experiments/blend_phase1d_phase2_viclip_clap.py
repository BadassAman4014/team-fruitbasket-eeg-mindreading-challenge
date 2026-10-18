"""Phase 1D (Ridge) + Phase 2 (Contrastive ViCLIP + CLAP) Blend.

Combines linear Ridge regression (Phase 1D) with the non-linear deep
contrastive network trained on ViCLIP video + CLAP audio (Phase 2).

This cross-modal ensemble achieves tri-fusion:
- Linear decoding in EEG space + Non-linear contrastive projection
- AST acoustic classification + CLAP contrastive audio semantics
- ViCLIP spatio-temporal video alignment

Usage
-----
    python experiments/blend_phase1d_phase2_viclip_clap.py --device cuda
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_train_eeg, load_test_eeg, decimate_eeg, normalize_eeg_trials
from src.dataset import load_eval_test_data
from src.pipeline import EncodingModel
from src.models.contrastive_eeg import MultimodalContrastiveModel
from src.metrics import compute_ranks, retrieval_metrics, print_metrics, apply_anti_hubness


TEST_DIR = "./tst"
GT_DIR = "./tst_stim"
TRN_DIR = "./trn"
TEST_IDS = list(range(1001, 1103))  # 102 test videos
EVAL_SUBJECTS = [1, 2, 3, 4]
PRED_SUBJECTS = [5, 6]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1D + Phase 2 (ViCLIP+CLAP) Blend")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feat_cache_p1d", type=str, default="./cache/viclip_ast", help="Phase 1D features (1536-d)")
    parser.add_argument("--feat_cache_p2", type=str, default="./cache/viclip_clap", help="Phase 2 CLAP features (1280-d)")
    parser.add_argument("--p1d_dir", type=str, default="./models/phase1d")
    parser.add_argument("--p2_dir", type=str, default="./models/phase2_clap")
    parser.add_argument("--seeds", type=str, default="42,123,999", help="Phase 2 seeds to ensemble")
    parser.add_argument("--sub_dir", type=str, default="./submissions")
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument("--hub_penalty", type=float, default=0.25)
    return parser.parse_args()


def get_phase1d_similarities(args, test_feats_p1d: np.ndarray, all_subjects_eeg: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Load or fit Phase 1D Ridge models and compute similarity matrices."""
    os.makedirs(args.p1d_dir, exist_ok=True)
    p1d_sims = {}

    train_feat_path = os.path.join(args.feat_cache_p1d, "viclip_ast_train_all.npy")
    train_feats = np.load(train_feat_path)

    for sub in EVAL_SUBJECTS + PRED_SUBJECTS:
        model_path = os.path.join(args.p1d_dir, f"phase1d_sub-{sub:02d}.joblib")
        if not os.path.exists(model_path):
            npz_path = os.path.join(TRN_DIR, f"sub-{sub:02d}_trn.npz")
            if not os.path.exists(npz_path):
                model_path = os.path.join(args.p1d_dir, "phase1d_sub-01.joblib")
                if os.path.exists(model_path):
                    model = EncodingModel.load(model_path)
                else:
                    continue
            else:
                ids, raw_eeg = load_train_eeg(npz_path)
                eeg_dec = decimate_eeg(raw_eeg)
                eeg_norm = normalize_eeg_trials(eeg_dec)
                sub_feats = train_feats[ids]

                print(f"  Fitting Phase 1D Ridge for sub-{sub:02d} ...")
                model = EncodingModel()
                model.fit(sub_feats, eeg_norm)
                model.save(model_path)
        else:
            model = EncodingModel.load(model_path)

        if sub in all_subjects_eeg:
            eeg_sub = all_subjects_eeg[sub]
            S_p1 = model.similarity_matrix(eeg_sub, test_feats_p1d)
            p1d_sims[sub] = S_p1

    return p1d_sims


@torch.no_grad()
def get_phase2_clap_similarities(args, test_feats_p2: np.ndarray, all_subjects_eeg_erp: Dict[int, torch.Tensor]) -> Dict[int, np.ndarray]:
    """Compute similarity matrices from trained Phase 2 ViCLIP+CLAP checkpoints."""
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    test_feats_tensor = torch.as_tensor(test_feats_p2, dtype=torch.float32, device=args.device)

    per_seed_sims = {}
    stim_dim = test_feats_p2.shape[1]  # 1280

    for s in seed_list:
        ckpt_path = os.path.join(args.p2_dir, f"clap_contrastive_seed_{s}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Note: Checkpoint {ckpt_path} not found, skipping Seed {s}")
            continue

        ckpt = torch.load(ckpt_path, map_location=args.device)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt

        model = MultimodalContrastiveModel(
            stim_dim=stim_dim,
            proj_dim=256,
            init_temp=0.07,
            dropout=0.25,
            use_subject_adapter=True,
            num_subjects=6,
        ).to(args.device)
        model.load_state_dict(state)
        model.eval()

        sims_s = {}
        for sub, eeg_t in all_subjects_eeg_erp.items():
            eeg_dev = eeg_t.to(args.device)
            sub_dev = torch.full((len(eeg_dev),), sub, dtype=torch.long, device=args.device)
            sim = model.compute_similarity(eeg_dev, test_feats_tensor, sub_ids=sub_dev)
            sims_s[sub] = sim.cpu().numpy()

        per_seed_sims[s] = sims_s

    if not per_seed_sims:
        raise FileNotFoundError(
            f"No ViCLIP+CLAP checkpoints found in {args.p2_dir}. "
            "Please run 'python experiments/phase2_viclip_clap.py --device cuda' first!"
        )

    p2_ens_sims = {}
    all_target_subjects = list(all_subjects_eeg_erp.keys())

    for sub in all_target_subjects:
        norm_matrices = []
        for s, sims_s in per_seed_sims.items():
            if sub in sims_s:
                S = sims_s[sub]
                S_norm = (S - S.mean()) / (S.std() + 1e-8)
                norm_matrices.append(S_norm)

        if norm_matrices:
            p2_ens_sims[sub] = np.mean(norm_matrices, axis=0)

    print(f"  Aggregated Phase 2 CLAP predictions across {len(per_seed_sims)} seed(s): {list(per_seed_sims.keys())}")
    return p2_ens_sims


def main():
    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)

    print("=" * 70)
    print("  Phase 1D (Ridge: ViCLIP+AST) + Phase 2 (Contrastive: ViCLIP+CLAP) Blend")
    print(f"  Device: {args.device} | CSLS: k={args.csls_k}, lambda={args.hub_penalty}")
    print("=" * 70)

    # 1. Load test features for both models
    test_p1d_path = os.path.join(args.feat_cache_p1d, "viclip_ast_test.npy")
    test_p2_path = os.path.join(args.feat_cache_p2, "viclip_clap_test.npy")

    if not os.path.exists(test_p1d_path):
        raise FileNotFoundError(f"Phase 1D test features not found at {test_p1d_path}")
    if not os.path.exists(test_p2_path):
        raise FileNotFoundError(f"Phase 2 CLAP test features not found at {test_p2_path}")

    test_feats_p1d = np.load(test_p1d_path)  # (102, 1536)
    test_feats_p2 = np.load(test_p2_path)    # (102, 1280)

    # 2. Preload EEG data
    all_eeg_full = {}
    for sub in EVAL_SUBJECTS:
        sorted_eeg, _ = load_eval_test_data(sub, test_dir=TEST_DIR, gt_dir=GT_DIR, eeg_window=None, norm_mode="trial")
        all_eeg_full[sub] = sorted_eeg

    for sub in PRED_SUBJECTS:
        tst_path = os.path.join(TEST_DIR, f"sub-{sub:02d}_tst.npy")
        if os.path.exists(tst_path):
            raw = load_test_eeg(tst_path)
            dec = decimate_eeg(raw)
            norm = normalize_eeg_trials(dec)
            all_eeg_full[sub] = norm

    all_eeg_erp = {}
    for sub in EVAL_SUBJECTS:
        sorted_eeg, _ = load_eval_test_data(sub, test_dir=TEST_DIR, gt_dir=GT_DIR, eeg_window=(5, 60), norm_mode="trial")
        all_eeg_erp[sub] = torch.as_tensor(sorted_eeg, dtype=torch.float32)

    for sub in PRED_SUBJECTS:
        tst_path = os.path.join(TEST_DIR, f"sub-{sub:02d}_tst.npy")
        if os.path.exists(tst_path):
            raw = load_test_eeg(tst_path)
            dec = decimate_eeg(raw)[..., 5:60]
            norm = normalize_eeg_trials(dec)
            all_eeg_erp[sub] = torch.as_tensor(norm, dtype=torch.float32)

    # 3. Extract similarities from both models
    print("\n[Step 1/3] Extracting Phase 1D (Ridge: ViCLIP+AST) similarity matrices ...")
    sims_p1d = get_phase1d_similarities(args, test_feats_p1d, all_eeg_full)

    print("\n[Step 2/3] Extracting Phase 2 (Contrastive: ViCLIP+CLAP) similarity matrices ...")
    sims_p2_clap = get_phase2_clap_similarities(args, test_feats_p2, all_eeg_erp)

    # 4. Sweep alpha
    print("\n[Step 3/3] Sweeping Blend Alpha: S_blend = alpha * S_P2_CLAP + (1 - alpha) * S_P1D")
    print("-" * 75)
    print(f"  {'Alpha (P2 CLAP)':<18} | {'Top-1 Accuracy':<16} | {'Top-5 Accuracy':<16} | {'Median Rank':<12}")
    print("-" * 75)

    alpha_grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    best_alpha = 1.0
    best_top1 = -1.0
    best_top5 = -1.0
    best_med = 999.0

    for alpha in alpha_grid:
        top1_list = []
        top5_list = []
        med_list = []

        for sub in EVAL_SUBJECTS:
            S2 = sims_p2_clap[sub]
            S1 = sims_p1d[sub]

            S2_norm = (S2 - S2.mean()) / (S2.std() + 1e-8)
            S1_norm = (S1 - S1.mean()) / (S1.std() + 1e-8)

            S_blend = alpha * S2_norm + (1.0 - alpha) * S1_norm
            S_csls = apply_anti_hubness(S_blend, method="csls", k=args.csls_k, hub_penalty=args.hub_penalty)

            r = compute_ranks(S_csls)
            m = retrieval_metrics(r)
            top1_list.append(m["top1"])
            top5_list.append(m["top5"])
            med_list.append(m["median_rank"])

        avg_top1 = float(np.mean(top1_list))
        avg_top5 = float(np.mean(top5_list))
        avg_med = float(np.mean(med_list))

        is_best = (avg_top1 > best_top1) or (avg_top1 == best_top1 and avg_top5 > best_top5)
        marker = "  [★ BEST BLEND]" if is_best else ""
        if is_best:
            best_top1 = avg_top1
            best_top5 = avg_top5
            best_med = avg_med
            best_alpha = alpha

        label = f"alpha={alpha:.2f}"
        if alpha == 0.0:
            label += " (P1D only)"
        elif alpha == 1.0:
            label += " (P2 CLAP only)"

        print(f"  {label:<18} | {avg_top1*100:5.2f}% ({avg_top1*408:2.0f}/408)     | "
              f"{avg_top5*100:5.2f}% ({avg_top5*408:2.0f}/408)     | {avg_med:5.1f}{marker}")

    print("-" * 75)
    print(f"\nWinning Blend: Alpha = {best_alpha:.2f} (P2 CLAP: {best_alpha*100:.0f}%, P1D Ridge: {(1-best_alpha)*100:.0f}%)")
    print(f"  Blended Top-1: {best_top1*100:.2f}% ({best_top1*408:.0f}/408 hits)")
    print(f"  Blended Top-5: {best_top5*100:.2f}% ({best_top5*408:.0f}/408 hits)")
    print(f"  Blended Median Rank: {best_med:.1f}")

    # 5. Subject Breakdown of the Winning Blend
    print("\n" + "=" * 70)
    print(f"  Optimal Blended Performance Breakdown (Alpha={best_alpha:.2f})")
    print("=" * 70)
    print(f"  {'Subject':<10} | {'Top-1 Accuracy':<16} | {'Top-5 Accuracy':<16} | {'Median Rank':<12}")
    print("-" * 70)

    for sub in EVAL_SUBJECTS:
        S2_norm = (sims_p2_clap[sub] - sims_p2_clap[sub].mean()) / (sims_p2_clap[sub].std() + 1e-8)
        S1_norm = (sims_p1d[sub] - sims_p1d[sub].mean()) / (sims_p1d[sub].std() + 1e-8)
        S_blend = best_alpha * S2_norm + (1.0 - best_alpha) * S1_norm
        S_csls = apply_anti_hubness(S_blend, method="csls", k=args.csls_k, hub_penalty=args.hub_penalty)

        r = compute_ranks(S_csls)
        m = retrieval_metrics(r)
        print(f"  sub-{sub:02d}     | {m['top1']*100:5.2f}% ({m['top1']*102:2.0f}/102)     | "
              f"{m['top5']*100:5.2f}% ({m['top5']*102:2.0f}/102)     | {m['median_rank']:5.1f}")

    print("-" * 70)

    # 6. Generate and save blended competition submissions for sub-05 and sub-06
    video_ids = np.array(TEST_IDS)
    print("\n" + "=" * 70)
    print("  Generating Blended Competition Submissions for sub-05 and sub-06")
    print("=" * 70)

    for sub in PRED_SUBJECTS:
        if sub in sims_p2_clap and sub in sims_p1d:
            S2_norm = (sims_p2_clap[sub] - sims_p2_clap[sub].mean()) / (sims_p2_clap[sub].std() + 1e-8)
            S1_norm = (sims_p1d[sub] - sims_p1d[sub].mean()) / (sims_p1d[sub].std() + 1e-8)
            S_blend = best_alpha * S2_norm + (1.0 - best_alpha) * S1_norm
            S_csls = apply_anti_hubness(S_blend, method="csls", k=args.csls_k, hub_penalty=args.hub_penalty)

            ranked_indices = np.argsort(-S_csls, axis=1)
            ranked_candidates = video_ids[ranked_indices]

            out_path = os.path.join(args.sub_dir, f"blend_clap_sub-{sub:02d}_ranked_candidate_list.npy")
            np.save(out_path, ranked_candidates)
            print(f"  ✓ sub-{sub:02d}: Saved Blended CLAP Submission -> {out_path} (shape: {ranked_candidates.shape})")

            # Integrity checks
            assert ranked_candidates.shape == (102, 102), f"Expected (102, 102), got {ranked_candidates.shape}"
            assert (ranked_candidates.min() == 1001) and (ranked_candidates.max() == 1102), "ID bounds violated"
            for i in range(102):
                assert len(np.unique(ranked_candidates[i])) == 102, f"Trial {i} has duplicate rankings"

    print("\nPhase 1D + Phase 2 (ViCLIP+CLAP) blending complete!")


if __name__ == "__main__":
    main()
