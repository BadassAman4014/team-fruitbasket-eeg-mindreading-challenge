"""Phase 2 — Test Prediction, Evaluation, and Submission Generation.

Evaluates trained Phase 2 contrastive model on sub-01..04 with ground truth,
and produces competition submission files for sub-05 and sub-06.
Supports optional ensembling with Phase 1D (Ridge regression).

Usage
-----
    python experiments/phase2_predict.py --device cuda --model_ckpt ./models/phase2/phase2_best_contrastive.pt
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_test_eeg, decimate_eeg, zscore_per_subject
from src.dataset import load_eval_test_data
from src.models.contrastive_eeg import MultimodalContrastiveModel
from src.metrics import compute_ranks, retrieval_metrics, print_metrics, apply_anti_hubness
from src.pipeline import EncodingModel


TEST_DIR = "./tst"
GT_DIR = "./tst_stim"
TEST_IDS = list(range(1001, 1103))  # 102 test videos
EVAL_SUBJECTS = [1, 2, 3, 4]
PRED_SUBJECTS = [5, 6]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Prediction & Submission")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_ckpt", type=str, default="./models/phase2/phase2_best_contrastive.pt")
    parser.add_argument("--feat_cache", type=str, default="./cache/viclip_ast")
    parser.add_argument("--output_dir", type=str, default="./submissions")
    parser.add_argument("--anti_hub", type=str, default="none", choices=["none", "csls", "mean"], help="Anti-hubness method")
    parser.add_argument("--csls_k", type=int, default=10, help="CSLS neighborhood size")
    parser.add_argument("--ensemble_phase1d_dir", type=str, default="", help="Optional path to Phase 1D models for ensembling")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for Phase 2 in ensemble (1 - alpha for Phase 1D)")
    return parser.parse_args()


def load_contrastive_model(ckpt_path: str, device: str) -> MultimodalContrastiveModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt

    # Deduce stim_dim and proj_dim from state dict
    stim_dim = state_dict["stim_projector.net.0.weight"].shape[1]
    proj_dim = state_dict["stim_projector.net.4.weight"].shape[0]
    use_adapter = any("subject_adapters" in k for k in state_dict.keys())

    model = MultimodalContrastiveModel(
        stim_dim=stim_dim,
        proj_dim=proj_dim,
        use_subject_adapter=use_adapter,
        num_subjects=6,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model



def main():
    args = parse_args()
    print("=" * 60)
    print("  Phase 2 — Test Prediction and Submission Generation")
    print(f"  Checkpoint: {args.model_ckpt}")
    print(f"  Anti-hubness: {args.anti_hub}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load test features
    test_feat_path = os.path.join(args.feat_cache, "viclip_ast_test.npy")
    if not os.path.exists(test_feat_path):
        raise FileNotFoundError(f"Test features not found at {test_feat_path}")
    test_feats = np.load(test_feat_path)  # (102, 1536)
    test_feats_tensor = torch.as_tensor(test_feats, dtype=torch.float32, device=args.device)

    # 2. Load Phase 2 model
    model = load_contrastive_model(args.model_ckpt, args.device)
    video_ids = np.array(TEST_IDS)  # (102,) [1001, ..., 1102]

    # 3. Evaluate on sub-01 through sub-04 (with ground truth)
    all_metrics = {}
    print("\n--- Evaluating on Test Subjects (with ground truth) ---")
    for sub in EVAL_SUBJECTS:
        try:
            sorted_eeg, sorted_gt = load_eval_test_data(sub, test_dir=TEST_DIR, gt_dir=GT_DIR)
        except Exception as e:
            print(f"  Could not load sub-{sub:02d}: {e}")
            continue

        eeg_tensor = torch.as_tensor(sorted_eeg, dtype=torch.float32, device=args.device)
        sub_tensor = torch.full((len(eeg_tensor),), sub, dtype=torch.long, device=args.device)
        sim_p2 = model.compute_similarity(eeg_tensor, test_feats_tensor, sub_ids=sub_tensor).cpu().numpy()  # (102, 102)

        # Optional Ensemble with Phase 1D
        if args.ensemble_phase1d_dir:
            p1_path = os.path.join(args.ensemble_phase1d_dir, f"phase1d_sub-{sub:02d}.joblib")
            if os.path.exists(p1_path):
                p1_model = EncodingModel.load(p1_path)
                sim_p1 = p1_model.similarity_matrix(sorted_eeg, test_feats)
                # Normalize both similarity matrices before blending
                sim_p2_norm = (sim_p2 - sim_p2.mean()) / (sim_p2.std() + 1e-8)
                sim_p1_norm = (sim_p1 - sim_p1.mean()) / (sim_p1.std() + 1e-8)
                S = args.alpha * sim_p2_norm + (1.0 - args.alpha) * sim_p1_norm
            else:
                S = sim_p2
        else:
            S = sim_p2

        if args.anti_hub != "none":
            S = apply_anti_hubness(S, method=args.anti_hub, k=args.csls_k)

        ranks = compute_ranks(S)
        metrics = retrieval_metrics(ranks)
        all_metrics[sub] = metrics
        print_metrics(metrics, label=f"sub-{sub:02d} [TEST]")

        # Save sub-01..04 npy files
        np.save(os.path.join(args.output_dir, f"sub-{sub:02d}_ranks.npy"), ranks)
        np.save(os.path.join(args.output_dir, f"sub-{sub:02d}_similarity.npy"), S)
        ranked_candidates = video_ids[np.argsort(-S, axis=1)]
        np.save(os.path.join(args.output_dir, f"sub-{sub:02d}_ranked_candidate_list.npy"), ranked_candidates)

    print(f"\nSaved sub-01..04 evaluation .npy files (ranks, similarity, candidate lists) to: {args.output_dir}")

    if all_metrics:
        avg_top1 = float(np.mean([m["top1"] for m in all_metrics.values()]))
        avg_top5 = float(np.mean([m["top5"] for m in all_metrics.values()]))
        avg_med = float(np.mean([m["median_rank"] for m in all_metrics.values()]))
        print("\n" + "─" * 60)
        print("  Phase 2 Test Summary:")
        print(f"  Avg top-1: {avg_top1:.4f}  |  Avg top-5: {avg_top5:.4f}  |  Avg median rank: {avg_med:.1f}")
        print("─" * 60)

    # 4. Generate Submissions for sub-05 and sub-06
    print("\n--- Generating Predictions for Target Subjects (sub-05, sub-06) ---")
    for sub in PRED_SUBJECTS:
        tst_path = os.path.join(TEST_DIR, f"sub-{sub:02d}_tst.npy")
        if not os.path.exists(tst_path):
            print(f"  Warning: {tst_path} not found, skipping sub-{sub:02d}")
            continue

        raw_test = load_test_eeg(tst_path)
        dec_test = decimate_eeg(raw_test)
        norm_test = zscore_per_subject(dec_test)  # (102, 17, 300)

        eeg_tensor = torch.as_tensor(norm_test, dtype=torch.float32, device=args.device)
        sub_tensor = torch.full((len(eeg_tensor),), sub, dtype=torch.long, device=args.device)
        sim_p2 = model.compute_similarity(eeg_tensor, test_feats_tensor, sub_ids=sub_tensor).cpu().numpy()

        if args.ensemble_phase1d_dir:
            p1_path = os.path.join(args.ensemble_phase1d_dir, f"phase1d_sub-{sub:02d}.joblib")
            if os.path.exists(p1_path):
                p1_model = EncodingModel.load(p1_path)
                sim_p1 = p1_model.similarity_matrix(norm_test, test_feats)
                sim_p2_norm = (sim_p2 - sim_p2.mean()) / (sim_p2.std() + 1e-8)
                sim_p1_norm = (sim_p1 - sim_p1.mean()) / (sim_p1.std() + 1e-8)
                S = args.alpha * sim_p2_norm + (1.0 - args.alpha) * sim_p1_norm
            else:
                S = sim_p2
        else:
            S = sim_p2

        if args.anti_hub != "none":
            S = apply_anti_hubness(S, method=args.anti_hub, k=args.csls_k)

        # For submission subjects: rows are kept in their original recorded trial order!
        # Order candidates per trial best-first
        ranked_indices = np.argsort(-S, axis=1)
        ranked_candidates = video_ids[ranked_indices]  # (102, 102)

        out_path = os.path.join(args.output_dir, f"sub-{sub:02d}_ranked_candidate_list.npy")
        sim_path = os.path.join(args.output_dir, f"sub-{sub:02d}_similarity.npy")
        np.save(out_path, ranked_candidates)
        np.save(sim_path, S)
        print(f"  Saved submission -> {out_path} (shape: {ranked_candidates.shape}, dtype: {ranked_candidates.dtype})")

        # Submission integrity check
        assert ranked_candidates.shape == (102, 102), f"Expected shape (102, 102), got {ranked_candidates.shape}"
        assert (ranked_candidates.min() == 1001) and (ranked_candidates.max() == 1102), "Stimulus IDs out of range"
        for i in range(102):
            assert len(np.unique(ranked_candidates[i])) == 102, f"Trial {i} has duplicate rankings"

    print("\nAll submission integrity checks passed successfully!")


if __name__ == "__main__":
    main()

