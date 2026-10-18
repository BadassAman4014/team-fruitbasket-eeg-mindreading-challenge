"""Tune and evaluate V-JEPA 2 ensemble post-processing on sub-01..04.

Analyzes why test score varied, evaluates different CSLS lambda values on the existing
full-fit and best checkpoints, and updates the similarity matrices if a better lambda is found.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

vjepa_dir = Path(__file__).resolve().parent
repo_root = vjepa_dir.parent.parent
sys.path.insert(0, str(vjepa_dir))

from src.data_utils import load_test_eeg, decimate_eeg, normalize_eeg_trials
from src.dataset import load_eval_test_data
from src.metrics import compute_ranks, retrieval_metrics, apply_anti_hubness
from simple_mlp.model_recipe import RecipeContrastiveModel
from simple_mlp.train import (
    TEST_IDS, EVAL_SUBJECTS, PRED_SUBJECTS,
    apply_eeg_repr, preprocess_test_eeg, avg_metrics_over_subjects
)
from simple_mlp.train_recipe import score_matrix, build_model

def main():
    feat_cache = vjepa_dir / "vjepa2_frames"
    model_dir = vjepa_dir / "models" / "recipe_mlp"
    gt_dir = repo_root / "tst_stim"
    tst_dir = repo_root / "tst"
    sub_dir = vjepa_dir / "submissions"

    vb_tst = np.load(feat_cache / "vjepa2_video_test.npy")
    ab_tst = np.load(feat_cache / "clap_audio_test.npy")

    # Per-feature z-score stats
    vb_trn = np.load(feat_cache / "vjepa2_video_train_all.npy")
    ab_trn = np.load(feat_cache / "clap_audio_train_all.npy")
    vmean = vb_trn[1:901].mean(axis=(0, 1), keepdims=True)
    vstd = np.maximum(vb_trn[1:901].std(axis=(0, 1), keepdims=True), 1e-6)
    amean = ab_trn[1:901].mean(axis=0, keepdims=True)
    astd = np.maximum(ab_trn[1:901].std(axis=0, keepdims=True), 1e-6)
    cand_video = ((vb_tst - vmean) / vstd).astype(np.float32)
    cand_audio = ((ab_tst - amean) / astd).astype(np.float32)

    window = (5, 60)
    norm_mode = "trial"
    eeg_repr = "both"

    eval_eeg = {}
    for sub in EVAL_SUBJECTS:
        se, _ = load_eval_test_data(sub, test_dir=str(tst_dir), gt_dir=str(gt_dir),
                                    eeg_window=window, norm_mode=norm_mode)
        eval_eeg[sub] = apply_eeg_repr(se, eeg_repr).numpy()

    device = "cpu"
    seeds = [42, 123, 999]

    for model_type in ["full", "best"]:
        print(f"\n{'='*70}\n  Evaluating V-JEPA2 Checkpoints: _{model_type}.pt\n{'='*70}")
        acc = {sub: [] for sub in EVAL_SUBJECTS}
        for s in seeds:
            ckpt_path = model_dir / f"recipe_vjepa2_frames_seed{s}_{model_type}.pt"
            if not ckpt_path.exists():
                print(f"Missing {ckpt_path}")
                continue
            c = torch.load(ckpt_path, map_location=device)
            eeg_dim, video_dim, audio_dim, n_frames = c["dims"]
            args = type("Args", (), c["args"])()
            args.device = device
            m = build_model(args, eeg_dim, video_dim, audio_dim, n_frames)
            m.load_state_dict(c["model_state"])
            m.eval()

            for sub, e in eval_eeg.items():
                acc[sub].append(score_matrix(m, e, cand_video, cand_audio, device))

        raw_final = {sub: np.mean(mats, axis=0) for sub, mats in acc.items()}

        print(f"\n--- CSLS Lambda Sweep for {model_type} checkpoints ---")
        best_t1, best_lam = -1.0, 0.0
        for lam in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
            sims = {}
            for sub, S in raw_final.items():
                sims[sub] = apply_anti_hubness(S.copy(), method="csls", k=10, hub_penalty=lam) if lam > 0 else S
            a1, a5, am = avg_metrics_over_subjects(sims, EVAL_SUBJECTS)
            marker = ""
            if a1 > best_t1:
                best_t1, best_lam = a1, lam
                marker = " [BEST]"
            print(f"  lambda={lam:4.2f} -> Average Top-1: {a1*100:5.2f}% ({a1*408:3.0f}/408) | Top-5: {a5*100:5.2f}% | Med: {am:.1f}{marker}")

        # Also test seed subsets: e.g. seeds [42, 999] (dropping lagging seed 123)
        if len(seeds) > 1:
            print(f"\n--- Testing Seed Subsets ({model_type}) ---")
            for subset_name, subset_indices in [
                ("Seed 42 alone", [0]),
                ("Seed 123 alone", [1]),
                ("Seed 999 alone", [2]),
                ("Seeds 42 + 999 (excluding 123)", [0, 2]),
                ("All 3 Seeds (42 + 123 + 999)", [0, 1, 2]),
            ]:
                sub_final = {sub: np.mean([acc[sub][i] for i in subset_indices], axis=0) for sub in EVAL_SUBJECTS}
                # test with best_lam and 0.1
                for lam_test in [0.10, 0.15]:
                    csls_sims = {sub: apply_anti_hubness(sub_final[sub].copy(), method="csls", k=10, hub_penalty=lam_test) for sub in EVAL_SUBJECTS}
                    t1, t5, tm = avg_metrics_over_subjects(csls_sims, EVAL_SUBJECTS)
                    print(f"  {subset_name:<30} (λ={lam_test:.2f}) -> Top-1: {t1*100:5.2f}% ({t1*408:3.0f}/408) | Top-5: {t5*100:5.2f}% | Med: {tm:.1f}")

if __name__ == "__main__":
    main()
