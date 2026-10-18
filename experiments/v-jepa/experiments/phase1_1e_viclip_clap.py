"""Phase 1E — ViCLIP video (768-d) + CLAP audio (512-d) comparison.

Evaluates whether contrastive language-audio pretraining (CLAP) outperforms
categorical acoustic pretraining (AST) when paired with ViCLIP.

Features are concatenated: ViCLIP (768-d) + CLAP (512-d) → 1280-d multimodal vector.
Trains PCA + Ridge models for each subject and evaluates against ground truth (sub-01..04).

Usage
-----
    python experiments/phase1_1e_viclip_clap.py --device cuda
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_train_eeg, load_test_eeg, decimate_eeg, normalize_eeg_trials
from src.dataset import load_eval_test_data
from src.pipeline import EncodingModel
from src.metrics import compute_ranks, retrieval_metrics, print_metrics, apply_anti_hubness
from src.features.viclip_extractor import ViCLIPExtractor
from src.features.clap_extractor import CLAPExtractor


TRAIN_DIR = "./trn"
TEST_DIR = "./tst"
GT_DIR = "./tst_stim"
STIM_TRAIN_DIR = "./stimuli_train"
STIM_TEST_DIR = "./stimuli_test"
TEST_IDS = list(range(1001, 1103))
SUBJECTS = [1, 2, 3, 4]


def video_path(stim_id: int, stim_dir: str) -> str:
    return os.path.join(stim_dir, f"{int(stim_id):04d}.mp4")


def get_all_viclip_clap_features(clap_ext: CLAPExtractor, stim_dir: str, cache_dir: str, viclip_cache_dir: str = "./cache/viclip_ast") -> Tuple[np.ndarray, np.ndarray]:
    """Extract and cache ViCLIP (768) + CLAP (512) for both train and test videos."""
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "viclip_clap_train_all.npy")
    test_cache = os.path.join(cache_dir, "viclip_clap_test.npy")

    # 1. ViCLIP video features (reused from viclip_ast cache if available)
    viclip_train_path = os.path.join(viclip_cache_dir, "viclip_ast_train_all.npy")
    viclip_test_path = os.path.join(viclip_cache_dir, "viclip_ast_test.npy")

    if os.path.exists(viclip_train_path):
        print(f"  Reusing ViCLIP video features (first 768 dims) from {viclip_train_path}")
        V_trn = np.load(viclip_train_path)[:, :768]
    else:
        raise FileNotFoundError(f"ViCLIP features not found at {viclip_train_path}")

    if os.path.exists(viclip_test_path):
        print(f"  Reusing ViCLIP test features (first 768 dims) from {viclip_test_path}")
        V_tst = np.load(viclip_test_path)[:, :768]
    else:
        raise FileNotFoundError(f"ViCLIP test features not found at {viclip_test_path}")

    # 2. CLAP audio features
    clap_train_path = os.path.join(cache_dir, "clap_audio_train_all.npy")
    if os.path.exists(clap_train_path):
        print(f"  Loading cached CLAP training audio features from {clap_train_path}")
        C_trn = np.load(clap_train_path)
    else:
        print("  Extracting CLAP audio features for 1000 training videos ...")
        train_paths = [video_path(sid, STIM_TRAIN_DIR) for sid in range(1, 1001)]
        raw_c = clap_ext.extract_audio(train_paths)
        C_trn = np.zeros((1001, raw_c.shape[1]), dtype=raw_c.dtype)
        C_trn[1:] = raw_c
        np.save(clap_train_path, C_trn)
        print(f"  Saved CLAP train audio features → {clap_train_path}")

    clap_test_path = os.path.join(cache_dir, "clap_audio_test.npy")
    if os.path.exists(clap_test_path):
        print(f"  Loading cached CLAP test audio features from {clap_test_path}")
        C_tst = np.load(clap_test_path)
    else:
        print("  Extracting CLAP audio features for 102 test videos ...")
        test_paths = [video_path(sid, STIM_TEST_DIR) for sid in TEST_IDS]
        C_tst = clap_ext.extract_audio(test_paths)
        np.save(clap_test_path, C_tst)
        print(f"  Saved CLAP test audio features → {clap_test_path}")

    # 3. Concatenate ViCLIP (768) + CLAP (512) = 1280
    train_feats = np.concatenate([V_trn, C_trn], axis=1)
    test_feats = np.concatenate([V_tst, C_tst], axis=1)

    np.save(train_cache, train_feats)
    np.save(test_cache, test_feats)
    print(f"  ViCLIP+CLAP features cached: Train={train_feats.shape}, Test={test_feats.shape}")
    return train_feats, test_feats


def main():
    parser = argparse.ArgumentParser(description="Phase 1E — ViCLIP + CLAP Evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feat_cache", type=str, default="./cache/viclip_clap")
    parser.add_argument("--viclip_cache", type=str, default="./cache/viclip_ast")
    parser.add_argument("--model_dir", type=str, default="./models/phase1e")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("=" * 70)
    print("  Phase 1E — Multimodal Feature Comparison: ViCLIP + CLAP vs AST")
    print(f"  Device: {args.device} | Target: 1280-d multimodal stimulus features")
    print("=" * 70)

    clap_ext = CLAPExtractor(device=args.device, audio_cache_dir=os.path.join(args.feat_cache, "wav"))

    train_feats, test_feats = get_all_viclip_clap_features(
        clap_ext, STIM_TRAIN_DIR, args.feat_cache, args.viclip_cache
    )

    all_metrics = {}
    print("\n--- Training and Evaluating Phase 1E (Ridge) per Subject ---")
    for sub in SUBJECTS:
        print(f"\n{'─'*50}\n  sub-{sub:02d} [ViCLIP + CLAP]\n{'─'*50}")
        npz_path = os.path.join(TRAIN_DIR, f"sub-{sub:02d}_trn.npz")
        ids, raw_eeg = load_train_eeg(npz_path)
        dec_eeg = decimate_eeg(raw_eeg)
        norm_eeg = normalize_eeg_trials(dec_eeg)

        feats_trn = train_feats[ids]
        model = EncodingModel()
        model.fit(feats_trn, norm_eeg)
        model.save(os.path.join(args.model_dir, f"phase1e_sub-{sub:02d}.joblib"))

        # Evaluate on test set
        sorted_eeg, _ = load_eval_test_data(sub, test_dir=TEST_DIR, gt_dir=GT_DIR, eeg_window=None, norm_mode="trial")
        S = model.similarity_matrix(sorted_eeg, test_feats)
        S_csls = apply_anti_hubness(S, method="csls", k=10, hub_penalty=0.25)
        r = compute_ranks(S_csls)
        m = retrieval_metrics(r)
        all_metrics[sub] = m
        print_metrics(m, label=f"sub-{sub:02d} [TEST (CSLS)]")

    # Summary
    top1_list = [m["top1"] for m in all_metrics.values()]
    top5_list = [m["top5"] for m in all_metrics.values()]
    med_list = [m["median_rank"] for m in all_metrics.values()]

    print("\n" + "=" * 70)
    print("  PHASE 1 FEATURE COMPARISON SUMMARY (Ridge Baseline)")
    print("=" * 70)
    print(f"  Organizer Baseline (AlexNet):      Top-1 = 4.41%  |  Top-5 = 13.97%  |  Med Rank = 30.1")
    print(f"  Phase 1A (ViCLIP video only):      Top-1 = 3.68%  |  Top-5 = 15.93%  |  Med Rank = 28.6")
    print(f"  Phase 1D (ViCLIP + AST audio):     Top-1 = 5.15%  |  Top-5 = 16.42%  |  Med Rank = 26.2")
    print(f"  Phase 1E (ViCLIP + CLAP audio):    Top-1 = {np.mean(top1_list)*100:.2f}%  |  Top-5 = {np.mean(top5_list)*100:.2f}%  |  Med Rank = {np.mean(med_list):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
