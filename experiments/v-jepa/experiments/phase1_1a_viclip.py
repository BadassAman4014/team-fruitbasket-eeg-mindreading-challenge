"""Phase 1A — ViCLIP video-only features + PCA/Ridge pipeline.

Runs for all training subjects (sub-01 to sub-04), prints metrics,
and saves per-subject models to ./models/phase1a/.

Usage (cluster)
---------------
    python experiments/phase1_1a_viclip.py --device cuda

    (Weights are downloaded automatically from Hugging Face on the first run.
     Optionally pass --viclip_ckpt if using a custom local checkpoint.)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Allow `python experiments/phase1_1a_viclip.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils     import load_train_eeg, decimate_eeg, zscore_per_subject
from src.pipeline       import EncodingModel
from src.metrics        import compute_ranks, retrieval_metrics, print_metrics
from src.features.viclip_extractor import ViCLIPExtractor


# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_DIR  = "./trn"
STIM_DIR   = "./stimuli_train"
SUBJECTS   = [1, 2, 3, 4]


# ── Helpers ──────────────────────────────────────────────────────────────────

def video_path(stim_id: int, stim_dir: str) -> str:
    return os.path.join(stim_dir, f"{int(stim_id):04d}.mp4")


def get_all_train_features(
    extractor: ViCLIPExtractor,
    stim_dir: str,
    cache_dir: str,
    train_dir: str,
) -> np.ndarray:
    """Load or extract features for all 1000 training stimuli (1..1000).

    Returns an array of shape (1001, embed_dim) so that features for
    stimulus ID `sid` can be indexed directly as `all_feats[sid]`.
    """
    cache_path = os.path.join(cache_dir, "viclip_train_all.npy")
    if os.path.exists(cache_path):
        print(f"  Loading complete feature cache from {cache_path}")
        return np.load(cache_path)

    os.makedirs(cache_dir, exist_ok=True)
    all_feats = None

    # Check if a previous run cached sub-01 (968 videos); if so, reuse and only extract the 32 missing
    legacy_path = os.path.join(cache_dir, "viclip_train.npy")
    if os.path.exists(legacy_path):
        print(f"  Found previous cache at {legacy_path}, checking contents...")
        prev_feats = np.load(legacy_path)
        sub1_ids, _ = load_train_eeg(os.path.join(train_dir, "sub-01_trn.npz"))
        if len(prev_feats) == len(sub1_ids):
            embed_dim = prev_feats.shape[1]
            all_feats = np.zeros((1001, embed_dim), dtype=prev_feats.dtype)
            all_feats[sub1_ids] = prev_feats

            missing_ids = [sid for sid in range(1, 1001) if sid not in set(sub1_ids)]
            if missing_ids:
                print(f"  Reusing {len(sub1_ids)} cached features from sub-01; extracting only remaining {len(missing_ids)} videos...")
                paths = [video_path(sid, stim_dir) for sid in missing_ids]
                missing_feats = extractor.extract(paths)
                all_feats[missing_ids] = missing_feats
            else:
                print(f"  All 1000 videos already present.")

    if all_feats is None:
        print("  Extracting ViCLIP features for all 1000 training videos ...")
        paths = [video_path(sid, stim_dir) for sid in range(1, 1001)]
        feats = extractor.extract(paths)
        embed_dim = feats.shape[1]
        all_feats = np.zeros((1001, embed_dim), dtype=feats.dtype)
        all_feats[1:] = feats

    np.save(cache_path, all_feats)
    print(f"  Cached complete 1000-stimulus features → {cache_path}")
    return all_feats


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    extractor = ViCLIPExtractor(ckpt_path=args.viclip_ckpt, device=args.device)
    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Extract/load features for all 1000 training stimuli (indexed 1..1000)
    all_feats = get_all_train_features(extractor, STIM_DIR, args.feat_cache, TRAIN_DIR)

    # 2. Train and evaluate per subject
    for sub in SUBJECTS:
        print(f"\n{'='*60}")
        print(f"  sub-{sub:02d}")
        print(f"{'='*60}")

        # Load and preprocess EEG (z-score removes DC drift & electrode offsets)
        ids, eeg_raw = load_train_eeg(
            os.path.join(TRAIN_DIR, f"sub-{sub:02d}_trn.npz")
        )
        eeg = decimate_eeg(eeg_raw)
        eeg = zscore_per_subject(eeg)

        # Slice video features aligned to this subject's valid trials
        feats = all_feats[ids]
        assert len(feats) == len(eeg), f"Feature / EEG mismatch: {len(feats)} vs {len(eeg)}"

        # Train encoding model
        print("  Training PCA + Ridge ...")
        model = EncodingModel()
        model.fit(feats, eeg)

        # Evaluate on the training split (upper-bound sanity check)
        S     = model.similarity_matrix(eeg, feats)
        ranks = compute_ranks(S)
        print_metrics(retrieval_metrics(ranks), label=f"sub-{sub:02d} [TRAIN]")

        # Save model for inference on test split
        save_path = os.path.join(args.model_dir, f"phase1a_sub-{sub:02d}.joblib")
        model.save(save_path)

    print("\nDone. Run phase1_predict.py to generate test rankings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1A — ViCLIP features")
    parser.add_argument("--viclip_ckpt", default=None,
                        help="HuggingFace model ID or path to .pth (defaults to auto-download 'OpenGVLab/ViCLIP-L-14-hf')")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--feat_cache", default="./cache/viclip",
                        help="Directory to cache extracted features")
    parser.add_argument("--model_dir",  default="./models/phase1a",
                        help="Directory to save trained models")
    main(parser.parse_args())
