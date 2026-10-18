"""Phase 1C — ViCLIP video + ImageBind audio features (hybrid).

Combines the best of both worlds:
  - ViCLIP: stronger spatiotemporal video representation
  - ImageBind: audio embedding aligned to the same metric space

Features are concatenated: ViCLIP (768-d) + ImageBind audio (1024-d) → 1792-d.
PCA will reduce this; Ridge handles the rest.

Usage (cluster)
---------------
    python experiments/phase1_1c_hybrid.py --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils      import load_train_eeg, decimate_eeg, zscore_per_subject
from src.pipeline        import EncodingModel
from src.metrics         import compute_ranks, retrieval_metrics, print_metrics
from src.features.viclip_extractor    import ViCLIPExtractor
from src.features.imagebind_extractor import ImageBindExtractor


TRAIN_DIR = "./trn"
STIM_DIR  = "./stimuli_train"
SUBJECTS  = [1, 2, 3, 4]


def video_path(stim_id: int, stim_dir: str) -> str:
    return os.path.join(stim_dir, f"{int(stim_id):04d}.mp4")


def get_all_hybrid_features(
    viclip_ext: ViCLIPExtractor,
    ib_ext: ImageBindExtractor,
    stim_dir: str,
    cache_dir: str,
    viclip_cache_dir: str = "./cache/viclip",
) -> np.ndarray:
    """Extract and cache ViCLIP (video) + ImageBind (audio) features for 1000 stimuli."""
    cache_path = os.path.join(cache_dir, "hybrid_train_all.npy")
    if os.path.exists(cache_path):
        print(f"  Loading complete hybrid feature cache from {cache_path}")
        return np.load(cache_path)

    os.makedirs(cache_dir, exist_ok=True)
    paths = [video_path(sid, stim_dir) for sid in range(1, 1001)]

    # 1. ViCLIP video features (1001, 768)
    viclip_all_path = os.path.join(viclip_cache_dir, "viclip_train_all.npy")
    if os.path.exists(viclip_all_path):
        print(f"  Reusing complete ViCLIP video features from {viclip_all_path}")
        V = np.load(viclip_all_path)
    else:
        print(f"  Extracting ViCLIP features for all 1000 videos ...")
        raw_v = viclip_ext.extract(paths)
        V = np.zeros((1001, raw_v.shape[1]), dtype=raw_v.dtype)
        V[1:] = raw_v

    # 2. ImageBind audio features (1001, 1024)
    audio_cache = os.path.join(cache_dir, "imagebind_audio_train_all.npy")
    if os.path.exists(audio_cache):
        print(f"  Reusing complete ImageBind audio features from {audio_cache}")
        A = np.load(audio_cache)
    else:
        print(f"  Extracting ImageBind audio features for all 1000 videos ...")
        raw_a = ib_ext.extract_audio(paths)
        A = np.zeros((1001, raw_a.shape[1]), dtype=raw_a.dtype)
        A[1:] = raw_a
        np.save(audio_cache, A)

    all_feats = np.concatenate([V, A], axis=1)   # (1001, 768+1024=1792)
    np.save(cache_path, all_feats)
    print(f"  Hybrid features cached → {cache_path}  shape={all_feats.shape}")
    return all_feats


def main(args):
    viclip_ext = ViCLIPExtractor(ckpt_path=args.viclip_ckpt, device=args.device)
    ib_ext     = ImageBindExtractor(
        device=args.device,
        audio_cache_dir=os.path.join(args.feat_cache, "wav"),
    )
    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Extract/load features for all 1000 training stimuli
    all_feats = get_all_hybrid_features(
        viclip_ext, ib_ext, STIM_DIR, args.feat_cache
    )

    # 2. Train and evaluate per subject
    for sub in SUBJECTS:
        print(f"\n{'='*60}")
        print(f"  sub-{sub:02d}  [Phase 1C — ViCLIP + ImageBind audio]")
        print(f"{'='*60}")

        ids, eeg_raw = load_train_eeg(
            os.path.join(TRAIN_DIR, f"sub-{sub:02d}_trn.npz")
        )
        eeg   = decimate_eeg(eeg_raw)
        eeg   = zscore_per_subject(eeg)
        feats = all_feats[ids]
        assert len(feats) == len(eeg), f"Feature / EEG mismatch: {len(feats)} vs {len(eeg)}"

        print("  Training PCA + Ridge ...")
        model = EncodingModel()
        model.fit(feats, eeg)

        S     = model.similarity_matrix(eeg, feats)
        ranks = compute_ranks(S)
        print_metrics(retrieval_metrics(ranks), label=f"sub-{sub:02d} [TRAIN]")

        save_path = os.path.join(
            args.model_dir, f"phase1c_sub-{sub:02d}.joblib"
        )
        model.save(save_path)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1C — ViCLIP video + ImageBind audio (hybrid)"
    )
    parser.add_argument("--viclip_ckpt", default=None,
                        help="HuggingFace model ID or path to .pth (defaults to auto-download 'OpenGVLab/ViCLIP-L-14-hf')")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--feat_cache", default="./cache/hybrid")
    parser.add_argument("--model_dir",  default="./models/phase1c")
    main(parser.parse_args())
