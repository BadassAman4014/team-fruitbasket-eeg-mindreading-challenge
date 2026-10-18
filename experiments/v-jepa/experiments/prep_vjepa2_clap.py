"""Extract and cache V-JEPA2 per-sub-clip video (N,8,1024) + CLAP audio (N,512).

exp4-faithful layout: video features are kept FRAME-LEVEL (8 sub-clips) and audio
is kept SEPARATE, so the recipe model can pool frames (DinoPool) and fuse with the
audio branch exactly like exp4. CLAP audio is reused from ./cache/viclip_clap.

Caches written to <cache_dir>:
    vjepa2_video_train_all.npy   (1001, 8, 1024)   1-based ID index (row 0 = pad)
    vjepa2_video_test.npy        ( 102, 8, 1024)
    clap_audio_train_all.npy     (1001, 512)
    clap_audio_test.npy          ( 102, 512)

Usage (cluster, needs transformers>=4.52):
    python experiments/prep_vjepa2_clap.py --device cuda
    python simple_mlp/train_recipe.py --device cuda --seeds 42,123,999 \
        --select_on heldout --anti_hub csls --final_full_fit \
        --feat_cache ./cache/vjepa2_clap
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.vjepa2_extractor import VJEPA2Extractor, EMBED_DIM, N_SUBCLIPS
from src.features.clap_extractor import CLAPExtractor

STIM_TRAIN_DIR = "./stimuli_train"
STIM_TEST_DIR = "./stimuli_test"
TEST_IDS = list(range(1001, 1103))


def train_video_path(sid: int) -> str:
    return os.path.join(STIM_TRAIN_DIR, f"{int(sid):04d}.mp4")


def test_video_path(sid: int) -> str:
    return os.path.join(STIM_TEST_DIR, f"{int(sid)}.mp4")


def get_vjepa2_frames(ext, cache_dir):
    """Per-sub-clip video features. Returns (1001,K,1024) train, (102,K,1024) test,
    where K = ext.n_subclips (NOT the module default)."""
    k = ext.n_subclips
    trn = os.path.join(cache_dir, "vjepa2_video_train_all.npy")
    tst = os.path.join(cache_dir, "vjepa2_video_test.npy")

    if os.path.exists(trn):
        V_trn = np.load(trn); print(f"  Reusing V-JEPA2 train frames from {trn}")
    else:
        print(f"  Extracting V-JEPA2 ({k} sub-clips) for 1000 training videos ...")
        raw = ext.extract([train_video_path(s) for s in range(1, 1001)])  # (1000,k,1024)
        V_trn = np.zeros((1001, raw.shape[1], raw.shape[2]), dtype=raw.dtype)
        V_trn[1:] = raw
        np.save(trn, V_trn)
        print(f"  Saved → {trn}  {V_trn.shape}")

    if os.path.exists(tst):
        V_tst = np.load(tst); print(f"  Reusing V-JEPA2 test frames from {tst}")
    else:
        print(f"  Extracting V-JEPA2 ({k} sub-clips) for 102 test videos ...")
        V_tst = ext.extract([test_video_path(s) for s in TEST_IDS])       # (102,k,1024)
        np.save(tst, V_tst)
        print(f"  Saved → {tst}  {V_tst.shape}")
    return V_trn, V_tst


def get_clap_audio(clap_ext, cache_dir, reuse_dir):
    trn = os.path.join(cache_dir, "clap_audio_train_all.npy")
    tst = os.path.join(cache_dir, "clap_audio_test.npy")
    r_trn = os.path.join(reuse_dir, "clap_audio_train_all.npy")
    r_tst = os.path.join(reuse_dir, "clap_audio_test.npy")

    if os.path.exists(trn):
        C_trn = np.load(trn)
    elif os.path.exists(r_trn):
        C_trn = np.load(r_trn); np.save(trn, C_trn); print(f"  Reused CLAP train from {r_trn}")
    else:
        print("  Extracting CLAP train audio ...")
        raw = clap_ext.extract_audio([train_video_path(s) for s in range(1, 1001)])
        C_trn = np.zeros((1001, raw.shape[1]), dtype=raw.dtype); C_trn[1:] = raw
        np.save(trn, C_trn)

    if os.path.exists(tst):
        C_tst = np.load(tst)
    elif os.path.exists(r_tst):
        C_tst = np.load(r_tst); np.save(tst, C_tst); print(f"  Reused CLAP test from {r_tst}")
    else:
        print("  Extracting CLAP test audio ...")
        C_tst = clap_ext.extract_audio([test_video_path(s) for s in TEST_IDS])
        np.save(tst, C_tst)
    return C_trn, C_tst


def main():
    ap = argparse.ArgumentParser(description="Prep V-JEPA2 (8 sub-clips) + CLAP")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_subclips", type=int, default=8,
                    help="Number of temporal sub-clips per video (tunable; extraction-side).")
    ap.add_argument("--frames_per_subclip", type=int, default=8)
    # Per-count cache dir so different --n_subclips never collide. Default dir embeds the count.
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="Default: ./cache/vjepa2_frames_k<n_subclips>")
    ap.add_argument("--clap_reuse_dir", type=str, default="./cache/viclip_clap")
    ap.add_argument("--vjepa2_ckpt", type=str, default=None)
    args = ap.parse_args()

    if args.cache_dir is None:
        args.cache_dir = f"./cache/vjepa2_frames_k{args.n_subclips}"
    os.makedirs(args.cache_dir, exist_ok=True)
    v_trn = os.path.join(args.cache_dir, "vjepa2_video_train_all.npy")
    v_tst = os.path.join(args.cache_dir, "vjepa2_video_test.npy")
    c_trn = os.path.join(args.cache_dir, "clap_audio_train_all.npy")
    c_tst = os.path.join(args.cache_dir, "clap_audio_test.npy")
    if all(os.path.exists(p) for p in (v_trn, v_tst, c_trn, c_tst)):
        print(f"All frame + audio caches exist in {args.cache_dir}; nothing to do.")
        print(f"  video train {np.load(v_trn).shape}  audio train {np.load(c_trn).shape}")
        return

    print("=" * 70)
    print(f"  Prep V-JEPA2 per-sub-clip video (N,{args.n_subclips},1024) + CLAP audio (N,512)")
    print(f"  Cache dir: {args.cache_dir}")
    print("=" * 70)

    vj = VJEPA2Extractor(model_id=args.vjepa2_ckpt, device=args.device,
                         n_subclips=args.n_subclips, frames_per_subclip=args.frames_per_subclip)
    V_trn, V_tst = get_vjepa2_frames(vj, args.cache_dir)
    assert V_trn.shape[1:] == (args.n_subclips, EMBED_DIM), f"bad video shape {V_trn.shape}"

    clap = CLAPExtractor(device=args.device, audio_cache_dir=os.path.join(args.cache_dir, "wav"))
    C_trn, C_tst = get_clap_audio(clap, args.cache_dir, args.clap_reuse_dir)

    print(f"\n  Done. Video frames train {V_trn.shape} test {V_tst.shape}; "
          f"audio train {C_trn.shape} test {C_tst.shape}")
    print("\n  Train with (exp4-faithful, concat fusion):")
    print("    python simple_mlp/train_recipe.py --device cuda --seeds 42,123,999 \\")
    print("      --select_on heldout --anti_hub csls --final_full_fit \\")
    print(f"      --feat_cache {args.cache_dir}")


if __name__ == "__main__":
    main()
