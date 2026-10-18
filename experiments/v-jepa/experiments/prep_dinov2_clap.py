"""Extract and cache DINOv2 per-frame video (N,8,1536) + CLAP audio (N,512).

Frame-level layout identical to the V-JEPA2 recipe cache, so DINOv2 flows through
the SAME simple_mlp/train_recipe.py pipeline (DinoPool mean + concat fusion). This
guarantees DINOv2 and V-JEPA2 share backbone/loss/audio/protocol — only the video
encoder differs, which is exactly what the DINO<->VJEPA score blend combines.

Caches written to <cache_dir> (default ./cache/dinov2_frames):
    vjepa2_video_train_all.npy   (1001, 8, 1536)   NOTE: same prefix as the recipe
    vjepa2_video_test.npy        ( 102, 8, 1536)      trainer expects (video_prefix).
    clap_audio_train_all.npy     (1001, 512)       reused from ./cache/viclip_clap
    clap_audio_test.npy          ( 102, 512)

The video prefix is kept as 'vjepa2_video' so train_recipe.py's defaults work
unchanged; just point --feat_cache at this dir. (It is DINOv2 content despite the
filename — the dir name disambiguates.)

Usage (cluster):
    python experiments/prep_dinov2_clap.py --device cuda
    python simple_mlp/train_recipe.py --device cuda --seeds 42,123,999 \
        --select_on heldout --anti_hub csls --final_full_fit \
        --feat_cache ./cache/dinov2_frames
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.dinov2_extractor import DINOv2Extractor, EMBED_DIM
from src.features.clap_extractor import CLAPExtractor

STIM_TRAIN_DIR = "./stimuli_train"
STIM_TEST_DIR = "./stimuli_test"
TEST_IDS = list(range(1001, 1103))


def train_video_path(sid: int) -> str:
    return os.path.join(STIM_TRAIN_DIR, f"{int(sid):04d}.mp4")


def test_video_path(sid: int) -> str:
    return os.path.join(STIM_TEST_DIR, f"{int(sid)}.mp4")


def get_dinov2_frames(ext, cache_dir):
    k = ext.n_frames
    trn = os.path.join(cache_dir, "vjepa2_video_train_all.npy")  # recipe video_prefix
    tst = os.path.join(cache_dir, "vjepa2_video_test.npy")
    if os.path.exists(trn):
        V_trn = np.load(trn); print(f"  Reusing DINOv2 train frames from {trn}")
    else:
        print(f"  Extracting DINOv2 ({k} frames) for 1000 training videos ...")
        raw = ext.extract([train_video_path(s) for s in range(1, 1001)])  # (1000,k,1536)
        V_trn = np.zeros((1001, raw.shape[1], raw.shape[2]), dtype=raw.dtype)
        V_trn[1:] = raw
        np.save(trn, V_trn); print(f"  Saved → {trn}  {V_trn.shape}")
    if os.path.exists(tst):
        V_tst = np.load(tst); print(f"  Reusing DINOv2 test frames from {tst}")
    else:
        print(f"  Extracting DINOv2 ({k} frames) for 102 test videos ...")
        V_tst = ext.extract([test_video_path(s) for s in TEST_IDS])       # (102,k,1536)
        np.save(tst, V_tst); print(f"  Saved → {tst}  {V_tst.shape}")
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
    ap = argparse.ArgumentParser(description="Prep DINOv2 (8 frames) + CLAP")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--layer", type=int, default=11,
                    help="DINOv2 transformer block to read (exp4 uses encoder.layer.11).")
    ap.add_argument("--cache_dir", type=str, default="./cache/dinov2_frames")
    ap.add_argument("--clap_reuse_dir", type=str, default="./cache/viclip_clap")
    ap.add_argument("--dinov2_ckpt", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    v_trn = os.path.join(args.cache_dir, "vjepa2_video_train_all.npy")
    v_tst = os.path.join(args.cache_dir, "vjepa2_video_test.npy")
    c_trn = os.path.join(args.cache_dir, "clap_audio_train_all.npy")
    c_tst = os.path.join(args.cache_dir, "clap_audio_test.npy")
    if all(os.path.exists(p) for p in (v_trn, v_tst, c_trn, c_tst)):
        print(f"All DINOv2 frame + audio caches exist in {args.cache_dir}; nothing to do.")
        print(f"  video train {np.load(v_trn).shape}  audio train {np.load(c_trn).shape}")
        return

    print("=" * 70)
    print(f"  Prep DINOv2 per-frame video (N,{args.n_frames},1536) + CLAP audio (N,512)")
    print(f"  Cache dir: {args.cache_dir}")
    print("=" * 70)

    dn = DINOv2Extractor(model_id=args.dinov2_ckpt, device=args.device,
                         n_frames=args.n_frames, layer=args.layer)
    V_trn, V_tst = get_dinov2_frames(dn, args.cache_dir)
    assert V_trn.shape[1:] == (args.n_frames, EMBED_DIM), f"bad video shape {V_trn.shape}"

    clap = CLAPExtractor(device=args.device, audio_cache_dir=os.path.join(args.cache_dir, "wav"))
    C_trn, C_tst = get_clap_audio(clap, args.cache_dir, args.clap_reuse_dir)

    print(f"\n  Done. DINOv2 frames train {V_trn.shape} test {V_tst.shape}; "
          f"audio train {C_trn.shape} test {C_tst.shape}")
    print("\n  Train with the SAME recipe (concat fusion):")
    print("    python simple_mlp/train_recipe.py --device cuda --seeds 42,123,999 \\")
    print("      --select_on heldout --anti_hub csls --final_full_fit \\")
    print(f"      --feat_cache {args.cache_dir}")


if __name__ == "__main__":
    main()
