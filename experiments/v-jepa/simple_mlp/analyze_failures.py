"""Failure-mode analysis for the simple-MLP rankings — makes the hard cases human-viewable.

Answers three questions the plain diagnostics could not:

  1. WHICH video files are the hard trials?  For each recurring hard trial it prints
     the true video ID, its .mp4 path (so you can open it), the rank the model gave
     it, and the top-k videos the model wrongly ranked ABOVE it (the confusers).

  2. WHAT are the attractor (hub) videos, and WHY?  It measures each candidate's
     mean similarity to all EEG trials (hubness), and separately recomputes hubness
     using only the VIDEO half (ViCLIP) vs only the AUDIO half (CLAP) of the feature.
     This tests the hypothesis that videos with weak/non-semantic audio become hubs
     because their CLAP embedding is degenerate and sits near the centre of the space.

  3. The HARD-CORE video IDs (wrong for every eval subject) listed as .mp4 paths,
     so you can eyeball what they share.

Everything uses sub-01..04 ground truth for ANALYSIS only — never for training/selection.

Usage
-----
    python simple_mlp/analyze_failures.py
    python simple_mlp/analyze_failures.py --hard_trials 19,96,82,30,50,78 --topk 5
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_ranks

TEST_IDS = np.array(range(1001, 1103))  # candidate video IDs
EVAL_SUBJECTS = [1, 2, 3, 4]
ALL_SUBJECTS = [1, 2, 3, 4, 5, 6]
VICLIP_DIM = 768  # leading video block in the cached feature


def parse_args():
    p = argparse.ArgumentParser(description="Failure-mode analysis for simple-MLP")
    p.add_argument("--sub_dir", type=str, default="./submissions",
                   help="Dir with simple_mlp_sub-XX_similarity.npy")
    p.add_argument("--gt_dir", type=str, default="./tst_stim")
    p.add_argument("--stim_test_dir", type=str, default="./stimuli_test",
                   help="Folder with the 102 test videos 1001.mp4 .. 1102.mp4")
    p.add_argument("--feat_cache", type=str, default="./cache/viclip_clap")
    p.add_argument("--feat_prefix", type=str, default="viclip_clap")
    p.add_argument("--vid_dim", type=int, default=VICLIP_DIM)
    p.add_argument("--hard_trials", type=str, default="",
                   help="Comma-separated trial indices to inspect; blank = auto-detect recurring hard trials.")
    p.add_argument("--topk", type=int, default=5, help="How many confuser videos to list per hard trial.")
    p.add_argument("--n_hub", type=int, default=12, help="How many top hub videos to list.")
    return p.parse_args()


def video_path(stim_dir: str, vid_id: int) -> str:
    return os.path.join(stim_dir, f"{int(vid_id)}.mp4")


def load_eval_sims(sub_dir: str, gt_dir: str):
    """Return {sub: (S, sorted_gt_ids)} for eval subjects that have a saved sim matrix.

    S rows are label-sorted (row i ↔ sorted_gt_ids[i]); column j ↔ TEST_IDS[j].
    """
    out = {}
    for sub in EVAL_SUBJECTS:
        sp = os.path.join(sub_dir, f"simple_mlp_sub-{sub:02d}_similarity.npy")
        gp = os.path.join(gt_dir, f"sub-{sub:02d}_tst_stim.npy")
        if not (os.path.exists(sp) and os.path.exists(gp)):
            continue
        S = np.load(sp)
        gt = np.load(gp)
        out[sub] = (S, np.sort(gt))  # rows were sorted by gt id when saved
    return out


def auto_hard_trials(eval_sims: dict, min_subjects: int = 3, worst_n: int = 15) -> list:
    """Trial indices that are in the worst-N for at least `min_subjects` eval subjects."""
    counter = Counter()
    for sub, (S, _) in eval_sims.items():
        ranks = compute_ranks(S)
        worst = np.argsort(-ranks)[:worst_n]
        counter.update(int(i) for i in worst)
    return [idx for idx, c in counter.most_common() if c >= min_subjects]


def section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main():
    args = parse_args()
    eval_sims = load_eval_sims(args.sub_dir, args.gt_dir)
    if not eval_sims:
        print(f"No eval similarity matrices found in {args.sub_dir}. Run train.py first.")
        return

    # ---- 1. Hard trials → video files + confusers ----
    section("1. Hard trials — true video, file path, and the confusers ranked above it")

    if args.hard_trials.strip():
        hard = [int(x) for x in args.hard_trials.split(",") if x.strip()]
        print(f"  (using user-supplied trial indices: {hard})")
    else:
        hard = auto_hard_trials(eval_sims)
        print(f"  (auto-detected recurring hard trials: {hard})")

    for sub, (S, sorted_gt) in eval_sims.items():
        print(f"\n  --- sub-{sub:02d} ---")
        ranks = compute_ranks(S)
        for ti in hard:
            if ti >= S.shape[0]:
                continue
            true_id = int(sorted_gt[ti])
            r = int(ranks[ti])
            # Columns (videos) ranked above the true one = confusers.
            order = np.argsort(-S[ti])          # best → worst column indices
            true_col = int(np.where(TEST_IDS == true_id)[0][0])
            above = [int(TEST_IDS[c]) for c in order if c != true_col][: args.topk]
            print(f"    trial {ti:3d} | true={true_id} ({video_path(args.stim_test_dir, true_id)}) "
                  f"| rank={r:3d}")
            print(f"               confusers (ranked above): "
                  + ", ".join(str(v) for v in above))

    # ---- 2. Hub videos & the video-vs-audio hubness split ----
    section("2. Attractor (hub) videos — overall, and video-only vs audio-only")

    # Count rank-1 wins across eval subjects (observed hub behaviour).
    win = Counter()
    for sub, (S, _) in eval_sims.items():
        win.update(int(TEST_IDS[np.argmax(S[i])]) for i in range(S.shape[0]))
    print("  Observed rank-1 wins across eval subjects (top hubs):")
    print("    " + ", ".join(f"{v}×{c}" for v, c in win.most_common(args.n_hub)))

    # Feature-space centrality: a hub video should have high mean cosine similarity
    # to all other candidate videos. Compute this for both / video / audio feature halves.
    test_feat_path = os.path.join(args.feat_cache, f"{args.feat_prefix}_test.npy")
    if os.path.exists(test_feat_path):
        F = np.load(test_feat_path).astype(np.float64)  # (102, full_dim)

        def centrality(block: np.ndarray) -> np.ndarray:
            # cosine centrality = mean similarity of each video to all others
            n = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-8)
            Sv = n @ n.T
            np.fill_diagonal(Sv, np.nan)
            return np.nanmean(Sv, axis=1)

        cent_both = centrality(F)
        cent_vid = centrality(F[:, : args.vid_dim])
        cent_aud = centrality(F[:, args.vid_dim:])

        def top_central(cent, k=args.n_hub):
            idx = np.argsort(-cent)[:k]
            return [(int(TEST_IDS[i]), float(cent[i])) for i in idx]

        print("\n  Feature-space centrality (higher = closer to centre = more hub-prone):")
        print("    BOTH  :", ", ".join(f"{v}({c:.2f})" for v, c in top_central(cent_both)))
        print("    VIDEO :", ", ".join(f"{v}({c:.2f})" for v, c in top_central(cent_vid)))
        print("    AUDIO :", ", ".join(f"{v}({c:.2f})" for v, c in top_central(cent_aud)))

        # Correlate observed rank-1 wins with audio-centrality to test the hypothesis.
        win_vec = np.array([win.get(int(v), 0) for v in TEST_IDS], dtype=float)
        for name, cent in [("both", cent_both), ("video", cent_vid), ("audio", cent_aud)]:
            if win_vec.std() > 0 and cent.std() > 0:
                r = float(np.corrcoef(win_vec, cent)[0, 1])
                print(f"    corr(rank-1 wins, {name:5s} centrality) = {r:+.3f}")
        print("    → a strong positive AUDIO correlation supports: weak-audio videos "
              "become hubs via degenerate CLAP embeddings.")
    else:
        print(f"\n  (feature cache {test_feat_path} not found — skipping centrality analysis)")

    # ---- 3. Hard-core videos (wrong for every eval subject) ----
    section("3. Hard-core videos — the true video wrong (rank>5) for ALL eval subjects")

    # Build per-subject correctness keyed by VIDEO ID (not trial index), then intersect.
    wrong_sets = []
    for sub, (S, sorted_gt) in eval_sims.items():
        ranks = compute_ranks(S)
        wrong_ids = {int(sorted_gt[i]) for i in range(S.shape[0]) if ranks[i] > 5}
        wrong_sets.append(wrong_ids)

    hard_core = []
    if wrong_sets:
        hard_core = sorted(set.intersection(*wrong_sets))
        print(f"  {len(hard_core)} videos are top-5-wrong for ALL {len(wrong_sets)} eval subjects:")
        for vid in hard_core:
            print(f"    {vid}  ->  {video_path(args.stim_test_dir, vid)}")
        print("\n  Eyeball these .mp4s for a shared trait (low motion, weak/no audio, "
              "visually generic). That trait is the stimulus-side accuracy ceiling.")

    # ---- 4. Objective test of the 'noisy / mismatched audio' pattern ----
    #     For each video, measure:
    #       (a) ViCLIP<->CLAP AGREEMENT: do the video-half and audio-half agree on
    #           which OTHER videos are its neighbours? Low agreement = the two
    #           modalities describe different scenes = muddled fused embedding.
    #       (b) DISTINCTIVENESS: nearest-neighbour cosine distance in the fused space.
    #           Low = generic, easily-confused video.
    #     Then compare the hard-core group vs the readable group on both.
    section("4. Audio-visual agreement & distinctiveness — hard-core vs readable")

    test_feat_path = os.path.join(args.feat_cache, f"{args.feat_prefix}_test.npy")
    if not os.path.exists(test_feat_path):
        print(f"  (feature cache {test_feat_path} not found — skipping)")
        print("\nFailure analysis complete.")
        return
    if not hard_core:
        print("  (no hard-core set — skipping)")
        print("\nFailure analysis complete.")
        return

    F = np.load(test_feat_path).astype(np.float64)  # (102, full_dim)

    def l2norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    V = l2norm(F[:, : args.vid_dim])     # ViCLIP half
    A = l2norm(F[:, args.vid_dim:])      # CLAP half
    B = l2norm(F)                        # fused

    Sv = V @ V.T                         # (102,102) video-half similarity
    Sa = A @ A.T                         # audio-half similarity
    Sb = B @ B.T                         # fused similarity
    for M in (Sv, Sa, Sb):
        np.fill_diagonal(M, np.nan)

    # (a) Agreement: per-video correlation between its video-half and audio-half
    #     similarity profile to all other videos. High = modalities agree.
    agreement = np.zeros(len(F))
    for i in range(len(F)):
        v = Sv[i]; a = Sa[i]
        mask = ~(np.isnan(v) | np.isnan(a))
        agreement[i] = np.corrcoef(v[mask], a[mask])[0, 1] if mask.sum() > 2 else np.nan

    # (b) Distinctiveness: 1 - max similarity to any other video in the fused space.
    distinctiveness = 1.0 - np.nanmax(Sb, axis=1)

    id_to_idx = {int(vid): k for k, vid in enumerate(TEST_IDS)}
    hard_idx = np.array([id_to_idx[v] for v in hard_core])
    read_idx = np.array([k for k in range(len(TEST_IDS)) if TEST_IDS[k] not in set(hard_core)])

    def summarize(name, arr):
        h = arr[hard_idx]; r = arr[read_idx]
        hm, rm = np.nanmean(h), np.nanmean(r)
        # Pooled-std effect size (Cohen's d): magnitude of the gap between groups.
        hs, rs = np.nanstd(h), np.nanstd(r)
        pooled = np.sqrt((hs**2 + rs**2) / 2) + 1e-8
        d = (hm - rm) / pooled
        print(f"  {name:16s} | hard-core mean {hm:+.3f} | readable mean {rm:+.3f} | "
              f"gap {hm - rm:+.3f} | Cohen's d {d:+.2f}")
        return hm, rm

    print(f"  Groups: hard-core n={len(hard_idx)}  vs  readable n={len(read_idx)}\n")
    summarize("AV agreement", agreement)
    summarize("distinctiveness", distinctiveness)

    print("\n  Reading the result:")
    print("    - If hard-core has LOWER AV agreement (negative gap, |d|>=0.3): supports the")
    print("      'noisy / mismatched audio' pattern — audio & video describe different scenes.")
    print("    - If hard-core has LOWER distinctiveness: the videos are generic/confusable,")
    print("      a visual-typicality ceiling independent of audio.")
    print("    - If neither gap is meaningful (|d|<0.2): the hard core is NOT explained by")
    print("      these feature statistics — the limit is elsewhere (EEG signal / subject).")

    # Rank the hard-core videos by lowest AV agreement — the clearest examples to show.
    order = hard_idx[np.argsort(agreement[hard_idx])]
    print("\n  Hard-core videos with the LOWEST audio-visual agreement (best slide examples):")
    for k in order[:8]:
        print(f"    {int(TEST_IDS[k])}  AV_agreement={agreement[k]:+.3f}  "
              f"distinct={distinctiveness[k]:+.3f}  ->  {video_path(args.stim_test_dir, int(TEST_IDS[k]))}")

    print("\nFailure analysis complete.")


if __name__ == "__main__":
    main()
