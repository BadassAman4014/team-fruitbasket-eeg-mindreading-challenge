"""Diagnostics for the simple-MLP rankings on sub-01..04 (and hubness on sub-05/06).

Uses the per-subject similarity matrices that train.py saves
(./submissions/simple_mlp_sub-XX_similarity.npy) and the ground-truth labels in
tst_stim/ (available for sub-01..04 only). Ground-truth labels are used purely
for ANALYSIS here — never for training or model selection.

What it checks (the failure modes discussed):
  - Hubness: how concentrated the rank-1 predictions are across videos. If a few
    "generic" videos win rank 1 for many trials, that's a hubness bug.
  - Rank distribution of the true video (how often it lands in each rank bucket).
  - Per-subject top-1 / top-5 / median rank.
  - Worst cases: trials where the true video was ranked very low.
  - Cross-subject agreement: are the same videos consistently (in)correct?

Usage
-----
    python simple_mlp/diagnostics.py
    python simple_mlp/diagnostics.py --sub_dir ./submissions --topn 10
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_ranks, retrieval_metrics

TEST_IDS = np.array(range(1001, 1103))  # 102 candidate video IDs
EVAL_SUBJECTS = [1, 2, 3, 4]
PRED_SUBJECTS = [5, 6]


def parse_args():
    p = argparse.ArgumentParser(description="Simple-MLP ranking diagnostics")
    p.add_argument("--sub_dir", type=str, default="./submissions",
                   help="Directory with simple_mlp_sub-XX_similarity.npy from train.py")
    p.add_argument("--gt_dir", type=str, default="./tst_stim")
    p.add_argument("--topn", type=int, default=10, help="How many top hub videos / worst cases to list")
    return p.parse_args()


def hubness_report(ranked_ids: np.ndarray, topn: int) -> None:
    """ranked_ids: (102, 102) of video IDs, most→least likely per row.

    Reports how concentrated rank-1 predictions are. A healthy model spreads
    rank-1 across many distinct videos; a hub problem shows a few IDs dominating.
    """
    rank1 = ranked_ids[:, 0]
    counts = Counter(rank1.tolist())
    n_trials = len(rank1)
    n_distinct = len(counts)

    # Herfindahl-style concentration: 1/n_distinct is the uniform ideal.
    freqs = np.array(list(counts.values()), dtype=float) / n_trials
    concentration = float((freqs ** 2).sum())  # 1/n_trials (all unique) → 1.0 (one hub)

    print(f"    rank-1 predictions: {n_distinct} distinct videos out of {n_trials} trials")
    print(f"    concentration index: {concentration:.3f} "
          f"(ideal≈{1.0/n_trials:.3f} all-unique, 1.000 single hub)")
    top = counts.most_common(topn)
    hub_line = ", ".join(f"{vid}×{c}" for vid, c in top)
    print(f"    top rank-1 videos: {hub_line}")
    worst_hub, worst_c = top[0]
    if worst_c >= max(5, n_trials // 10):
        print(f"    ⚠ possible hub: video {worst_hub} is rank-1 for {worst_c}/{n_trials} trials")


def rank_histogram(ranks: np.ndarray) -> None:
    """Bucketed distribution of the true video's rank."""
    buckets = [(1, 1), (2, 5), (6, 10), (11, 25), (26, 51), (52, 102)]
    n = len(ranks)
    parts = []
    for lo, hi in buckets:
        c = int(((ranks >= lo) & (ranks <= hi)).sum())
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        parts.append(f"[{label}]={c} ({c/n*100:.0f}%)")
    print("    true-video rank buckets: " + "  ".join(parts))


def main():
    args = parse_args()

    print("=" * 70)
    print("  Simple-MLP Ranking Diagnostics")
    print("=" * 70)

    gt_ranks_by_sub = {}

    for sub in EVAL_SUBJECTS + PRED_SUBJECTS:
        sim_path = os.path.join(args.sub_dir, f"simple_mlp_sub-{sub:02d}_similarity.npy")
        if not os.path.exists(sim_path):
            print(f"\nsub-{sub:02d}: no similarity matrix at {sim_path} — run train.py first. Skipping.")
            continue

        S = np.load(sim_path)  # (102, 102)
        ranked_ids = TEST_IDS[np.argsort(-S, axis=1)]  # (102, 102) video IDs

        print("\n" + "-" * 70)
        print(f"  sub-{sub:02d}")
        print("-" * 70)

        # Hubness is label-free, so we report it for every subject (incl. held-out).
        print("  [Hubness]")
        hubness_report(ranked_ids, args.topn)

        gt_path = os.path.join(args.gt_dir, f"sub-{sub:02d}_tst_stim.npy")
        if not os.path.exists(gt_path):
            print("  [Accuracy] no ground truth (held-out subject) — hubness only.")
            continue

        # For eval subjects, the saved matrix rows are label-sorted so diagonal = correct.
        gt = np.load(gt_path)
        m = retrieval_metrics(compute_ranks(S))
        print("  [Accuracy]")
        print(f"    top-1 {m['top1']*100:5.2f}% ({m['top1']*102:2.0f}/102)  |  "
              f"top-5 {m['top5']*100:5.2f}% ({m['top5']*102:2.0f}/102)  |  "
              f"median rank {m['median_rank']:.1f}")

        ranks = compute_ranks(S)
        gt_ranks_by_sub[sub] = ranks
        print("  [Rank distribution]")
        rank_histogram(ranks)

        # Worst cases: the correct video sorted furthest down.
        order = np.argsort(-ranks)[:args.topn]
        print(f"  [Worst {args.topn} cases] (trial_idx: true_rank)")
        worst = ", ".join(f"{int(i)}:{int(ranks[i])}" for i in order)
        print(f"    {worst}")

    # Cross-subject agreement on the eval subjects.
    common = [s for s in EVAL_SUBJECTS if s in gt_ranks_by_sub]
    if len(common) >= 2:
        print("\n" + "-" * 70)
        print("  Cross-subject consistency (sub-01..04 with ground truth)")
        print("-" * 70)
        # Note: trials are label-sorted per subject, so trial index i corresponds to
        # the i-th sorted stimulus ID — comparable across subjects only if the sorted
        # label sets match. We compare the per-trial correctness pattern length-wise.
        hit1 = {s: (gt_ranks_by_sub[s] == 1).astype(int) for s in common}
        stacked = np.vstack([hit1[s] for s in common])  # (n_sub, 102)
        agree_all = int((stacked.sum(axis=0) == len(common)).sum())
        agree_none = int((stacked.sum(axis=0) == 0).sum())
        print(f"    trials correct for ALL {len(common)} subjects: {agree_all}/102")
        print(f"    trials correct for NO subject:            {agree_none}/102")
        print("    (high 'all-correct' count may indicate feature-side signal "
              "rather than EEG-side; high 'none' is the hard core.)")

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
