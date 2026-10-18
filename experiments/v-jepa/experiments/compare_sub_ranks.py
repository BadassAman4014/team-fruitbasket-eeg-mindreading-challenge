"""Diagnostic & Error Analysis Tool for Sub-01 to Sub-04 against Gold Ground Truth.

Evaluates predicted ranks directly against the true stimulus labels (tst_stim).
Can analyze a single model (e.g. Contrastive) or compare two models side-by-side.

Usage
-----
    # Analyze Contrastive model against original ground truth:
    python experiments/compare_sub_ranks.py --pred_dir ./models/phase2 --gt_dir ./tst_stim

    # Optional side-by-side comparison:
    python experiments/compare_sub_ranks.py --pred_dir ./models/phase2 --baseline_dir ./submissions --gt_dir ./tst_stim
"""

import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted ranks against gold ground truth")
    parser.add_argument("--pred_dir", default="./models/phase2", help="Directory with model predictions (.npy files)")
    parser.add_argument("--baseline_dir", default=None, help="Optional directory with baseline model for comparison")
    parser.add_argument("--gt_dir", default="./tst_stim", help="Directory with ground truth sub-XX_tst_stim.npy")
    parser.add_argument("--subjects", default="1,2,3,4", help="Comma-separated subject numbers")
    return parser.parse_args()


def load_ranks_for_subject(sub_dir: str, sub: int, gt_stim: np.ndarray = None) -> np.ndarray:
    """Load ranks for a subject from whatever file format exists in sub_dir."""
    # 1. Direct ranks file
    candidates = [
        os.path.join(sub_dir, f"sub-{sub:02d}_ranks.npy"),
        os.path.join(sub_dir, f"phase1d_sub-{sub:02d}_ranks.npy"),
        os.path.join(sub_dir, f"phase1a_sub-{sub:02d}_ranks.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return np.load(c)

    # 2. Compute from similarity matrix
    sim_candidates = [
        os.path.join(sub_dir, f"sub-{sub:02d}_similarity.npy"),
        os.path.join(sub_dir, f"phase1d_sub-{sub:02d}_similarity.npy"),
        os.path.join(sub_dir, f"phase1a_sub-{sub:02d}_similarity.npy"),
    ]
    for c in sim_candidates:
        if os.path.exists(c):
            S = np.load(c)
            idx = np.arange(len(S))
            return (-S).argsort(axis=1).argsort(axis=1)[idx, idx] + 1

    # 3. Compute from ranked candidate list
    list_candidates = [
        os.path.join(sub_dir, f"sub-{sub:02d}_ranked_candidate_list.npy"),
        os.path.join(sub_dir, f"phase1d_sub-{sub:02d}_ranked_candidate_list.npy"),
    ]
    for c in list_candidates:
        if os.path.exists(c) and gt_stim is not None:
            ranked_list = np.load(c)  # (102, 102)
            order = np.argsort(gt_stim)
            sorted_gt = gt_stim[order]
            ranks = []
            for i, true_id in enumerate(sorted_gt):
                row = ranked_list[i]
                pos = np.where(row == true_id)[0]
                ranks.append(int(pos[0] + 1) if len(pos) > 0 else 102)
            return np.array(ranks)

    return None


def main():
    args = parse_args()
    subjects = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]

    print("=" * 75)
    print("  Trial-by-Trial Retrieval Diagnostics vs Gold Ground Truth")
    print(f"  Prediction Directory : {args.pred_dir}")
    print(f"  Ground Truth (Gold)  : {args.gt_dir}")
    if args.baseline_dir:
        print(f"  Baseline Directory   : {args.baseline_dir}")
    print("=" * 75)

    all_p_top1 = 0
    all_p_top5 = 0
    all_b_top1 = 0
    all_b_top5 = 0
    either_top1 = 0
    total_trials = 0

    for sub in subjects:
        gt_file = os.path.join(args.gt_dir, f"sub-{sub:02d}_tst_stim.npy")
        if not os.path.exists(gt_file):
            print(f"\n[sub-{sub:02d}] Missing gold ground truth at {gt_file}")
            continue

        gt_stim = np.load(gt_file)
        order = np.argsort(gt_stim)
        sorted_gt = gt_stim[order]  # (102,) true stimulus IDs

        ranks_pred = load_ranks_for_subject(args.pred_dir, sub, gt_stim)
        if ranks_pred is None:
            print(f"\n[sub-{sub:02d}] No prediction files found in {args.pred_dir}")
            continue

        n = len(ranks_pred)
        total_trials += n

        p_t1 = (ranks_pred == 1)
        p_t5 = (ranks_pred <= 5)
        all_p_top1 += p_t1.sum()
        all_p_top5 += p_t5.sum()

        print(f"\n{'─'*75}")
        print(f"  Subject: sub-{sub:02d} ({n} test trials)")
        print(f"{'─'*75}")
        print(f"  Top-1 Accuracy : {p_t1.sum():2d}/{n} ({p_t1.mean()*100:.1f}%) [chance = 0.98%]")
        print(f"  Top-5 Accuracy : {p_t5.sum():2d}/{n} ({p_t5.mean()*100:.1f}%) [chance = 4.90%]")
        print(f"  Median Rank    : {np.median(ranks_pred):.1f} / {n} [chance = 51.5]")

        # List all Top-1 Hits
        top1_indices = np.where(p_t1)[0]
        if len(top1_indices) > 0:
            hits = [f"Trial {i:2d} (Video {sorted_gt[i]})" for i in top1_indices]
            print(f"\n  🎯 Perfect Top-1 Hits ({len(top1_indices)} trials):")
            for h in hits:
                print(f"     • {h}")

        # List Near Misses (Rank 2 or 3)
        near_misses = np.where((ranks_pred >= 2) & (ranks_pred <= 3))[0]
        if len(near_misses) > 0:
            misses = [f"Trial {i:2d} (Video {sorted_gt[i]} -> Rank {ranks_pred[i]})" for i in near_misses]
            print(f"\n  🔍 Near Misses [Ranks 2-3] ({len(near_misses)} trials):")
            for m in misses:
                print(f"     • {m}")

        # If baseline provided, compare side-by-side
        if args.baseline_dir:
            ranks_base = load_ranks_for_subject(args.baseline_dir, sub, gt_stim)
            if ranks_base is not None:
                b_t1 = (ranks_base == 1)
                b_t5 = (ranks_base <= 5)
                all_b_top1 += b_t1.sum()
                all_b_top5 += b_t5.sum()
                union_t1 = np.sum(p_t1 | b_t1)
                either_top1 += union_t1

                print(f"\n  Comparison vs Baseline:")
                print(f"    Baseline Top-1 : {b_t1.sum():2d}/{n} ({b_t1.mean()*100:.1f}%) | Top-5: {b_t5.sum():2d}/{n} ({b_t5.mean()*100:.1f}%) | Median: {np.median(ranks_base):.1f}")
                print(f"    Both Hit       : {np.sum(p_t1 & b_t1)}")
                print(f"    Only Model     : {np.sum(p_t1 & ~b_t1)}")
                print(f"    Only Baseline  : {np.sum(~p_t1 & b_t1)}")
                print(f"    Union (Either) : {union_t1}/{n} ({union_t1/n*100:.1f}%)")

    if total_trials > 0:
        print(f"\n{'='*75}")
        print(f"  TOTAL SUMMARY ({total_trials} Test Trials across sub-01..04)")
        print(f"{'='*75}")
        print(f"  Average Top-1 : {all_p_top1}/{total_trials} ({all_p_top1/total_trials*100:.2f}%)")
        print(f"  Average Top-5 : {all_p_top5}/{total_trials} ({all_p_top5/total_trials*100:.2f}%)")
        if args.baseline_dir and all_b_top1 > 0:
            print(f"  Baseline Top-1: {all_b_top1}/{total_trials} ({all_b_top1/total_trials*100:.2f}%)")
            print(f"  Baseline Top-5: {all_b_top5}/{total_trials} ({all_b_top5/total_trials*100:.2f}%)")
            print(f"  Combined Ceiling (Oracle picking better per trial): {either_top1}/{total_trials} ({either_top1/total_trials*100:.2f}%)")
        print("=" * 75)


if __name__ == "__main__":
    main()
