"""
src/cv.py - Cross-Validation Splits and Video-Level Partitioning
================================================================
Implements WP4 of the Contrastive Retrieval Plan:
- make_video_folds: splits videos across all subjects at once (preventing subject leakage).
- candidate_subsets: generates standard candidate pools for evaluation.
"""

import numpy as np


def make_video_folds(train_ids, k=5, seed=0):
    """
    Partitions video IDs into k balanced folds.
    Holding out a video holds it out for all subjects simultaneously.
    Returns: list of (train_video_ids, val_video_ids)
    """
    rng = np.random.default_rng(seed)
    shuffled_vids = rng.permutation(np.array(sorted(list(train_ids))))
    folds = np.array_split(shuffled_vids, k)

    fold_splits = []
    for f_idx in range(k):
        val_vids = set(folds[f_idx])
        trn_vids = [v for v in shuffled_vids if v not in val_vids]
        fold_splits.append((trn_vids, list(val_vids)))

    return fold_splits


def get_trial_table_for_videos(eeg_bank, video_ids, subjects=None):
    """
    Filters the EEGBank trials to include only the specified video IDs.
    Returns list of (sub, vid, row_in_sub).
    """
    subs = subjects or list(eeg_bank.train_data.keys())
    vid_set = set(video_ids)
    trial_table = []
    for s in subs:
        if s not in eeg_bank.train_ids:
            continue
        for row, vid in enumerate(eeg_bank.train_ids[s]):
            if vid in vid_set:
                trial_table.append((s, int(vid), row))
    return trial_table
