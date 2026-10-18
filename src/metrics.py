"""
src/metrics.py - Retrieval Evaluation Metrics and Hubness Calibration
=====================================================================
Computes Top-1, Top-5, MRR, Median Rank, and CSLS / Mean-Centering debiasing.
"""

import numpy as np


def compute_csls(S, k=10):
    """
    Cross-Domain Similarity Local Scaling (CSLS):
    Penalizes candidates with dense neighborhoods to mitigate hubness.
    CSLS(i, j) = 2 * S(i, j) - r_T(i) - r_C(j)
    """
    n_queries, n_cands = S.shape
    k_val = min(k, n_queries, n_cands)
    # Average similarity to k nearest neighbors for queries and candidates
    r_T = np.sort(S, axis=1)[:, -k_val:].mean(axis=1, keepdims=True)
    r_C = np.sort(S, axis=0)[-k_val:, :].mean(axis=0, keepdims=True)
    return 2.0 * S - r_T - r_C


def compute_mean_center(S):
    """Candidate mean centering: S_adj = S - mean_cands."""
    return S - np.mean(S, axis=0, keepdims=True)


def ranked_from_scores(S, candidate_ids):
    """Returns candidate ranking (N_queries, N_candidates) ordered best-first."""
    order = np.argsort(-S, axis=1)
    return np.array(candidate_ids)[order]


def evaluate_ranking(ranked, gt):
    """
    Evaluates candidate ranking against ground truth video IDs.
    Returns dict: top1, top5, mrr, median_rank, mean_rank.
    """
    n_trials = len(gt)
    top1 = (ranked[:, 0] == gt).mean()
    top5 = (ranked[:, :5] == gt[:, None]).any(axis=1).mean()

    ranks = []
    reciprocal_ranks = []
    for i in range(n_trials):
        match = np.where(ranked[i] == gt[i])[0]
        if len(match) > 0:
            rank = match[0] + 1
        else:
            rank = ranked.shape[1]
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)

    return {
        'top1': float(top1),
        'top5': float(top5),
        'mrr': float(np.mean(reciprocal_ranks)),
        'median_rank': float(np.median(ranks)),
        'mean_rank': float(np.mean(ranks)),
        'ranks': np.array(ranks),
    }


def true_ranks(S, candidate_ids, gt):
    """Returns 1-based ranks of the true stimulus for each query."""
    ranked = ranked_from_scores(S, candidate_ids)
    ranks = []
    for i, g in enumerate(gt):
        match = np.where(ranked[i] == g)[0]
        ranks.append(match[0] + 1 if len(match) > 0 else ranked.shape[1])
    return np.array(ranks)


def summarize_ranks(ranks):
    """Summarizes rank array into top1, top5, mrr, and median."""
    return {
        'top1': float((ranks == 1).mean()),
        'top5': float((ranks <= 5).mean()),
        'mrr': float(np.mean(1.0 / ranks)),
        'median': float(np.median(ranks)),
    }
