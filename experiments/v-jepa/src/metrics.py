"""Ranking evaluation metrics for EEG video-retrieval."""

import numpy as np


def compute_ranks(S: np.ndarray) -> np.ndarray:
    """Get per-trial rank of the correct candidate from a similarity matrix.

    Assumes a *square* matrix where row i corresponds to trial i and the
    correct candidate for trial i is column i (diagonal = correct match).

    Parameters
    ----------
    S : (N, N) similarity matrix

    Returns
    -------
    ranks : (N,) int array — 1-indexed rank of the ground-truth candidate.
    """
    N   = S.shape[0]
    idx = np.arange(N)
    return (-S).argsort(axis=1).argsort(axis=1)[idx, idx] + 1


def retrieval_metrics(ranks: np.ndarray) -> dict:
    """Aggregate top-k accuracy and median rank.

    Parameters
    ----------
    ranks : (N,) int array

    Returns
    -------
    dict with keys: top1, top5, median_rank, chance_top1, chance_top5
    """
    N = len(ranks)
    return {
        "top1":         float((ranks == 1).mean()),
        "top5":         float((ranks <= 5).mean()),
        "median_rank":  float(np.median(ranks)),
        "chance_top1":  1.0 / N,
        "chance_top5":  5.0 / N,
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    tag = f"[{label}]  " if label else ""
    print(
        f"{tag}"
        f"top-1: {metrics['top1']:.4f} (chance {metrics['chance_top1']:.4f})  |  "
        f"top-5: {metrics['top5']:.4f} (chance {metrics['chance_top5']:.4f})  |  "
        f"median rank: {metrics['median_rank']:.1f}"
    )


def apply_anti_hubness(
    S: np.ndarray,
    method: str = "csls",
    k: int = 10,
    hub_penalty: float = 1.0,
) -> np.ndarray:
    """Adjust similarity matrix S (N_queries, N_candidates) along candidates to mitigate hubness.

    Hubness is a candidate-side phenomenon where certain candidate items act as universal
    nearest neighbors ("hubs") across many queries. Penalizing candidates based on their
    global or top-k neighborhood similarity directly reduces their dominance and improves retrieval.

    Parameters
    ----------
    S : (N, M) ndarray
        Similarity matrix (e.g. 102 queries x 102 candidates).
    method : str, default="csls"
        "csls" : Cross-Domain Similarity Local Scaling (subtracts candidate's top-k query mean).
        "mean" : Subtracts candidate's global mean similarity across all queries.
    k : int, default=10
        Neighborhood size for CSLS.
    hub_penalty : float, default=1.0
        Scaling factor for the hub penalty.

    Returns
    -------
    S_adj : (N, M) ndarray
        Candidate-adjusted similarity matrix.
    """
    if method == "csls":
        k_val = min(k, S.shape[0])
        # Sort queries along column axis in ascending order; average the top-k highest matches
        topk_queries = np.sort(S, axis=0)[-k_val:, :]  # (k_val, M)
        r_cand = topk_queries.mean(axis=0, keepdims=True)  # (1, M)
        return S - hub_penalty * r_cand
    elif method == "mean":
        cand_mean = S.mean(axis=0, keepdims=True)  # (1, M)
        return S - hub_penalty * cand_mean
    else:
        return S

