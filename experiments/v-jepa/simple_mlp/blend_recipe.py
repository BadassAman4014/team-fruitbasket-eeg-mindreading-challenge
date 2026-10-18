"""Honest DINOv2 <-> V-JEPA2 score blend (same backbone, same recipe, different video).

Consumes the tagged artifacts that train_recipe.py saves for each run:
    recipe_<tag>_sub-XX_similarity.npy      per-subject TEST score (102,102)
    recipe_<tag>_heldout_scores.npz         heldout ensemble scores + query/cand ids

Blends:  S = alpha * S_A + (1-alpha) * S_B   (raw cosine scores; same scale),
with alpha chosen on the HELD-OUT 901-1000 split (NOT dev), then CSLS applied once.

Because DINOv2 (appearance) and V-JEPA2 (motion) share backbone/loss/audio/protocol
and differ only in the video encoder, their errors are partly orthogonal — the blend
promotes near-miss trials. alpha is the one honest knob.

Usage
-----
    python simple_mlp/blend_recipe.py --tag_a dinov2_frames --tag_b vjepa2_frames \
        --device cpu --anti_hub csls
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_ranks, retrieval_metrics, apply_anti_hubness
from simple_mlp.train import TEST_IDS, EVAL_SUBJECTS, PRED_SUBJECTS, avg_metrics_over_subjects


def parse_args():
    p = argparse.ArgumentParser(description="DINOv2<->V-JEPA2 honest score blend")
    p.add_argument("--sub_dir", type=str, default="./submissions")
    p.add_argument("--tag_a", type=str, required=True, help="e.g. dinov2_frames")
    p.add_argument("--tag_b", type=str, required=True, help="e.g. vjepa2_frames")
    p.add_argument("--gt_dir", type=str, default="./tst_stim")
    p.add_argument("--anti_hub", type=str, default="csls", choices=["none", "csls", "mean"])
    p.add_argument("--csls_k", type=int, default=10)
    p.add_argument("--zstandardize", action="store_true",
                   help="z-standardize each model's score matrix before blending "
                        "(default off: both are cosine on the same scale).")
    return p.parse_args()


def zstd(S):
    return (S - S.mean()) / (S.std() + 1e-8)


def ranks_from_scores(S, cand_ids, query_ids):
    order = np.argsort(-S, axis=1)
    ranked = cand_ids[order]
    return (ranked == query_ids[:, None]).argmax(axis=1) + 1


def _find_file(sub_dir: str, fname: str) -> str:
    p = os.path.join(sub_dir, fname)
    if os.path.exists(p):
        return p
    here = Path(__file__).resolve().parent
    vjepa_root = here.parent
    repo_root = vjepa_root.parent.parent
    candidates = [
        vjepa_root / "submissions" / fname,
        repo_root / "submissions" / fname,
        Path.cwd() / "submissions" / fname,
        Path.cwd() / "experiments" / "v-jepa" / "submissions" / fname,
    ]
    for c in candidates:
        if os.path.exists(str(c)):
            return str(c)
    return p


def load_test_sims(sub_dir, tag):
    out = {}
    for sub in EVAL_SUBJECTS + PRED_SUBJECTS:
        fname = f"recipe_{tag}_sub-{sub:02d}_similarity.npy"
        p = _find_file(sub_dir, fname)
        if os.path.exists(p):
            out[sub] = np.load(p)
    return out


def load_heldout(sub_dir, tag):
    fname = f"recipe_{tag}_heldout_scores.npz"
    p = _find_file(sub_dir, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p} — rerun train_recipe.py (--select_on heldout) for tag '{tag}'.")
    d = np.load(p)
    return d["scores"], d["query_ids"], d["cand_ids"]


def main():
    args = parse_args()
    print("=" * 70)
    print(f"  Honest score blend: A='{args.tag_a}'  B='{args.tag_b}'")
    print("=" * 70)

    simsA = load_test_sims(args.sub_dir, args.tag_a)
    simsB = load_test_sims(args.sub_dir, args.tag_b)
    common = sorted(set(simsA) & set(simsB))
    print(f"  Subjects with both models: {common}")
    if not common:
        print("  No overlapping subjects — check tags / that both runs saved similarities.")
        return

    hoA, qA, cA = load_heldout(args.sub_dir, args.tag_a)
    hoB, qB, cB = load_heldout(args.sub_dir, args.tag_b)

    # Alignment sanity: heldout query/candidate ids must match between the two runs.
    if not (np.array_equal(qA, qB) and np.array_equal(cA, cB)):
        raise ValueError(
            "Heldout query/candidate IDs differ between the two runs — the models were "
            "not evaluated on the same split; blending would misalign. Re-run both with "
            "the same --val_lo/--val_hi and seeds."
        )
    if hoA.shape != hoB.shape:
        raise ValueError(f"Heldout score shapes differ: {hoA.shape} vs {hoB.shape}")

    pre = (lambda S: zstd(S)) if args.zstandardize else (lambda S: S)
    hoA_p, hoB_p = pre(hoA), pre(hoB)

    # --- tune alpha (and CSLS lambda) on HELDOUT ---
    alpha_grid = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    lam_grid = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3] if args.anti_hub != "none" else [0.0]
    best = (-1.0, 1.0, 0.0)  # (heldout_top1, alpha, lam)
    print("\n  alpha sweep on heldout (top-1):")
    for a in alpha_grid:
        S = a * hoA_p + (1 - a) * hoB_p
        best_a_t1 = -1.0; best_a_lam = 0.0
        for lam in lam_grid:
            adj = apply_anti_hubness(S, method=args.anti_hub, k=args.csls_k, hub_penalty=lam) if lam > 0 else S
            t1 = float((ranks_from_scores(adj, cA, qA) == 1).mean())
            if t1 > best_a_t1:
                best_a_t1, best_a_lam = t1, lam
        marker = ""
        if best_a_t1 > best[0]:
            best = (best_a_t1, a, best_a_lam); marker = "  [*]"
        print(f"    alpha={a:.2f} (A wt) | heldout top-1 {best_a_t1*100:5.2f}%  (λ={best_a_lam}){marker}")
    _, alpha, lam = best
    print(f"\n  → chosen alpha={alpha:.2f} (A={args.tag_a} {alpha*100:.0f}% / "
          f"B={args.tag_b} {(1-alpha)*100:.0f}%), CSLS λ={lam}  "
          f"[heldout top-1 {best[0]*100:.2f}%]")

    # --- blend TEST matrices ---
    final: Dict[int, np.ndarray] = {}
    for sub in common:
        S = alpha * pre(simsA[sub]) + (1 - alpha) * pre(simsB[sub])
        if lam > 0:
            S = apply_anti_hubness(S, method=args.anti_hub, k=args.csls_k, hub_penalty=lam)
        final[sub] = S

    # --- report on sub-01..04 ---
    print("\n" + "-" * 70)
    print(f"  Blend TEST estimate sub-01..04 (alpha={alpha:.2f}, CSLS λ={lam})")
    print("-" * 70)
    print(f"  {'Subject':<9} | {'Top-1':<16} | {'Top-5':<16} | Median")
    print("-" * 70)
    evals = [s for s in EVAL_SUBJECTS if s in final]
    for sub in evals:
        m = retrieval_metrics(compute_ranks(final[sub]))
        print(f"  sub-{sub:02d}    | {m['top1']*100:5.2f}% ({m['top1']*102:2.0f}/102)  | "
              f"{m['top5']*100:5.2f}% ({m['top5']*102:2.0f}/102)  | {m['median_rank']:.1f}")
    if evals:
        a1, a5, am = avg_metrics_over_subjects(final, evals)
        print("-" * 70)
        print(f"  {'AVERAGE':<9} | {a1*100:5.2f}% ({a1*408:3.0f}/408)  | {a5*100:5.2f}% ({a5*408:3.0f}/408)  | {am:.1f}")
        print(f"  {'CHANCE':<9} |  0.98% ( 4/408)  |  4.90% (20/408)  |  51.5")

    # --- blended submissions ---
    vids = np.array(TEST_IDS)
    print("\n  Blended submissions:")
    for sub in PRED_SUBJECTS:
        if sub not in final:
            continue
        ranked = vids[np.argsort(-final[sub], axis=1)]
        assert ranked.shape == (102, 102)
        assert ranked.min() == 1001 and ranked.max() == 1102
        for i in range(102):
            assert len(np.unique(ranked[i])) == 102
        out = os.path.join(args.sub_dir, f"blend_{args.tag_a}_{args.tag_b}_sub-{sub:02d}_ranked_candidate_list.npy")
        np.save(out, ranked)
        print(f"    ✓ sub-{sub:02d}: {out}")

    print("\n[blend] done.")


if __name__ == "__main__":
    main()
