"""Honest ensemble: simple-MLP (contrastive) + Phase 1D Ridge (linear encoding).

The last remaining lever. The two models make DIFFERENT errors — the MLP works on
a unit hypersphere via InfoNCE, the Ridge model predicts EEG voltages linearly and
ranks by Pearson correlation — so averaging their similarity matrices can promote
near-miss (rank 2-5) trials into rank 1. The diagnostics showed a large rank-2-5
mass, which is exactly what a complementary model can convert.

HONESTY: the blend weight alpha is tuned on the HELD-OUT training stimuli (901-1000),
NOT on sub-01..04 test. The Ridge model is trained on the SAME split as the MLP
(stimuli 1-900 for selection; 1-1000 for the final full fit). sub-01..04 test stays
an untouched generalization estimate.

    S_blend = alpha * z(S_MLP) + (1 - alpha) * z(S_Ridge)        [z = z-standardize]

This reuses the simple-MLP checkpoints produced by train.py (eeg_repr=both) and the
Phase 1D EncodingModel from src/pipeline.py.

Usage (cluster)
---------------
    # 1) First train the simple-MLP 'both' models (selection fit on 1-900):
    python simple_mlp/train.py --device cuda --epochs 30 --seeds 42,123,999 \
        --select_on heldout --anti_hub csls --eeg_repr both
    # 2) Then blend:
    python simple_mlp/blend.py --device cuda --seeds 42,123,999
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_train_eeg, load_test_eeg, decimate_eeg, normalize_eeg_trials
from src.dataset import load_eval_test_data, load_val_retrieval_data
from src.pipeline import EncodingModel
from src.metrics import compute_ranks, retrieval_metrics, apply_anti_hubness

from simple_mlp.model import SimpleContrastiveModel
from simple_mlp.eeg_features import build_eeg_representation
from simple_mlp.train import (
    TEST_IDS, EVAL_SUBJECTS, PRED_SUBJECTS, ALL_TRAIN_SUBJECTS,
    apply_eeg_repr, zstandardize, heldout_metrics_from_sim,
    preprocess_test_eeg, avg_metrics_over_subjects,
)

VICLIP_DIM = 768


def parse_args():
    p = argparse.ArgumentParser(description="Simple-MLP + Phase 1D Ridge honest blend")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=str, default="42,123,999")
    p.add_argument("--eeg_repr", type=str, default="both", choices=["raw", "bandpower", "both", "rich", "both_rich"])
    p.add_argument("--window_start", type=int, default=5)
    p.add_argument("--window_end", type=int, default=60)
    p.add_argument("--norm_mode", type=str, default="trial", choices=["trial", "subject"])
    p.add_argument("--val_lo", type=int, default=901)
    p.add_argument("--val_hi", type=int, default=1000)

    # Model dims must match the checkpoints produced by train.py.
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--init_temp", type=float, default=0.07)

    p.add_argument("--feat_cache", type=str, default="./cache/viclip_clap")
    p.add_argument("--feat_prefix", type=str, default="viclip_clap")
    p.add_argument("--trn_dir", type=str, default="./trn")
    p.add_argument("--tst_dir", type=str, default="./tst")
    p.add_argument("--gt_dir", type=str, default="./tst_stim")
    p.add_argument("--model_dir", type=str, default="./models/simple_mlp")
    p.add_argument("--ridge_dir", type=str, default="./models/blend_ridge")
    p.add_argument("--sub_dir", type=str, default="./submissions")

    p.add_argument("--anti_hub", type=str, default="csls", choices=["none", "csls", "mean"])
    p.add_argument("--csls_k", type=int, default=10)
    return p.parse_args()


# ─────────────────────────── simple-MLP similarities ───────────────────────────

def load_mlp_model(ckpt_path: str, args, eeg_dim: int, stim_dim: int) -> SimpleContrastiveModel:
    ckpt = torch.load(ckpt_path, map_location=args.device)
    m = SimpleContrastiveModel(
        eeg_dim=eeg_dim, stim_dim=stim_dim, proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim, dropout=args.dropout, init_temp=args.init_temp,
    ).to(args.device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@torch.no_grad()
def mlp_similarity(model, eeg_feat_t, stim_feats_t) -> np.ndarray:
    return model.compute_similarity(eeg_feat_t.to(stim_feats_t.device), stim_feats_t).cpu().numpy()


def ensemble_mlp_sims(ckpts, args, eeg_dim, stim_dim, eeg_by_sub, stim_feats_t) -> Dict[int, np.ndarray]:
    """z-standardized seed-average of MLP similarity, per subject."""
    acc: Dict[int, List[np.ndarray]] = {s: [] for s in eeg_by_sub}
    for cp in ckpts:
        m = load_mlp_model(cp, args, eeg_dim, stim_dim)
        for sub, feat in eeg_by_sub.items():
            acc[sub].append(zstandardize(mlp_similarity(m, feat, stim_feats_t)))
    return {sub: np.mean(mats, axis=0) for sub, mats in acc.items()}


# ─────────────────────────── Phase 1D Ridge similarities ───────────────────────

def fit_ridge_for_subject(sub, train_feats_full, args, stim_cap):
    """Fit a Phase 1D Ridge model on this subject's training trials with stim id <= stim_cap.

    Returns a fitted EncodingModel (features -> decimated 300-sample EEG).
    """
    npz = os.path.join(args.trn_dir, f"sub-{sub:02d}_trn.npz")
    ids, raw = load_train_eeg(npz)
    dec = decimate_eeg(raw)                 # (N, 17, 300)
    dec = normalize_eeg_trials(dec)         # trial z-score (matches similarity path)
    mask = ids <= stim_cap
    model = EncodingModel()
    model.fit(train_feats_full[ids[mask]], dec[mask])
    return model


def ridge_test_similarity(model, test_eeg_full, test_feats_full) -> np.ndarray:
    """Ridge similarity for the 102 test trials vs 102 test candidates."""
    return model.similarity_matrix(test_eeg_full, test_feats_full)


# ─────────────────────────────────── main ──────────────────────────────────────

def main():
    args = parse_args()
    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    window = (args.window_start, args.window_end)
    os.makedirs(args.ridge_dir, exist_ok=True)
    os.makedirs(args.sub_dir, exist_ok=True)

    print("=" * 70)
    print("  Honest Blend — simple-MLP (both) + Phase 1D Ridge")
    print(f"  Seeds: {seed_list}  |  eeg_repr(MLP): {args.eeg_repr}  |  Device: {args.device}")
    print(f"  Alpha tuned on HELD-OUT stimuli {args.val_lo}-{args.val_hi}")
    print("=" * 70)

    # Stimulus features (full) — MLP uses full fused dim; Ridge uses the same.
    tr_feats = np.load(os.path.join(args.feat_cache, f"{args.feat_prefix}_train_all.npy"))
    te_feats = np.load(os.path.join(args.feat_cache, f"{args.feat_prefix}_test.npy"))
    stim_dim = tr_feats.shape[1]
    te_feats_t = torch.as_tensor(te_feats, dtype=torch.float32, device=args.device)

    # Checkpoints from the selection fit (train.py saved these on 1-900).
    ckpts = [os.path.join(args.model_dir, f"simple_mlp_{args.feat_prefix}_seed{s}_best.pt") for s in seed_list]
    for cp in ckpts:
        if not os.path.exists(cp):
            raise FileNotFoundError(f"Missing MLP checkpoint {cp}. Run train.py (eeg_repr={args.eeg_repr}) first.")

    # ---- EEG, both representations: windowed-feature for MLP, full-300 for Ridge ----
    # Eval subjects (label-sorted so diagonal = correct).
    mlp_eeg: Dict[int, torch.Tensor] = {}
    ridge_eeg_full: Dict[int, np.ndarray] = {}
    for sub in EVAL_SUBJECTS:
        win_eeg, _ = load_eval_test_data(sub, test_dir=args.tst_dir, gt_dir=args.gt_dir,
                                         eeg_window=window, norm_mode=args.norm_mode)
        full_eeg, _ = load_eval_test_data(sub, test_dir=args.tst_dir, gt_dir=args.gt_dir,
                                          eeg_window=None, norm_mode=args.norm_mode)
        mlp_eeg[sub] = apply_eeg_repr(win_eeg, args.eeg_repr)
        ridge_eeg_full[sub] = full_eeg
    for sub in PRED_SUBJECTS:
        tp = os.path.join(args.tst_dir, f"sub-{sub:02d}_tst.npy")
        if not os.path.exists(tp):
            continue
        raw = load_test_eeg(tp)
        mlp_eeg[sub] = apply_eeg_repr(preprocess_test_eeg(raw, window, args.norm_mode), args.eeg_repr)
        ridge_eeg_full[sub] = preprocess_test_eeg(raw, None, args.norm_mode)

    eeg_dim = int(mlp_eeg[EVAL_SUBJECTS[0]].shape[1])

    # ---- Held-out validation pack (for honest alpha selection) ----
    pack = load_val_retrieval_data(subjects=ALL_TRAIN_SUBJECTS, stim_feats=tr_feats,
                                   trn_dir=args.trn_dir, val_range=(args.val_lo, args.val_hi),
                                   eeg_window=window, norm_mode=args.norm_mode)
    ho_mlp_eeg = apply_eeg_repr(pack["val_eeg"], args.eeg_repr)
    ho_stims = pack["val_stims"]
    ho_cand_ids = pack["cand_ids"]
    ho_cand_feats_t = torch.as_tensor(tr_feats[ho_cand_ids], dtype=torch.float32, device=args.device)
    # Ridge heldout needs full-300 EEG for the same val trials:
    pack_full = load_val_retrieval_data(subjects=ALL_TRAIN_SUBJECTS, stim_feats=tr_feats,
                                        trn_dir=args.trn_dir, val_range=(args.val_lo, args.val_hi),
                                        eeg_window=None, norm_mode=args.norm_mode)
    ho_ridge_eeg_full = pack_full["val_eeg"].numpy()

    # ---- MLP similarities (seed-ensemble, z-standardized) ----
    print("\n[1/3] Simple-MLP similarities (seed ensemble) ...")
    mlp_test_sims = ensemble_mlp_sims(ckpts, args, eeg_dim, stim_dim, mlp_eeg, te_feats_t)
    # MLP heldout similarity (ensemble, z-standardized)
    ho_mlp_mats = []
    for cp in ckpts:
        m = load_mlp_model(cp, args, eeg_dim, stim_dim)
        ho_mlp_mats.append(zstandardize(mlp_similarity(m, ho_mlp_eeg, ho_cand_feats_t)))
    ho_mlp_sim = np.mean(ho_mlp_mats, axis=0)

    # ---- Ridge similarities (per subject, trained on 1-900 for honest alpha) ----
    print("\n[2/3] Phase 1D Ridge similarities (trained on stimuli 1-900) ...")
    ridge_test_sims: Dict[int, np.ndarray] = {}
    for sub in EVAL_SUBJECTS + [s for s in PRED_SUBJECTS if s in ridge_eeg_full]:
        rm = fit_ridge_for_subject(sub, tr_feats, args, stim_cap=args.val_lo - 1)
        ridge_test_sims[sub] = zstandardize(rm.similarity_matrix(ridge_eeg_full[sub], te_feats))

    # Ridge heldout similarity: one model is enough (features->EEG is subject-specific,
    # but the heldout val pack mixes subjects). We fit a pooled Ridge on 1-900 across
    # all subjects for the alpha-selection signal only.
    ids_all, eeg_all = [], []
    for sub in ALL_TRAIN_SUBJECTS:
        npz = os.path.join(args.trn_dir, f"sub-{sub:02d}_trn.npz")
        if not os.path.exists(npz):
            continue
        ids, raw = load_train_eeg(npz)
        dec = normalize_eeg_trials(decimate_eeg(raw))
        m = ids <= (args.val_lo - 1)
        ids_all.append(ids[m]); eeg_all.append(dec[m])
    pooled = EncodingModel()
    pooled.fit(tr_feats[np.concatenate(ids_all)], np.concatenate(eeg_all, axis=0))
    ho_ridge_sim = zstandardize(pooled.similarity_matrix(ho_ridge_eeg_full, tr_feats[ho_cand_ids]))

    # ---- Honest alpha sweep on heldout ----
    print("\n[3/3] Tuning alpha on held-out stimuli ...")
    alpha_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_alpha, best_ho_t1 = 1.0, -1.0
    for a in alpha_grid:
        S = a * ho_mlp_sim + (1 - a) * ho_ridge_sim
        if args.anti_hub != "none":
            S = apply_anti_hubness(S, method=args.anti_hub, k=args.csls_k, hub_penalty=0.1)
        t1, t5 = heldout_metrics_from_sim(S, ho_stims, ho_cand_ids)
        tag = " [*]" if t1 > best_ho_t1 else ""
        print(f"    alpha={a:.2f} (MLP wt) | heldout top-1 {t1*100:5.2f}% / top-5 {t5*100:5.2f}%{tag}")
        if t1 > best_ho_t1:
            best_ho_t1, best_alpha = t1, a
    print(f"  → chosen alpha={best_alpha:.2f} (MLP {best_alpha*100:.0f}% / Ridge {(1-best_alpha)*100:.0f}%)")

    # ---- Blend test matrices and report ----
    final_sims: Dict[int, np.ndarray] = {}
    for sub in mlp_test_sims:
        if sub not in ridge_test_sims:
            final_sims[sub] = mlp_test_sims[sub]
            continue
        S = best_alpha * mlp_test_sims[sub] + (1 - best_alpha) * ridge_test_sims[sub]
        if args.anti_hub != "none":
            S = apply_anti_hubness(S, method=args.anti_hub, k=args.csls_k, hub_penalty=0.1)
        final_sims[sub] = S

    print("\n" + "-" * 70)
    print(f"  Blend TEST estimate on sub-01..04 (alpha={best_alpha:.2f}, {args.anti_hub})")
    print("-" * 70)
    print(f"  {'Subject':<9} | {'Top-1':<16} | {'Top-5':<16} | {'Median Rank'}")
    print("-" * 70)
    for sub in EVAL_SUBJECTS:
        m = retrieval_metrics(compute_ranks(final_sims[sub]))
        print(f"  sub-{sub:02d}    | {m['top1']*100:5.2f}% ({m['top1']*102:2.0f}/102)  | "
              f"{m['top5']*100:5.2f}% ({m['top5']*102:2.0f}/102)  | {m['median_rank']:.1f}")
    at1, at5, amed = avg_metrics_over_subjects(final_sims, EVAL_SUBJECTS)
    print("-" * 70)
    print(f"  {'AVERAGE':<9} | {at1*100:5.2f}% ({at1*408:3.0f}/408)  | "
          f"{at5*100:5.2f}% ({at5*408:3.0f}/408)  | {amed:.1f}")
    print(f"  {'CHANCE':<9} |  0.98% ( 4/408)  |  4.90% (20/408)  |  51.5")

    # ---- Save similarity matrices + blended submissions ----
    for sub, S in final_sims.items():
        np.save(os.path.join(args.sub_dir, f"blend_sub-{sub:02d}_similarity.npy"), S)
    video_ids = np.array(TEST_IDS)
    print("\n  Generating blended submissions for sub-05 / sub-06 ...")
    for sub in PRED_SUBJECTS:
        if sub not in final_sims:
            continue
        ranked = video_ids[np.argsort(-final_sims[sub], axis=1)]
        assert ranked.shape == (102, 102)
        assert ranked.min() == 1001 and ranked.max() == 1102
        for i in range(102):
            assert len(np.unique(ranked[i])) == 102
        out = os.path.join(args.sub_dir, f"blend_sub-{sub:02d}_ranked_candidate_list.npy")
        np.save(out, ranked)
        print(f"    ✓ sub-{sub:02d}: {out}")

    print("\nBlend complete.")


if __name__ == "__main__":
    main()
