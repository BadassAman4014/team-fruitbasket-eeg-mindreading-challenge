"""exp4-faithful recipe on V-JEPA2 (8 sub-clips) + CLAP, concat fusion.

Loads frame-level video (N,8,1024) and separate audio (N,512) caches, pools
video inside the model (DinoPool mean), fuses by CONCAT, trains with
multipositive InfoNCE + modality dropout, selects epoch + CSLS lambda on the
held-out 901-1000 split, raw-score seed-averages, optional --final_full_fit.

Per-feature z-score (fit on train stimuli) is applied to the pooled-video-space
and audio independently: we normalize each sub-clip-feature column and each audio
column using stats computed over the training stimuli (frame-mean for video, to
match exp4's fit on mean-over-frames).

Usage (needs the frame-level caches from experiments/prep_vjepa2_clap.py)
-------------------------------------------------------------------------
    # Exact Component-A params:
    python simple_mlp/train_recipe.py --device cuda --seeds 42,123,999 \
        --select_on heldout --anti_hub csls --final_full_fit

    # Auto-tune:
    python simple_mlp/train_recipe.py --device cuda --seeds 42,123,999 \
        --select_on heldout --anti_hub csls --final_full_fit \
        --dropout 0.4 --modality_dropout 0.2 --weight_decay 0.005
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_train_eeg, load_test_eeg, decimate_eeg, normalize_eeg_trials, zscore_per_subject
from src.dataset import load_eval_test_data
from src.metrics import compute_ranks, retrieval_metrics, apply_anti_hubness

from simple_mlp.model_recipe import RecipeContrastiveModel
from simple_mlp.train import (
    TEST_IDS, EVAL_SUBJECTS, PRED_SUBJECTS, ALL_TRAIN_SUBJECTS,
    set_seed, apply_eeg_repr, preprocess_test_eeg,
    avg_metrics_over_subjects, heldout_metrics_from_sim,
)


def parse_args():
    p = argparse.ArgumentParser(description="exp4-faithful recipe (V-JEPA2 frames + CLAP, concat)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=str, default="42,123,999")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--init_temp", type=float, default=0.07)
    p.add_argument("--modality_dropout", type=float, default=0.15)

    p.add_argument("--window_start", type=int, default=5)
    p.add_argument("--window_end", type=int, default=60)
    p.add_argument("--no_window", action="store_true")
    p.add_argument("--norm_mode", type=str, default="trial", choices=["trial", "subject"])
    p.add_argument("--eeg_repr", type=str, default="both",
                   choices=["raw", "bandpower", "both", "rich", "both_rich"])

    p.add_argument("--select_on", type=str, default="heldout", choices=["dev", "heldout"])
    p.add_argument("--val_lo", type=int, default=901)
    p.add_argument("--val_hi", type=int, default=1000)

    p.add_argument("--feat_cache", type=str, default="./cache/vjepa2_frames")
    p.add_argument("--video_prefix", type=str, default="vjepa2_video")
    p.add_argument("--audio_prefix", type=str, default="clap_audio")

    p.add_argument("--trn_dir", type=str, default="./trn")
    p.add_argument("--tst_dir", type=str, default="./tst")
    p.add_argument("--gt_dir", type=str, default="./tst_stim")
    p.add_argument("--model_dir", type=str, default="./models/recipe_mlp")
    p.add_argument("--sub_dir", type=str, default="./submissions")

    p.add_argument("--anti_hub", type=str, default="csls", choices=["none", "csls", "mean"])
    p.add_argument("--csls_k", type=int, default=10)
    p.add_argument("--hub_penalty", type=float, default=0.0)
    p.add_argument("--final_full_fit", action="store_true")
    p.add_argument("--tag", type=str, default=None,
                   help="Label for saved matrices/checkpoints (e.g. 'dinov2' or 'vjepa2'). "
                        "Defaults to the feat_cache dir name. Keeps DINO/VJEPA outputs separate for blending.")
    return p.parse_args()


# ───────────────────────────── EEG loading ─────────────────────────────

def load_subject_eeg_by_range(sub, trn_dir, lo, hi, window, norm_mode, eeg_repr):
    """Return (eeg_feat (N,D), stim_ids (N,), sub_ids (N,)) for one subject, ids in [lo,hi]."""
    ids, raw = load_train_eeg(os.path.join(trn_dir, f"sub-{sub:02d}_trn.npz"))
    dec = decimate_eeg(raw)
    if window is not None:
        dec = dec[..., window[0]:window[1]]
    dec = normalize_eeg_trials(dec) if norm_mode == "trial" else zscore_per_subject(dec)
    mask = (ids >= lo) & (ids <= hi)
    feat = apply_eeg_repr(dec[mask], eeg_repr).numpy()
    sids = ids[mask]
    subj = np.full(len(sids), sub, dtype=np.int64)
    return feat, sids, subj


# ───────────────────────────── stimulus indexing ─────────────────────────────

def index_video(video_bank: np.ndarray, stim_ids: np.ndarray) -> np.ndarray:
    """video_bank: (1001, 8, D) 1-based ID index → (N, 8, D) for each stim id."""
    return video_bank[stim_ids].astype(np.float32)


def index_audio(audio_bank: np.ndarray, stim_ids: np.ndarray) -> np.ndarray:
    return audio_bank[stim_ids].astype(np.float32)


# ───────────────────────────── scoring ─────────────────────────────

@torch.no_grad()
def score_matrix(model, eeg_feat, cand_video, cand_audio, device, bs=256) -> np.ndarray:
    """(N_eeg, N_cand) raw cosine. eeg_feat (N,D); cand_video (M,8,Dv); cand_audio (M,Da)."""
    model.eval()
    e = torch.as_tensor(eeg_feat, dtype=torch.float32, device=device)
    v = torch.as_tensor(cand_video, dtype=torch.float32, device=device)
    a = torch.as_tensor(cand_audio, dtype=torch.float32, device=device)
    z_eeg = []
    for s in range(0, len(e), bs):
        z_eeg.append(model.encode_eeg(e[s:s+bs]))
    z_eeg = torch.cat(z_eeg, 0)
    z_stim = model.encode_stim(v, a)
    return (z_eeg @ z_stim.T).cpu().numpy()


def ranks_from_scores(S, cand_ids, query_ids):
    order = np.argsort(-S, axis=1)
    ranked = cand_ids[order]
    hit = ranked == query_ids[:, None]
    return hit.argmax(axis=1) + 1


# ───────────────────────────── training ─────────────────────────────

def build_model(args, eeg_dim, video_dim, audio_dim, n_frames):
    return RecipeContrastiveModel(
        eeg_dim=eeg_dim, video_dim=video_dim, audio_dim=audio_dim, n_frames=n_frames,
        proj_dim=args.proj_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        init_temp=args.init_temp, modality_dropout=args.modality_dropout,
    ).to(args.device)


def train_one_seed(seed, args, train_pack, heldout_pack, dims, fixed_epochs, ckpt_suffix):
    set_seed(seed)
    eeg_dim, video_dim, audio_dim, n_frames = dims
    final_fit = fixed_epochs is not None
    n_epochs = fixed_epochs if final_fit else args.epochs

    print("\n" + "=" * 70)
    print(f"  [recipe] seed {seed} | {'FULL-FIT' if final_fit else args.select_on} | epochs={n_epochs}")
    print("=" * 70)

    eeg, vids, auds, sids = train_pack  # eeg (N,D), vids (N,8,Dv), auds (N,Da), sids (N,)
    ds = TensorDataset(torch.from_numpy(eeg), torch.from_numpy(vids),
                       torch.from_numpy(auds), torch.from_numpy(sids))
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=(args.device == "cuda"), generator=g)

    model = build_model(args, eeg_dim, video_dim, audio_dim, n_frames)
    opt = torch.optim.AdamW(model.param_groups(args.weight_decay), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    ckpt = os.path.join(args.model_dir, f"recipe_{args.tag}_seed{seed}_{ckpt_suffix}.pt")
    best_sel, best_epoch = -1.0, 0

    def _save(ep):
        torch.save({"epoch": ep, "seed": seed, "model_state": model.state_dict(),
                    "dims": dims, "args": vars(args)}, ckpt)

    for epoch in range(1, n_epochs + 1):
        model.train(); tot, nb = 0.0, 0
        for e_b, v_b, a_b, s_b in loader:
            e_b = e_b.to(args.device); v_b = v_b.to(args.device)
            a_b = a_b.to(args.device); s_b = s_b.to(args.device)
            opt.zero_grad()
            loss, _ = model(e_b, v_b, a_b, stim_ids=s_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); nb += 1
        sched.step(); eloss = tot / max(nb, 1)

        if final_fit:
            if epoch % 5 == 0 or epoch == n_epochs:
                print(f"  seed {seed} | ep {epoch:02d}/{n_epochs:02d} | loss {eloss:.4f} (full-fit)")
            best_epoch = epoch; _save(epoch); continue

        # Heldout selection signal.
        he, hv, ha, hqids, hcids = heldout_pack
        S = score_matrix(model, he, hv, ha, args.device)
        r = ranks_from_scores(S, hcids, hqids)
        sel_t1 = float((r == 1).mean()); sel_t5 = float((r <= 5).mean())
        is_best = sel_t1 > best_sel
        if epoch % 5 == 0 or is_best or epoch == n_epochs:
            print(f"  seed {seed} | ep {epoch:02d}/{n_epochs:02d} | loss {eloss:.4f} | "
                  f"heldout top-1 {sel_t1*100:5.2f}% / top-5 {sel_t5*100:5.2f}%{'  [*]' if is_best else ''}")
        if is_best:
            best_sel, best_epoch = sel_t1, epoch; _save(epoch)

    print(f"  --> seed {seed}: epoch {best_epoch:02d} ({'full-fit' if final_fit else f'heldout {best_sel*100:.2f}%'})")
    return ckpt, best_epoch


def load_model(ckpt, args):
    c = torch.load(ckpt, map_location=args.device)
    eeg_dim, video_dim, audio_dim, n_frames = c["dims"]
    m = build_model(args, eeg_dim, video_dim, audio_dim, n_frames)
    m.load_state_dict(c["model_state"]); m.eval()
    return m


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.model_dir, exist_ok=True); os.makedirs(args.sub_dir, exist_ok=True)
    window = None if args.no_window else (args.window_start, args.window_end)
    is_ens = len(seeds) > 1
    # Tag keeps DINOv2 and V-JEPA2 outputs separate (checkpoints, saved matrices).
    args.tag = args.tag or os.path.basename(os.path.normpath(args.feat_cache))

    # Seamless path resolution whether run from repo root or experiments/v-jepa
    here = Path(__file__).resolve().parent
    vjepa_root = here.parent
    repo_root = vjepa_root.parent.parent

    expected_v_file = f"{args.video_prefix}_train_all.npy"
    if not os.path.exists(os.path.join(args.feat_cache, expected_v_file)):
        candidates = [
            vjepa_root / "vjepa2_frames",
            vjepa_root / "vjepa2_clap",
            repo_root / "experiments" / "v-jepa" / "vjepa2_frames",
            repo_root / "cache" / os.path.basename(args.feat_cache),
            vjepa_root / "cache" / os.path.basename(args.feat_cache),
            Path.cwd() / "vjepa2_frames",
            Path.cwd() / "experiments" / "v-jepa" / "vjepa2_frames",
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(str(cand), expected_v_file)):
                print(f"  [Path] Auto-redirected feat_cache '{args.feat_cache}' -> '{cand}'")
                args.feat_cache = str(cand)
                break

    for dir_attr, sample_file, fallbacks in [
        ("trn_dir", "sub-01_trn.npz", [repo_root / "trn", vjepa_root / "trn", Path.cwd() / "trn", Path.cwd() / ".." / ".." / "trn"]),
        ("tst_dir", "sub-01_tst.npy", [repo_root / "tst", vjepa_root / "tst", Path.cwd() / "tst", Path.cwd() / ".." / ".." / "tst"]),
        ("gt_dir", "sub-01_tst_stim.npy", [repo_root / "tst_stim", vjepa_root / "tst_stim", Path.cwd() / "tst_stim", Path.cwd() / ".." / ".." / "tst_stim"]),
    ]:
        curr = getattr(args, dir_attr)
        if not os.path.exists(os.path.join(curr, sample_file)):
            for fb in fallbacks:
                if os.path.exists(os.path.join(str(fb), sample_file)):
                    print(f"  [Path] Auto-redirected {dir_attr} '{curr}' -> '{fb}'")
                    setattr(args, dir_attr, str(fb))
                    break

    print("=" * 70)
    print("  exp4-faithful RECIPE (V-JEPA2 8 sub-clips + CLAP, CONCAT fusion)")
    print(f"  Seeds {seeds} | wd={args.weight_decay} | mod_drop={args.modality_dropout} | dropout={args.dropout}")
    print(f"  Selection: {args.select_on} | full_fit: {args.final_full_fit} | Device: {args.device}")
    print("=" * 70)

    # Feature banks (1-based ID index).
    vb_trn = np.load(os.path.join(args.feat_cache, f"{args.video_prefix}_train_all.npy"))  # (1001,8,Dv)
    vb_tst = np.load(os.path.join(args.feat_cache, f"{args.video_prefix}_test.npy"))        # (102,8,Dv)
    ab_trn = np.load(os.path.join(args.feat_cache, f"{args.audio_prefix}_train_all.npy"))   # (1001,Da)
    ab_tst = np.load(os.path.join(args.feat_cache, f"{args.audio_prefix}_test.npy"))        # (102,Da)
    video_dim = vb_trn.shape[2]; audio_dim = ab_trn.shape[1]; n_frames = vb_trn.shape[1]
    print(f"  Video bank {vb_trn.shape}  Audio bank {ab_trn.shape}  (frames={n_frames}, Dv={video_dim}, Da={audio_dim})")

    # Per-feature z-score fit on TRAIN stimuli (1..fit_hi). Video stats fit on the
    # frame-MEAN (matches exp4 fit_normalizer on mean-over-frames), applied per sub-clip.
    fit_hi = (args.val_lo - 1) if args.select_on == "heldout" else 1000
    vmean = vb_trn[1:fit_hi+1].mean(axis=(0, 1), keepdims=True)   # (1,1,Dv)
    vstd = np.maximum(vb_trn[1:fit_hi+1].std(axis=(0, 1), keepdims=True), 1e-6)
    amean = ab_trn[1:fit_hi+1].mean(axis=0, keepdims=True)
    astd = np.maximum(ab_trn[1:fit_hi+1].std(axis=0, keepdims=True), 1e-6)
    vb_trn = ((vb_trn - vmean) / vstd).astype(np.float32)
    vb_tst = ((vb_tst - vmean) / vstd).astype(np.float32)
    ab_trn = ((ab_trn - amean) / astd).astype(np.float32)
    ab_tst = ((ab_tst - amean) / astd).astype(np.float32)
    print(f"  Per-feature z-score fit on train stimuli 1..{fit_hi}")

    # EEG for eval subjects (test, label-sorted) and candidate feats = test videos.
    eval_eeg: Dict[int, np.ndarray] = {}
    for sub in EVAL_SUBJECTS:
        se, _ = load_eval_test_data(sub, test_dir=args.tst_dir, gt_dir=args.gt_dir,
                                    eeg_window=window, norm_mode=args.norm_mode)
        eval_eeg[sub] = apply_eeg_repr(se, args.eeg_repr).numpy()
    eeg_dim = eval_eeg[EVAL_SUBJECTS[0]].shape[1]
    cand_video_test = vb_tst                    # (102,8,Dv) rows aligned to TEST_IDS
    cand_audio_test = ab_tst

    # Training pool (selection range).
    lo_hi = (1, args.val_lo - 1) if args.select_on == "heldout" else (1, 1000)
    e_list, s_list, su_list = [], [], []
    for sub in ALL_TRAIN_SUBJECTS:
        f, sids, subj = load_subject_eeg_by_range(sub, args.trn_dir, lo_hi[0], lo_hi[1],
                                                   window, args.norm_mode, args.eeg_repr)
        e_list.append(f); s_list.append(sids); su_list.append(subj)
    tr_eeg = np.concatenate(e_list); tr_sids = np.concatenate(s_list); tr_subj = np.concatenate(su_list)
    tr_video = index_video(vb_trn, tr_sids); tr_audio = index_audio(ab_trn, tr_sids)
    train_pack = (tr_eeg, tr_video, tr_audio, tr_sids.astype(np.int64))
    print(f"  Train pool: {len(tr_eeg)} trials (stimuli {lo_hi[0]}..{lo_hi[1]})")

    # Heldout validation pack (ids 901..1000), candidates = those stimuli.
    heldout_pack = None
    if args.select_on == "heldout":
        he_list, hs_list = [], []
        for sub in ALL_TRAIN_SUBJECTS:
            f, sids, _ = load_subject_eeg_by_range(sub, args.trn_dir, args.val_lo, args.val_hi,
                                                   window, args.norm_mode, args.eeg_repr)
            he_list.append(f); hs_list.append(sids)
        he = np.concatenate(he_list); hq = np.concatenate(hs_list)
        hcids = np.arange(args.val_lo, args.val_hi + 1)
        hv = index_video(vb_trn, hcids); ha = index_audio(ab_trn, hcids)
        heldout_pack = (he, hv, ha, hq, hcids)
        print(f"  Heldout: {len(he)} trials over {len(hcids)} candidates.")

    dims = (eeg_dim, video_dim, audio_dim, n_frames)

    # Selection fit.
    ckpts, epochs = [], []
    for s in seeds:
        cp, ep = train_one_seed(s, args, train_pack, heldout_pack, dims, None, "best")
        ckpts.append(cp); epochs.append(ep)

    # CSLS lambda tuned on heldout (raw-score ensemble).
    MILD = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
    lam = 0.0
    if args.anti_hub != "none":
        if args.hub_penalty > 0:
            lam = args.hub_penalty; print(f"\n[CSLS] fixed lambda={lam}")
        elif args.select_on == "heldout":
            he, hv, ha, hq, hcids = heldout_pack
            raw = [score_matrix(load_model(cp, args), he, hv, ha, args.device) for cp in ckpts]
            ho = np.mean(raw, axis=0)
            best = -1.0
            for pen in MILD:
                adj = apply_anti_hubness(ho, method=args.anti_hub, k=args.csls_k, hub_penalty=pen) if pen > 0 else ho
                r = ranks_from_scores(adj, hcids, hq)
                t1 = float((r == 1).mean())
                if t1 > best:
                    best, lam = t1, pen
            print(f"\n[CSLS] heldout-tuned lambda={lam} (heldout top-1 {best*100:.2f}%)")

    # Full fit.
    sub_ckpts = ckpts
    if args.final_full_fit and args.select_on == "heldout":
        print(f"\n  FULL-FIT on 1-1000 using epochs {dict(zip(seeds, epochs))}")
        # Rebuild train pool on 1..1000.
        e_list, s_list, su_list = [], [], []
        for sub in ALL_TRAIN_SUBJECTS:
            f, sids, subj = load_subject_eeg_by_range(sub, args.trn_dir, 1, 1000,
                                                      window, args.norm_mode, args.eeg_repr)
            e_list.append(f); s_list.append(sids); su_list.append(subj)
        fe = np.concatenate(e_list); fs = np.concatenate(s_list)
        full_pack = (fe, index_video(vb_trn, fs), index_audio(ab_trn, fs), fs.astype(np.int64))
        full = []
        for s, ep in zip(seeds, epochs):
            cp, _ = train_one_seed(s, args, full_pack, None, dims, ep, "full")
            full.append(cp)
        sub_ckpts = full

    # Target EEG (eval + pred) and test candidates.
    target_eeg: Dict[int, np.ndarray] = dict(eval_eeg)
    for sub in PRED_SUBJECTS:
        tp = os.path.join(args.tst_dir, f"sub-{sub:02d}_tst.npy")
        if os.path.exists(tp):
            target_eeg[sub] = apply_eeg_repr(preprocess_test_eeg(load_test_eeg(tp), window, args.norm_mode), args.eeg_repr).numpy()

    # Raw-score seed average, then CSLS once.
    acc: Dict[int, List[np.ndarray]] = {s: [] for s in target_eeg}
    for cp in sub_ckpts:
        m = load_model(cp, args)
        for sub, e in target_eeg.items():
            acc[sub].append(score_matrix(m, e, cand_video_test, cand_audio_test, args.device))
    final = {sub: np.mean(mats, axis=0) for sub, mats in acc.items()}
    if args.anti_hub != "none" and lam > 0:
        for sub in list(final):
            final[sub] = apply_anti_hubness(final[sub], method=args.anti_hub, k=args.csls_k, hub_penalty=lam)

    # Report.
    tag = "Ensemble" if is_ens else "Single"
    fit = "full-fit 1-1000" if (args.final_full_fit and args.select_on == "heldout") else "selection-fit"
    print("\n" + "-" * 70)
    print(f"  [recipe] {tag} TEST estimate sub-01..04 ({fit}, CSLS λ={lam})")
    print("-" * 70)
    print(f"  {'Subject':<9} | {'Top-1':<16} | {'Top-5':<16} | Median")
    print("-" * 70)
    vids = np.array(TEST_IDS)
    for sub in EVAL_SUBJECTS:
        m = retrieval_metrics(compute_ranks(final[sub]))
        print(f"  sub-{sub:02d}    | {m['top1']*100:5.2f}% ({m['top1']*102:2.0f}/102)  | "
              f"{m['top5']*100:5.2f}% ({m['top5']*102:2.0f}/102)  | {m['median_rank']:.1f}")
    a1, a5, am = avg_metrics_over_subjects(final, EVAL_SUBJECTS)
    print("-" * 70)
    print(f"  {'AVERAGE':<9} | {a1*100:5.2f}% ({a1*408:3.0f}/408)  | {a5*100:5.2f}% ({a5*408:3.0f}/408)  | {am:.1f}")
    print(f"  {'CHANCE':<9} |  0.98% ( 4/408)  |  4.90% (20/408)  |  51.5")

    # Save per-subject TEST score matrices, tagged (dinov2 / vjepa2) for blending.
    for sub, S in final.items():
        np.save(os.path.join(args.sub_dir, f"recipe_{args.tag}_sub-{sub:02d}_similarity.npy"), S)

    # Save the HELD-OUT ensemble raw-score matrix (from SELECTION-fit models, trained
    # on 1-900 only → heldout 901-1000 is untouched) so a blender can tune the blend
    # weight honestly on heldout. Also save query/candidate ids for alignment.
    if args.select_on == "heldout":
        he, hv, ha, hq, hcids = heldout_pack
        raw = [score_matrix(load_model(cp, args), he, hv, ha, args.device) for cp in ckpts]
        ho_ens = np.mean(raw, axis=0)
        np.savez(os.path.join(args.sub_dir, f"recipe_{args.tag}_heldout_scores.npz"),
                 scores=ho_ens, query_ids=hq, cand_ids=hcids)
        print(f"\n  Saved heldout score matrix → recipe_{args.tag}_heldout_scores.npz "
              f"(scores {ho_ens.shape})")

    print("\n  Submissions:")
    for sub in PRED_SUBJECTS:
        if sub not in final:
            continue
        ranked = vids[np.argsort(-final[sub], axis=1)]
        assert ranked.shape == (102, 102)
        assert ranked.min() == 1001 and ranked.max() == 1102
        for i in range(102):
            assert len(np.unique(ranked[i])) == 102
        out = os.path.join(args.sub_dir, f"recipe_{args.tag}_sub-{sub:02d}_ranked_candidate_list.npy")
        np.save(out, ranked)
        print(f"    ✓ sub-{sub:02d}: {out}")

    print("\n[recipe] done.")


if __name__ == "__main__":
    main()
