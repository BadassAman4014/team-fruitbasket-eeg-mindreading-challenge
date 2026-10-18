"""Simple-MLP contrastive baseline — train, eval, and submission.

The honest, minimal counterpart to experiments/phase2_multiseed.py. Reuses the
SAME cached stimulus features and the SAME EEG preprocessing / dataset loaders
as the heavy pipeline, so the comparison is apples-to-apples. The only thing
that changes is the model: two small MLPs + InfoNCE (see simple_mlp/model.py).

Supports three experiments on top of the raw single-seed baseline:
  1. Multi-seed averaging   (--seeds 42,123,999)
     Train each seed, z-standardize each similarity matrix, average. Cancels
     per-seed noise and tells you whether the headline number is stable.
  2. (diagnostics live in simple_mlp/diagnostics.py)
  3. Honest model selection (--select_on heldout)
     Pick the best epoch on a held-out TRAINING-stimuli split (default 901-1000)
     instead of sub-01..04 test labels, so sub-01..04 test stays an untouched
     generalization estimate.

Defaults to ViCLIP + CLAP features (1280-d); swap to AST with
--feat_cache ./cache/viclip_ast --feat_prefix viclip_ast.

Usage (cluster)
---------------
    # Single-seed baseline (original behavior):
    python simple_mlp/train.py --device cuda --epochs 30

    # Multi-seed ensemble:
    python simple_mlp/train.py --device cuda --epochs 30 --seeds 42,123,999

    # Honest selection on held-out training stimuli:
    python simple_mlp/train.py --device cuda --epochs 30 --select_on heldout
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make the repo root importable so we can reuse src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (
    EEGStimulusDataset,
    load_eval_test_data,
    load_val_retrieval_data,
)
from src.data_utils import load_test_eeg, decimate_eeg, normalize_eeg_trials, zscore_per_subject, SFREQ_DEC
from src.metrics import compute_ranks, retrieval_metrics, apply_anti_hubness

from simple_mlp.model import SimpleContrastiveModel
from simple_mlp.eeg_features import build_eeg_representation

# ViCLIP occupies the first 768 dims of the cached feature; CLAP the last 512.
# (AST caches are 768+768; the AST audio tail is sliced the same way via --vid_dim.)
VICLIP_DIM = 768

TEST_IDS = list(range(1001, 1103))  # 102 test video IDs
EVAL_SUBJECTS = [1, 2, 3, 4]        # have ground-truth labels → used for validation only
PRED_SUBJECTS = [5, 6]             # held-out → we must produce rankings
ALL_TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6]  # training EEG of all 6 subjects is allowed


def parse_args():
    p = argparse.ArgumentParser(description="Simple-MLP contrastive baseline")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=str, default="42",
                   help="Comma-separated seeds. One seed = single model; many = z-standardized average.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--init_temp", type=float, default=0.07)

    # EEG preprocessing — identical to the heavy pipeline's defaults.
    p.add_argument("--window_start", type=int, default=5, help="ERP window start (50 ms @ 100 Hz)")
    p.add_argument("--window_end", type=int, default=60, help="ERP window end (600 ms @ 100 Hz)")
    p.add_argument("--no_window", action="store_true", help="Use full 300 timepoints")
    p.add_argument("--norm_mode", type=str, default="trial", choices=["trial", "subject"])
    p.add_argument("--eeg_repr", type=str, default="raw",
                   choices=["raw", "bandpower", "both", "rich", "both_rich"],
                   help="EEG representation: raw voltages; bandpower (3-band); both; "
                        "rich (5-band 85 + connectivity 136 = 221); both_rich (raw + 221).")

    # Model selection signal.
    p.add_argument("--select_on", type=str, default="dev", choices=["dev", "heldout"],
                   help="dev: best epoch by sub-01..04 test; heldout: by a held-out training-stimuli split.")
    p.add_argument("--val_lo", type=int, default=901, help="Held-out stimuli ID low (inclusive)")
    p.add_argument("--val_hi", type=int, default=1000, help="Held-out stimuli ID high (inclusive)")

    # Feature cache (reuse existing .npy caches; nothing is extracted here).
    p.add_argument("--feat_cache", type=str, default="./cache/viclip_clap",
                   help="Directory with cached stimulus features")
    p.add_argument("--feat_prefix", type=str, default="viclip_clap",
                   help="Cache file prefix: '<prefix>_train_all.npy' and '<prefix>_test.npy'")
    p.add_argument("--stim_source", type=str, default="both", choices=["video", "audio", "both"],
                   help="Which part of the cached feature to use: video (ViCLIP), audio (CLAP/AST), or both.")
    p.add_argument("--vid_dim", type=int, default=VICLIP_DIM,
                   help="Dimension of the leading video (ViCLIP) block in the cached feature.")

    # Anti-hubness (CSLS) applied to the FINAL averaged similarity matrix. Off by default.
    p.add_argument("--anti_hub", type=str, default="none", choices=["none", "csls", "mean"],
                   help="Anti-hubness post-processing on the final similarity matrix.")
    p.add_argument("--csls_k", type=int, default=10, help="CSLS neighborhood size.")
    p.add_argument("--hub_penalty", type=float, default=0.0,
                   help="Fixed CSLS/mean penalty. If 0, the FIXED mild grid is tuned on the VALIDATION signal "
                        "(heldout stimuli when --select_on heldout; dev otherwise). Do NOT widen the grid to "
                        "chase the dev score — that overfits the knob to the eval subjects.")

    # Final submission fit: after selecting epoch + lambda on validation, retrain on ALL 1000 stimuli.
    p.add_argument("--final_full_fit", action="store_true",
                   help="Train/validate on 1-900 to select epoch & CSLS lambda, then RETRAIN each seed on all "
                        "1000 stimuli for the sub-05/06 submission. Only meaningful with --select_on heldout.")

    p.add_argument("--trn_dir", type=str, default="./trn")
    p.add_argument("--tst_dir", type=str, default="./tst")
    p.add_argument("--gt_dir", type=str, default="./tst_stim")
    p.add_argument("--model_dir", type=str, default="./models/simple_mlp")
    p.add_argument("--sub_dir", type=str, default="./submissions")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_test_eeg(raw: np.ndarray, window, norm_mode: str) -> np.ndarray:
    """Decimate → window → normalize test EEG, matching the dataset loader."""
    dec = decimate_eeg(raw)
    if window is not None:
        dec = dec[..., window[0]:window[1]]
    if norm_mode == "trial":
        return normalize_eeg_trials(dec)
    return zscore_per_subject(dec)


def apply_eeg_repr(eeg_3d, eeg_repr: str):
    """Transform (N,C,T) EEG → the chosen representation as a float32 torch tensor.

    repr='raw' flattens voltages; 'bandpower'/'both' use per-trial log band-power.
    Band-power is computed at the decimated rate (SFREQ_DEC). Returns (N, D) tensor.
    """
    arr = eeg_3d.numpy() if isinstance(eeg_3d, torch.Tensor) else np.asarray(eeg_3d)
    feat = build_eeg_representation(arr, eeg_repr, sfreq=float(SFREQ_DEC))
    return torch.as_tensor(feat, dtype=torch.float32)


@torch.no_grad()
def similarity_for_subjects(
    model: SimpleContrastiveModel,
    eeg_by_sub: Dict[int, torch.Tensor],
    test_feats_t: torch.Tensor,
    device: str,
) -> Dict[int, np.ndarray]:
    """Compute a raw (102, 102) cosine similarity matrix per subject."""
    model.eval()
    sims = {}
    for sub, eeg_t in eeg_by_sub.items():
        sim = model.compute_similarity(eeg_t.to(device), test_feats_t)
        sims[sub] = sim.cpu().numpy()
    return sims


def avg_metrics_over_subjects(sims: Dict[int, np.ndarray], subjects: List[int]) -> Tuple[float, float, float]:
    """Average top-1 / top-5 / median rank across the given subjects."""
    t1, t5, med = [], [], []
    for sub in subjects:
        m = retrieval_metrics(compute_ranks(sims[sub]))
        t1.append(m["top1"]); t5.append(m["top5"]); med.append(m["median_rank"])
    return float(np.mean(t1)), float(np.mean(t5)), float(np.mean(med))


def heldout_ranks_from_sim(S: np.ndarray, val_stims: np.ndarray, cand_ids: np.ndarray) -> np.ndarray:
    """1-indexed rank of the true candidate for each held-out trial, from a sim matrix.

    S : (N_val, N_cand). The correct candidate for row i is the column whose
    cand_id equals val_stims[i] (guaranteed present: candidates == val stimuli).
    """
    order = np.argsort(-S, axis=1)        # (N_val, N_cand) column indices, best first
    ranked_ids = cand_ids[order]          # (N_val, N_cand) stimulus IDs
    hit_pos = (ranked_ids == val_stims[:, None])
    return hit_pos.argmax(axis=1) + 1


def heldout_metrics_from_sim(S: np.ndarray, val_stims: np.ndarray, cand_ids: np.ndarray) -> Tuple[float, float]:
    ranks = heldout_ranks_from_sim(S, val_stims, cand_ids)
    return float((ranks == 1).mean()), float((ranks <= 5).mean())


@torch.no_grad()
def heldout_similarity(
    model: SimpleContrastiveModel,
    val_eeg_t: torch.Tensor,
    cand_feats_t: torch.Tensor,
    device: str,
) -> np.ndarray:
    """(N_val, N_cand) raw cosine similarity on the held-out training stimuli."""
    model.eval()
    return model.compute_similarity(val_eeg_t.to(device), cand_feats_t).cpu().numpy()


def train_one_seed(
    seed: int,
    args,
    window: Optional[Tuple[int, int]],
    train_feats: np.ndarray,
    stim_dim: int,
    test_feats_t: torch.Tensor,
    eval_eeg: Dict[int, torch.Tensor],
    heldout_pack: Optional[dict],
    train_stim_range: Optional[Tuple[int, int]] = None,
    fixed_epochs: Optional[int] = None,
    ckpt_suffix: str = "best",
) -> Tuple[str, int]:
    """Train one seed. Returns (ckpt_path, selected_epoch).

    Two modes:
      - Selection fit (default): train on `train_stim_range`, pick the best epoch
        by the validation signal (heldout or dev), save that epoch's weights.
      - Final full fit (`fixed_epochs` set): train on `train_stim_range` for exactly
        `fixed_epochs` epochs with NO per-epoch selection, save the last epoch.
        Used to retrain on all 1000 stimuli for the submission.
    """
    set_seed(seed)
    final_fit = fixed_epochs is not None
    n_epochs = fixed_epochs if final_fit else args.epochs

    print("\n" + "=" * 70)
    mode = f"FINAL full-fit ({n_epochs} epochs, no selection)" if final_fit else f"select_on={args.select_on}"
    print(f"  Training seed {seed}  |  {mode}  |  stim_range={train_stim_range or 'all'}")
    print("=" * 70)

    dataset = EEGStimulusDataset(
        subjects=ALL_TRAIN_SUBJECTS,
        stim_feats=train_feats,
        trn_dir=args.trn_dir,
        augment=False,
        stim_range=train_stim_range,
        eeg_window=window,
        norm_mode=args.norm_mode,
    )
    # Replace the raw windowed EEG with the chosen representation (raw/bandpower/both).
    # augment=False, so __getitem__ returns dataset.eeg[idx] unchanged → safe to make it 2D.
    dataset.eeg = apply_eeg_repr(dataset.eeg, args.eeg_repr)
    eeg_dim = int(dataset.eeg.shape[1])

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
        generator=g,
    )

    model = SimpleContrastiveModel(
        eeg_dim=eeg_dim,
        stim_dim=stim_dim,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        init_temp=args.init_temp,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    ckpt_path = os.path.join(args.model_dir, f"simple_mlp_{args.feat_prefix}_seed{seed}_{ckpt_suffix}.pt")
    best_sel, best_epoch = -1.0, 0

    def _save(ep: int):
        torch.save({
            "epoch": ep, "seed": seed, "model_state": model.state_dict(),
            "args": vars(args), "eeg_dim": eeg_dim, "stim_dim": stim_dim,
            "select_on": args.select_on, "best_select_score": best_sel,
            "final_fit": final_fit,
        }, ckpt_path)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for eeg_b, stim_b, stim_ids_b, _sub_ids_b in loader:
            eeg_b = eeg_b.to(args.device)
            stim_b = stim_b.to(args.device)
            stim_ids_b = stim_ids_b.to(args.device)

            optimizer.zero_grad()
            loss, _ = model(eeg_b, stim_b, stim_ids=stim_ids_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        epoch_loss = total_loss / max(n_batches, 1)

        # FINAL full-fit: no selection, just train to fixed_epochs and save the last epoch.
        if final_fit:
            if epoch % 5 == 0 or epoch == n_epochs:
                print(f"Seed {seed} | Epoch {epoch:02d}/{n_epochs:02d} | loss {epoch_loss:.4f} (full-fit)")
            best_epoch = epoch
            _save(epoch)
            continue

        # --- Model-selection signal (RAW cosine, no CSLS) ---
        if args.select_on == "heldout":
            S_val = heldout_similarity(
                model, heldout_pack["val_eeg"], heldout_pack["cand_feats"], args.device
            )
            sel_top1, sel_top5 = heldout_metrics_from_sim(
                S_val, heldout_pack["val_stims"], heldout_pack["cand_ids"]
            )
            sel_score = sel_top1
            # Dev numbers are still reported for information, but NOT used for selection.
            dev_sims = similarity_for_subjects(model, eval_eeg, test_feats_t, args.device)
            dev_t1, dev_t5, dev_med = avg_metrics_over_subjects(dev_sims, EVAL_SUBJECTS)
            sel_desc = f"heldout top-1 {sel_top1*100:5.2f}% / top-5 {sel_top5*100:5.2f}%"
            info_desc = f"[info] dev top-1 {dev_t1*100:5.2f}% top-5 {dev_t5*100:5.2f}% med {dev_med:4.1f}"
        else:
            dev_sims = similarity_for_subjects(model, eval_eeg, test_feats_t, args.device)
            dev_t1, dev_t5, dev_med = avg_metrics_over_subjects(dev_sims, EVAL_SUBJECTS)
            sel_score = dev_t1
            sel_desc = f"dev top-1 {dev_t1*100:5.2f}% ({dev_t1*408:4.0f}/408) / top-5 {dev_t5*100:5.2f}% / med {dev_med:4.1f}"
            info_desc = ""

        is_best = sel_score > best_sel
        marker = "  [* best]" if is_best else ""
        if epoch % 5 == 0 or is_best or epoch == args.epochs:
            line = f"Seed {seed} | Epoch {epoch:02d}/{n_epochs:02d} | loss {epoch_loss:.4f} | {sel_desc}{marker}"
            if info_desc:
                line += f"  {info_desc}"
            print(line)

        if is_best:
            best_sel, best_epoch = sel_score, epoch
            _save(epoch)

    if final_fit:
        print(f"--> Seed {seed}: full-fit complete at epoch {best_epoch:02d}  →  {ckpt_path}")
    else:
        print(f"--> Seed {seed}: best epoch {best_epoch:02d} ({args.select_on} score {best_sel*100:.2f}%)  →  {ckpt_path}")
    return ckpt_path, best_epoch


def load_model(ckpt_path: str, args, eeg_dim: int, stim_dim: int) -> SimpleContrastiveModel:
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model = SimpleContrastiveModel(
        eeg_dim=eeg_dim,
        stim_dim=stim_dim,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        init_temp=args.init_temp,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def zstandardize(S: np.ndarray) -> np.ndarray:
    """Z-standardize a similarity matrix to a common scale before averaging."""
    return (S - S.mean()) / (S.std() + 1e-8)


def slice_feature_source(feats: np.ndarray, stim_source: str, vid_dim: int) -> np.ndarray:
    """Select video-only / audio-only / both from a concatenated [video | audio] feature.

    Tests whether audio dims pollute the (posterior, visual) EEG regression.
    """
    if stim_source == "both":
        return feats
    if stim_source == "video":
        return feats[:, :vid_dim]
    if stim_source == "audio":
        return feats[:, vid_dim:]
    raise ValueError(f"Unknown stim_source: {stim_source}")


def main():
    args = parse_args()
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.sub_dir, exist_ok=True)

    window = None if args.no_window else (args.window_start, args.window_end)
    is_ensemble = len(seed_list) > 1

    print("=" * 70)
    print("  Simple-MLP Contrastive Baseline (two small MLPs + InfoNCE)")
    print(f"  Features: {args.feat_prefix}  |  Window: {window}  |  Norm: {args.norm_mode}")
    print(f"  Seeds: {seed_list}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  Device: {args.device}")
    print(f"  EEG repr: {args.eeg_repr}  |  Stim source: {args.stim_source}  |  Anti-hub: {args.anti_hub}"
          + (f" (k={args.csls_k}, λ={'auto' if args.hub_penalty == 0 else args.hub_penalty})" if args.anti_hub != "none" else ""))
    print(f"  Model selection: {args.select_on}"
          + (f" (held-out stimuli {args.val_lo}-{args.val_hi})" if args.select_on == "heldout" else " (sub-01..04 test)"))
    if args.final_full_fit:
        print(f"  Final full-fit: ON (retrain on all 1000 stimuli for the submission)")
    print("=" * 70)

    # 1. Load cached stimulus features (NOTHING is extracted here).
    train_feat_path = os.path.join(args.feat_cache, f"{args.feat_prefix}_train_all.npy")
    test_feat_path = os.path.join(args.feat_cache, f"{args.feat_prefix}_test.npy")
    for pth in (train_feat_path, test_feat_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(
                f"Missing feature cache: {pth}\n"
                f"Run the matching phase1/phase2 extraction first (e.g. "
                f"experiments/phase2_viclip_clap.py) to populate {args.feat_cache}."
            )
    train_feats = np.load(train_feat_path)   # (1001, full_dim), 1-based ID index
    test_feats = np.load(test_feat_path)     # (102, full_dim)
    full_dim = train_feats.shape[1]

    # Select video / audio / both to test whether audio pollutes the visual EEG mapping.
    train_feats = np.ascontiguousarray(slice_feature_source(train_feats, args.stim_source, args.vid_dim))
    test_feats = np.ascontiguousarray(slice_feature_source(test_feats, args.stim_source, args.vid_dim))
    stim_dim = train_feats.shape[1]
    test_feats_t = torch.as_tensor(test_feats, dtype=torch.float32, device=args.device)
    print(f"  Loaded features: full_dim={full_dim}  |  stim_source={args.stim_source}  |  using stim_dim={stim_dim}")
    print(f"  Shapes after slice: train {train_feats.shape}, test {test_feats.shape}")

    # 2. Evaluation EEG (sub-01..04, label-sorted so diagonal = correct match).
    eval_eeg: Dict[int, torch.Tensor] = {}
    for sub in EVAL_SUBJECTS:
        sorted_eeg, _ = load_eval_test_data(
            sub, test_dir=args.tst_dir, gt_dir=args.gt_dir,
            eeg_window=window, norm_mode=args.norm_mode,
        )
        eval_eeg[sub] = apply_eeg_repr(sorted_eeg, args.eeg_repr)

    # 3. Optional held-out-stimuli validation pack (only for selection; built from TRAIN data).
    heldout_pack = None
    if args.select_on == "heldout":
        pack = load_val_retrieval_data(
            subjects=ALL_TRAIN_SUBJECTS,
            stim_feats=train_feats,
            trn_dir=args.trn_dir,
            val_range=(args.val_lo, args.val_hi),
            eeg_window=window,
            norm_mode=args.norm_mode,
        )
        heldout_pack = {
            "val_eeg": apply_eeg_repr(pack["val_eeg"], args.eeg_repr),
            "val_stims": pack["val_stims"],
            "cand_feats": torch.as_tensor(pack["cand_feats"], dtype=torch.float32, device=args.device),
            "cand_ids": pack["cand_ids"],
        }
        print(f"  Held-out validation: {len(pack['val_stims'])} trials over "
              f"{len(pack['cand_ids'])} candidate stimuli ({args.val_lo}-{args.val_hi}).")

    # Determine eeg_dim once. eval_eeg is already transformed to the chosen
    # representation (2D: N x D), so its feature dim is the model input dim.
    eeg_dim = int(eval_eeg[EVAL_SUBJECTS[0]].shape[1])

    # The training range for the SELECTION fit: exclude the heldout stimuli when
    # selecting on heldout, so the validation signal is leakage-free.
    sel_range = (1, args.val_lo - 1) if args.select_on == "heldout" else None

    # 4. Train each seed (selection fit) and record the epoch chosen on validation.
    ckpt_paths, chosen_epochs = [], []
    for seed in seed_list:
        cp, ep = train_one_seed(
            seed, args, window, train_feats, stim_dim, test_feats_t, eval_eeg, heldout_pack,
            train_stim_range=sel_range, fixed_epochs=None, ckpt_suffix="best",
        )
        ckpt_paths.append(cp)
        chosen_epochs.append(ep)

    # 4b. Tune the CSLS penalty HONESTLY on the ensemble validation signal (NOT dev),
    #     using a FIXED mild grid. When select_on=heldout, the signal is the held-out
    #     stimuli retrieval; when select_on=dev, it falls back to dev (sub-01..04).
    MILD_GRID = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]  # fixed; do not widen to chase a score
    chosen_penalty = 0.0
    if args.anti_hub != "none":
        if args.hub_penalty > 0:
            chosen_penalty = args.hub_penalty
            print(f"\n[Anti-hubness] {args.anti_hub} fixed λ={chosen_penalty:.2f} (user-set, off-selection), k={args.csls_k}")
        elif args.select_on == "heldout":
            # Build the ensemble heldout similarity matrix (z-standardized per seed, averaged).
            ho_mats = []
            for cp in ckpt_paths:
                mdl = load_model(cp, args, eeg_dim, stim_dim)
                S_ho = heldout_similarity(mdl, heldout_pack["val_eeg"], heldout_pack["cand_feats"], args.device)
                ho_mats.append(zstandardize(S_ho) if is_ensemble else S_ho)
            S_ho_ens = np.mean(ho_mats, axis=0)

            best_pen, best_ho = 0.0, -1.0
            for pen in MILD_GRID:
                adj = apply_anti_hubness(S_ho_ens, method=args.anti_hub, k=args.csls_k, hub_penalty=pen) if pen > 0 else S_ho_ens
                t1, _ = heldout_metrics_from_sim(adj, heldout_pack["val_stims"], heldout_pack["cand_ids"])
                if t1 > best_ho:
                    best_ho, best_pen = t1, pen
            chosen_penalty = best_pen
            print(f"\n[Anti-hubness] {args.anti_hub}: HELDOUT-tuned over fixed grid {MILD_GRID} → "
                  f"λ={chosen_penalty:.2f} (heldout top-1 {best_ho*100:.2f}%), k={args.csls_k}")
        else:
            # select_on=dev → tuning on dev is consistent with the (dev) selection signal.
            # Build dev ensemble first (reuse section 5 logic inline on eval subjects).
            dev_mats = {s: [] for s in EVAL_SUBJECTS}
            for cp in ckpt_paths:
                mdl = load_model(cp, args, eeg_dim, stim_dim)
                sims = similarity_for_subjects(mdl, eval_eeg, test_feats_t, args.device)
                for s in EVAL_SUBJECTS:
                    dev_mats[s].append(zstandardize(sims[s]) if is_ensemble else sims[s])
            dev_ens = {s: np.mean(dev_mats[s], axis=0) for s in EVAL_SUBJECTS}
            best_pen, best_t1 = 0.0, -1.0
            for pen in MILD_GRID:
                adj = {s: (apply_anti_hubness(dev_ens[s], method=args.anti_hub, k=args.csls_k, hub_penalty=pen) if pen > 0 else dev_ens[s]) for s in EVAL_SUBJECTS}
                t1, _, _ = avg_metrics_over_subjects(adj, EVAL_SUBJECTS)
                if t1 > best_t1:
                    best_t1, best_pen = t1, pen
            chosen_penalty = best_pen
            print(f"\n[Anti-hubness] {args.anti_hub}: dev-tuned over fixed grid {MILD_GRID} → "
                  f"λ={chosen_penalty:.2f} (dev top-1 {best_t1*100:.2f}%), k={args.csls_k}")

    # 4c. FINAL FULL FIT: retrain each seed on ALL 1000 stimuli for the submission.
    #     Uses the per-seed epoch chosen on validation (no further selection).
    submission_ckpts = ckpt_paths
    if args.final_full_fit and args.select_on == "heldout":
        print("\n" + "=" * 70)
        print("  FINAL FULL FIT — retraining each seed on ALL 1000 stimuli")
        print(f"  Using validation-selected epochs per seed: {dict(zip(seed_list, chosen_epochs))}")
        print("=" * 70)
        full_ckpts = []
        for seed, ep in zip(seed_list, chosen_epochs):
            cp, _ = train_one_seed(
                seed, args, window, train_feats, stim_dim, test_feats_t, eval_eeg, heldout_pack=None,
                train_stim_range=None, fixed_epochs=ep, ckpt_suffix="full",
            )
            full_ckpts.append(cp)
        submission_ckpts = full_ckpts
    elif args.final_full_fit:
        print("\n[Note] --final_full_fit requires --select_on heldout; ignoring (using selection-fit models).")

    # 5. Build per-subject similarity matrices for every target subject (eval + pred),
    #    using the SUBMISSION models (full-fit if requested, else selection-fit).
    target_eeg: Dict[int, torch.Tensor] = dict(eval_eeg)
    for sub in PRED_SUBJECTS:
        tst_path = os.path.join(args.tst_dir, f"sub-{sub:02d}_tst.npy")
        if os.path.exists(tst_path):
            raw = load_test_eeg(tst_path)
            eeg = preprocess_test_eeg(raw, window, args.norm_mode)
            target_eeg[sub] = apply_eeg_repr(eeg, args.eeg_repr)
        else:
            print(f"  Warning: {tst_path} not found — sub-{sub:02d} submission will be skipped.")

    accum: Dict[int, List[np.ndarray]] = {sub: [] for sub in target_eeg}
    for cp in submission_ckpts:
        model = load_model(cp, args, eeg_dim, stim_dim)
        sims = similarity_for_subjects(model, target_eeg, test_feats_t, args.device)
        for sub, S in sims.items():
            accum[sub].append(zstandardize(S) if is_ensemble else S)

    final_sims: Dict[int, np.ndarray] = {}
    for sub, mats in accum.items():
        final_sims[sub] = np.mean(mats, axis=0) if is_ensemble else mats[0]

    # 5b. Apply the honestly-chosen CSLS penalty to the final matrices.
    if args.anti_hub != "none" and chosen_penalty > 0:
        for s in list(final_sims.keys()):
            final_sims[s] = apply_anti_hubness(final_sims[s], method=args.anti_hub,
                                               k=args.csls_k, hub_penalty=chosen_penalty)

    # 6. Final per-subject report on sub-01..04. Test videos (1001-1102) are disjoint
    #    from training stimuli (1-1000), so these TEST numbers remain an untouched
    #    generalization estimate even after --final_full_fit, as long as selection
    #    (epoch + CSLS lambda) was done on heldout rather than dev.
    tag = "Ensemble" if is_ensemble else "Single-seed"
    fit_note = "full-fit on 1-1000" if (args.final_full_fit and args.select_on == "heldout") else "selection-fit"
    hub_note = "no CSLS" if (args.anti_hub == "none" or chosen_penalty == 0) else f"{args.anti_hub} λ={chosen_penalty:.2f}"
    print("\n" + "-" * 70)
    print(f"  {tag} TEST estimate on sub-01..04 ({fit_note}, {hub_note})")
    print("-" * 70)
    print(f"  {'Subject':<9} | {'Top-1':<16} | {'Top-5':<16} | {'Median Rank'}")
    print("-" * 70)
    for sub in EVAL_SUBJECTS:
        m = retrieval_metrics(compute_ranks(final_sims[sub]))
        print(f"  sub-{sub:02d}    | {m['top1']*100:5.2f}% ({m['top1']*102:2.0f}/102)  | "
              f"{m['top5']*100:5.2f}% ({m['top5']*102:2.0f}/102)  | {m['median_rank']:.1f}")
    avg_t1, avg_t5, avg_med = avg_metrics_over_subjects(final_sims, EVAL_SUBJECTS)
    print("-" * 70)
    print(f"  {'AVERAGE':<9} | {avg_t1*100:5.2f}% ({avg_t1*408:3.0f}/408)  | "
          f"{avg_t5*100:5.2f}% ({avg_t5*408:3.0f}/408)  | {avg_med:.1f}")
    print(f"  {'CHANCE':<9} |  0.98% ( 4/408)  |  4.90% (20/408)  |  51.5")

    # 7. Save per-subject similarity matrices (for diagnostics.py) + submissions.
    for sub, S in final_sims.items():
        np.save(os.path.join(args.sub_dir, f"simple_mlp_sub-{sub:02d}_similarity.npy"), S)

    video_ids = np.array(TEST_IDS)
    print("\n" + "=" * 70)
    print("  Generating held-out submissions for sub-05 and sub-06")
    print("=" * 70)
    for sub in PRED_SUBJECTS:
        if sub not in final_sims:
            print(f"  Warning: no similarity for sub-{sub:02d}; skipped.")
            continue
        S = final_sims[sub]
        ranked = video_ids[np.argsort(-S, axis=1)]  # (102, 102), ORIGINAL row order

        assert ranked.shape == (102, 102), f"Expected (102, 102), got {ranked.shape}"
        assert ranked.min() == 1001 and ranked.max() == 1102, "Video ID bounds violated"
        for i in range(102):
            assert len(np.unique(ranked[i])) == 102, f"Row {i} is not a permutation"

        out_path = os.path.join(args.sub_dir, f"simple_mlp_sub-{sub:02d}_ranked_candidate_list.npy")
        np.save(out_path, ranked)
        print(f"  ✓ sub-{sub:02d}: saved -> {out_path}  (shape {ranked.shape}, integrity verified)")

    print("\nDone. Simple-MLP run complete.")


if __name__ == "__main__":
    main()
