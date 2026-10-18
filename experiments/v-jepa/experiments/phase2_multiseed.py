"""Phase 2 — Multi-Seed Contrastive Training and Ensembling.

Trains multiple independent contrastive models with different random seeds
(different weight initializations, mini-batch shuffles, and augmentation jitters),
then ensembles their predicted similarity matrices.

Averaging across multiple seeds cancels stochastic projector noise and
specifically elevates near-miss candidates (ranks 2-3) into rank 1.

Usage
-----
    # Train 3 seeds and ensemble:
    python experiments/phase2_multiseed.py --device cuda --seeds 42,123,999 --epochs 25

    # If seeds are already trained, run ensemble evaluation and submission only:
    python experiments/phase2_multiseed.py --device cuda --seeds 42,123,999 --eval_only
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_test_eeg, decimate_eeg, zscore_per_subject, normalize_eeg_trials
from src.dataset import EEGStimulusDataset, load_eval_test_data
from src.models.contrastive_eeg import MultimodalContrastiveModel
from src.metrics import compute_ranks, retrieval_metrics, print_metrics, apply_anti_hubness


TEST_DIR = "./tst"
GT_DIR = "./tst_stim"
TEST_IDS = list(range(1001, 1103))  # 102 test videos
EVAL_SUBJECTS = [1, 2, 3, 4]
PRED_SUBJECTS = [5, 6]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Contrastive Ensembling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=str, default="42,123,999", help="Comma-separated random seeds")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per seed")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for InfoNCE")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--proj_dim", type=int, default=256, help="Joint projection dimension")
    parser.add_argument("--dropout", type=float, default=0.25, help="EEG encoder dropout")
    parser.add_argument("--init_temp", type=float, default=0.07, help="Initial temperature")
    parser.add_argument("--use_adapter", action="store_true", default=True, help="Enable Subject Spatial Adapter")
    parser.add_argument("--no_adapter", action="store_false", dest="use_adapter")

    # Organizer baseline preprocessing parameters:
    parser.add_argument("--window_start", type=int, default=5, help="Start timepoint (50 ms at 100 Hz)")
    parser.add_argument("--window_end", type=int, default=60, help="End timepoint (600 ms at 100 Hz)")
    parser.add_argument("--no_window", action="store_true", help="Use full 300 timepoints instead of ERP window")
    parser.add_argument("--norm_mode", type=str, default="trial", choices=["trial", "subject"], help="Normalization mode (trial: per-trial z-score as in baseline)")

    parser.add_argument("--feat_cache", type=str, default="./cache/viclip_ast")
    parser.add_argument("--trn_dir", type=str, default="./trn")
    parser.add_argument("--tst_dir", type=str, default="./tst")
    parser.add_argument("--gt_dir", type=str, default="./tst_stim")
    parser.add_argument("--model_dir", type=str, default="./models/multiseed_erp")
    parser.add_argument("--sub_dir", type=str, default="./submissions")

    parser.add_argument("--anti_hub", type=str, default="csls", choices=["none", "csls", "mean"])
    parser.add_argument("--csls_k", type=int, default=10, help="CSLS neighborhood size")
    parser.add_argument("--hub_penalty", type=float, default=0.25, help="CSLS hub penalty weight (tuned)")
    parser.add_argument("--eval_only", action="store_true", help="Skip training; load saved checkpoints and ensemble")
    return parser.parse_args()


def set_seed(seed: int = 42):
    """Set random seed across python, numpy, and PyTorch for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def extract_subject_similarities(
    model: MultimodalContrastiveModel,
    test_cache: Dict,
    test_feats_tensor: torch.Tensor,
    device: str,
    use_adapter: bool,
) -> Dict[int, np.ndarray]:
    """Compute raw cosine similarity matrices (102, 102) for all subjects.
    
    Returns
    -------
    dict: {sub_id: similarity_matrix_numpy (102, 102)}
    """
    model.eval()
    sim_dict = {}
    for sub, eeg_data in test_cache.items():
        if isinstance(eeg_data, tuple):
            eeg_tensor = eeg_data[0].to(device)
        else:
            eeg_tensor = eeg_data.to(device)

        sub_dev = torch.full((len(eeg_tensor),), sub, dtype=torch.long, device=device) if use_adapter else None
        sim = model.compute_similarity(eeg_tensor, test_feats_tensor, sub_ids=sub_dev)
        sim_dict[sub] = sim.cpu().numpy()
    return sim_dict


def train_single_seed(seed: int, args, train_feats: np.ndarray, test_feats_tensor: torch.Tensor, test_cache: Dict, eeg_window: Optional[Tuple[int, int]] = (5, 60)) -> str:
    """Train one contrastive model with a given seed and save the best checkpoint."""
    set_seed(seed)
    ckpt_path = os.path.join(args.model_dir, f"contrastive_seed_{seed}_best.pt")
    
    print("\n" + "=" * 65)
    print(f"  Training Seed {seed}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  Window: {eeg_window}")
    print("=" * 65)

    # Dataset & loader with deterministic generator
    dataset = EEGStimulusDataset(
        subjects=[1, 2, 3, 4, 5, 6],
        stim_feats=train_feats,
        trn_dir=args.trn_dir,
        augment=True,
        eeg_window=eeg_window,
        norm_mode=args.norm_mode,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
        generator=g,
    )

    model = MultimodalContrastiveModel(
        stim_dim=train_feats.shape[1],
        proj_dim=args.proj_dim,
        init_temp=args.init_temp,
        dropout=args.dropout,
        use_subject_adapter=args.use_adapter,
        num_subjects=6,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_avg_top1 = -1.0
    best_avg_top5 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for eeg_b, stim_b, stim_ids_b, sub_ids_b in dataloader:
            eeg_b = eeg_b.to(args.device)
            stim_b = stim_b.to(args.device)
            stim_ids_b = stim_ids_b.to(args.device)
            sub_ids_b = sub_ids_b.to(args.device) if args.use_adapter else None

            optimizer.zero_grad()
            loss, _ = model(eeg_b, stim_b, stim_ids=stim_ids_b, sub_ids=sub_ids_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = total_loss / max(n_batches, 1)

        # Quick evaluation on sub-01..04
        eval_sims = extract_subject_similarities(
            model,
            {s: test_cache[s] for s in EVAL_SUBJECTS},
            test_feats_tensor,
            args.device,
            args.use_adapter,
        )
        
        # Compute raw top-1 and top-5
        top1_list = []
        top5_list = []
        med_list = []
        for s in EVAL_SUBJECTS:
            r = compute_ranks(eval_sims[s])
            m = retrieval_metrics(r)
            top1_list.append(m["top1"])
            top5_list.append(m["top5"])
            med_list.append(m["median_rank"])

        avg_top1 = float(np.mean(top1_list))
        avg_top5 = float(np.mean(top5_list))
        avg_med = float(np.mean(med_list))

        is_best = (avg_top1 > best_avg_top1) or (avg_top1 == best_avg_top1 and avg_top5 > best_avg_top5)
        marker = "  [★ BEST]" if is_best else ""

        if epoch % 5 == 0 or is_best or epoch == args.epochs:
            print(
                f"Seed {seed:3d} | Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Avg Top-1: {avg_top1:.4f} ({avg_top1*408:.0f}/408) | "
                f"Top-5: {avg_top5:.4f} | Med: {avg_med:.1f}{marker}"
            )

        if is_best:
            best_avg_top1 = avg_top1
            best_avg_top5 = avg_top5
            best_epoch = epoch
            torch.save({
                "seed": seed,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_avg_top1": best_avg_top1,
                "best_avg_top5": best_avg_top5,
            }, ckpt_path)

    print(f"--> Seed {seed} Best Checkpoint at Epoch {best_epoch:02d} (Top-1: {best_avg_top1:.4f}, Top-5: {best_avg_top5:.4f})")
    return ckpt_path


def load_model_from_ckpt(ckpt_path: str, stim_dim: int, proj_dim: int, init_temp: float, dropout: float, use_adapter: bool, device: str):
    """Load model weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model = MultimodalContrastiveModel(
        stim_dim=stim_dim,
        proj_dim=proj_dim,
        init_temp=init_temp,
        dropout=dropout,
        use_subject_adapter=use_adapter,
        num_subjects=6,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    args = parse_args()
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.sub_dir, exist_ok=True)

    print("=" * 70)
    print("  Phase 2 — Multi-Seed Contrastive Ensembling Pipeline")
    print(f"  Seeds: {seed_list}  |  Epochs: {args.epochs}  |  Adapter: {args.use_adapter}")
    print(f"  CSLS: k={args.csls_k}, lambda={args.hub_penalty}  |  Device: {args.device}")
    print("=" * 70)

    eeg_window = None if args.no_window else (args.window_start, args.window_end)
    print(f"  EEG Window: {eeg_window}  |  Norm: {args.norm_mode}")
    print("=" * 70)

    # 1. Load cached features
    train_feat_path = os.path.join(args.feat_cache, "viclip_ast_train_all.npy")
    test_feat_path = os.path.join(args.feat_cache, "viclip_ast_test.npy")
    train_feats = np.load(train_feat_path)
    test_feats = np.load(test_feat_path)
    test_feats_tensor = torch.as_tensor(test_feats, dtype=torch.float32, device=args.device)

    # 2. Pre-load all subject EEG data (eval subjects 1-4 and test subjects 5-6)
    all_subjects_eeg = {}
    
    # Eval subjects (1-4)
    for sub in EVAL_SUBJECTS:
        sorted_eeg, _ = load_eval_test_data(
            sub, test_dir=args.tst_dir, gt_dir=args.gt_dir,
            eeg_window=eeg_window, norm_mode=args.norm_mode
        )
        all_subjects_eeg[sub] = torch.as_tensor(sorted_eeg, dtype=torch.float32)

    # Pred subjects (5-6)
    for sub in PRED_SUBJECTS:
        tst_path = os.path.join(args.tst_dir, f"sub-{sub:02d}_tst.npy")
        if os.path.exists(tst_path):
            raw_test = load_test_eeg(tst_path)
            dec_test = decimate_eeg(raw_test)
            if eeg_window is not None:
                dec_test = dec_test[..., eeg_window[0]:eeg_window[1]]
            if args.norm_mode == "trial":
                norm_test = normalize_eeg_trials(dec_test)
            else:
                norm_test = zscore_per_subject(dec_test)
            all_subjects_eeg[sub] = torch.as_tensor(norm_test, dtype=torch.float32)

    # 3. Train or load models for each seed
    seed_checkpoints = {}
    for s in seed_list:
        ckpt_path = os.path.join(args.model_dir, f"contrastive_seed_{s}_best.pt")
        if args.eval_only and os.path.exists(ckpt_path):
            print(f"Found existing checkpoint for Seed {s} at {ckpt_path}")
            seed_checkpoints[s] = ckpt_path
        else:
            seed_checkpoints[s] = train_single_seed(s, args, train_feats, test_feats_tensor, all_subjects_eeg, eeg_window)

    # 4. Extract similarities from each seed
    print("\n" + "=" * 70)
    print("  Extracting Similarity Matrices Across Seeds")
    print("=" * 70)

    # seed -> sub -> similarity matrix
    per_seed_sims: Dict[int, Dict[int, np.ndarray]] = {}

    for s, ckpt_p in seed_checkpoints.items():
        model = load_model_from_ckpt(
            ckpt_p,
            stim_dim=train_feats.shape[1],
            proj_dim=args.proj_dim,
            init_temp=args.init_temp,
            dropout=args.dropout,
            use_adapter=args.use_adapter,
            device=args.device,
        )
        sims = extract_subject_similarities(model, all_subjects_eeg, test_feats_tensor, args.device, args.use_adapter)
        per_seed_sims[s] = sims

        # Evaluate individual seed performance on eval subjects
        top1_s, top5_s, med_s = [], [], []
        for sub in EVAL_SUBJECTS:
            sim_sub = sims[sub]
            if args.anti_hub == "csls":
                sim_sub = apply_anti_hubness(sim_sub, method="csls", k=args.csls_k, hub_penalty=args.hub_penalty)
            r = compute_ranks(sim_sub)
            m = retrieval_metrics(r)
            top1_s.append(m["top1"])
            top5_s.append(m["top5"])
            med_s.append(m["median_rank"])

        print(f"  Seed {s:3d} (with CSLS) -> Top-1: {np.mean(top1_s)*100:5.2f}% ({np.mean(top1_s)*408:4.0f}/408 hits) | "
              f"Top-5: {np.mean(top5_s)*100:5.2f}% | Med Rank: {np.mean(med_s):4.1f}")

    # 5. Ensemble: Standardized Similarity Averaging
    print("\n" + "=" * 70)
    print("  Ensemble Averaging (Z-Score Standardized per Seed)")
    print("=" * 70)

    ensemble_sims: Dict[int, np.ndarray] = {}
    all_target_subjects = EVAL_SUBJECTS + [s for s in PRED_SUBJECTS if s in all_subjects_eeg]

    for sub in all_target_subjects:
        norm_matrices = []
        for s in seed_list:
            S = per_seed_sims[s][sub]
            # Standardize similarity to equal scale (mean 0, std 1)
            S_norm = (S - S.mean()) / (S.std() + 1e-8)
            norm_matrices.append(S_norm)

        # Average across all seeds
        S_ens = np.mean(norm_matrices, axis=0)

        # Apply CSLS anti-hubness on the averaged similarity matrix
        if args.anti_hub == "csls":
            S_ens = apply_anti_hubness(S_ens, method="csls", k=args.csls_k, hub_penalty=args.hub_penalty)

        ensemble_sims[sub] = S_ens

    # 6. Detailed Evaluation & Comparison on Eval Subjects (1-4)
    print("\n" + "─" * 70)
    print("  Multi-Seed Ensemble Results on Test Subjects (sub-01 .. sub-04)")
    print("─" * 70)
    print(f"  {'Subject':<10} | {'Top-1 Accuracy':<16} | {'Top-5 Accuracy':<16} | {'Median Rank':<12}")
    print("─" * 70)

    ens_top1, ens_top5, ens_med = [], [], []
    all_ranks_ens = {}

    for sub in EVAL_SUBJECTS:
        r = compute_ranks(ensemble_sims[sub])
        all_ranks_ens[sub] = r
        m = retrieval_metrics(r)
        ens_top1.append(m["top1"])
        ens_top5.append(m["top5"])
        ens_med.append(m["median_rank"])
        print(f"  sub-{sub:02d}     | {m['top1']*100:5.2f}% ({m['top1']*102:2.0f}/102)     | "
              f"{m['top5']*100:5.2f}% ({m['top5']*102:2.0f}/102)     | {m['median_rank']:5.1f}")

    avg_top1 = float(np.mean(ens_top1))
    avg_top5 = float(np.mean(ens_top5))
    avg_med = float(np.mean(ens_med))

    print("─" * 70)
    print(f"  {'ENSEMBLE':<10} | {avg_top1*100:5.2f}% ({avg_top1*408:2.0f}/408)     | "
          f"{avg_top5*100:5.2f}% ({avg_top5*408:2.0f}/408)     | {avg_med:5.1f}")
    print(f"  {'CHANCE':<10} |  0.98% ( 4/408)     |  4.90% (20/408)     |  51.5")
    print("─" * 70)

    # Near-miss promotion analysis
    # Count how many trials achieved rank 1 in ensemble that were rank 2-5 in seed 42
    if 42 in per_seed_sims:
        s42_sub1_4_ranks = []
        ens_sub1_4_ranks = []
        for sub in EVAL_SUBJECTS:
            sim42 = per_seed_sims[42][sub]
            if args.anti_hub == "csls":
                sim42 = apply_anti_hubness(sim42, method="csls", k=args.csls_k, hub_penalty=args.hub_penalty)
            s42_sub1_4_ranks.append(compute_ranks(sim42))
            ens_sub1_4_ranks.append(all_ranks_ens[sub])
        
        s42_ranks_cat = np.concatenate(s42_sub1_4_ranks)
        ens_ranks_cat = np.concatenate(ens_sub1_4_ranks)

        promoted = np.sum((s42_ranks_cat > 1) & (s42_ranks_cat <= 5) & (ens_ranks_cat == 1))
        print(f"\n[Diagnostic] Near-Miss Promotion: {promoted} trials promoted from Rank 2-5 into Rank 1!")

    # 7. Generate and verify submissions for sub-05 and sub-06
    video_ids = np.array(TEST_IDS)  # [1001, ..., 1102]
    print("\n" + "=" * 70)
    print("  Generating Competition Submission Files for sub-05 and sub-06")
    print("=" * 70)

    for sub in PRED_SUBJECTS:
        if sub not in ensemble_sims:
            print(f"  Warning: No data found for sub-{sub:02d}, skipping")
            continue

        S_sub = ensemble_sims[sub]  # (102, 102)
        ranked_indices = np.argsort(-S_sub, axis=1)
        ranked_candidates = video_ids[ranked_indices]  # (102, 102)

        out_name = f"sub-{sub:02d}_ranked_candidate_list.npy"
        out_path = os.path.join(args.sub_dir, out_name)
        sim_path = os.path.join(args.sub_dir, f"sub-{sub:02d}_similarity.npy")

        np.save(out_path, ranked_candidates)
        np.save(sim_path, S_sub)

        # Integrity checks
        assert ranked_candidates.shape == (102, 102), f"Expected (102, 102), got {ranked_candidates.shape}"
        assert (ranked_candidates.min() == 1001) and (ranked_candidates.max() == 1102), "ID bounds violated"
        for i in range(102):
            assert len(np.unique(ranked_candidates[i])) == 102, f"Trial {i} has duplicate rankings"

        print(f"  ✓ sub-{sub:02d}: Saved -> {out_path} (shape: {ranked_candidates.shape}, integrity verified)")

    print("\nMulti-seed ensembling and submission generation complete!")


if __name__ == "__main__":
    main()
