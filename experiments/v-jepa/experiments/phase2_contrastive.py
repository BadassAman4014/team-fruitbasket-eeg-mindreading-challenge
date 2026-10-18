"""Phase 2 — Multimodal Contrastive Training (InfoNCE).

Trains a deep EEG encoder jointly across all subjects to align EEG neural
responses with multimodal (ViCLIP video + AST audio) stimulus embeddings.

Directly optimizes retrieval ranking via symmetric InfoNCE loss (NT-Xent).

Usage
-----
    python experiments/phase2_contrastive.py --device cuda --epochs 50 --batch_size 64
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import EEGStimulusDataset, load_eval_test_data
from src.models.contrastive_eeg import MultimodalContrastiveModel
from src.metrics import compute_ranks, retrieval_metrics, print_metrics, apply_anti_hubness


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Contrastive Training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for InfoNCE")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--proj_dim", type=int, default=256, help="Joint embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate in EEG encoder")
    parser.add_argument("--feat_cache", type=str, default="./cache/viclip_ast", help="Path to cached multimodal features")
    parser.add_argument("--model_dir", type=str, default="./models/phase2", help="Directory to save model checkpoints")
    parser.add_argument("--trn_dir", type=str, default="./trn", help="Directory containing training EEG npz files")
    parser.add_argument("--tst_dir", type=str, default="./tst", help="Directory containing test EEG npy files")
    parser.add_argument("--gt_dir", type=str, default="./tst_stim", help="Directory containing test ground truth")
    parser.add_argument("--train_subjects", type=str, default="1,2,3,4,5,6", help="Comma-separated list of train subjects")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable EEG data augmentation during training")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate on test set every N epochs")
    parser.add_argument("--anti_hub", type=str, default="csls", choices=["none", "csls", "mean"], help="Anti-hubness method")
    parser.add_argument("--csls_k", type=int, default=10, help="Neighborhood size for CSLS anti-hubness")
    parser.add_argument("--hub_penalty", type=float, default=1.0, help="CSLS hub penalty weight")
    parser.add_argument("--init_temp", type=float, default=0.07, help="Initial temperature")
    parser.add_argument("--use_adapter", action="store_true", default=False, help="Enable Subject Spatial Adapter")
    parser.add_argument("--hparams_file", type=str, default="", help="Optional JSON file from tuning to load best hyperparameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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
def evaluate_all_test_subjects(
    model,
    test_cache,
    test_feats_tensor,
    device,
    anti_hub="none",
    csls_k=10,
    hub_penalty=1.0,
    use_adapter=False,
):
    """Evaluate model on sub-01 through sub-04 test sets.

    Parameters
    ----------
    model : MultimodalContrastiveModel
    test_cache : dict of (sub -> (eeg_tensor, gt_ids))
    test_feats_tensor : (102, 1536) tensor on device
    anti_hub : str, "none", "csls", or "mean"
    csls_k : int, neighborhood size for CSLS
    hub_penalty : float, CSLS penalty weight
    use_adapter : bool, whether to pass subject ID to model

    Returns
    -------
    dict of subject -> (metrics, sim, ranks), and avg_metrics
    """
    model.eval()
    sub_metrics = {}
    sub_sims = {}
    sub_ranks = {}

    for sub, (eeg_tensor, gt_ids) in test_cache.items():
        eeg_dev = eeg_tensor.to(device)
        sub_dev = torch.full((len(eeg_dev),), sub, dtype=torch.long, device=device) if use_adapter else None
        sim = model.compute_similarity(eeg_dev, test_feats_tensor, sub_ids=sub_dev).cpu().numpy()  # (102, 102)

        if anti_hub != "none":
            sim = apply_anti_hubness(sim, method=anti_hub, k=csls_k, hub_penalty=hub_penalty)

        ranks = compute_ranks(sim)
        metrics = retrieval_metrics(ranks)
        sub_metrics[sub] = metrics
        sub_sims[sub] = sim
        sub_ranks[sub] = ranks


    avg_top1 = float(np.mean([m["top1"] for m in sub_metrics.values()]))
    avg_top5 = float(np.mean([m["top5"] for m in sub_metrics.values()]))
    avg_median = float(np.mean([m["median_rank"] for m in sub_metrics.values()]))

    avg_metrics = {
        "top1": avg_top1,
        "top5": avg_top5,
        "median_rank": avg_median,
        "chance_top1": 1.0 / 102,
        "chance_top5": 5.0 / 102,
    }
    return sub_metrics, avg_metrics, sub_sims, sub_ranks


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("  Phase 2 — Multimodal Contrastive Learning (InfoNCE)")
    print(f"  Device: {args.device} | Epochs: {args.epochs} | Batch size: {args.batch_size} | Seed: {args.seed}")
    print("=" * 60)

    if args.hparams_file and os.path.exists(args.hparams_file):
        import json
        print(f"Loading winning hyperparameters from {args.hparams_file} ...")
        with open(args.hparams_file) as f:
            hparams = json.load(f)
        cfg = hparams.get("config", {})
        args.batch_size = cfg.get("batch_size", args.batch_size)
        args.lr = cfg.get("lr", args.lr)
        args.proj_dim = cfg.get("proj_dim", args.proj_dim)
        args.init_temp = cfg.get("init_temp", args.init_temp)
        args.dropout = cfg.get("dropout", args.dropout)
        args.use_adapter = cfg.get("use_adapter", args.use_adapter)
        args.csls_k = hparams.get("csls_k", args.csls_k)
        args.hub_penalty = hparams.get("csls_lambda", args.hub_penalty)
        print(f"  Loaded config: Batch={args.batch_size}, LR={args.lr}, Proj={args.proj_dim}, "
              f"Adapter={args.use_adapter}, CSLS=(k={args.csls_k}, lambda={args.hub_penalty})")

    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Load cached stimulus features
    train_feat_path = os.path.join(args.feat_cache, "viclip_ast_train_all.npy")
    test_feat_path = os.path.join(args.feat_cache, "viclip_ast_test.npy")

    if not os.path.exists(train_feat_path):
        raise FileNotFoundError(
            f"Training features not found at {train_feat_path}. "
            "Please run Phase 1D extraction first."
        )
    if not os.path.exists(test_feat_path):
        raise FileNotFoundError(
            f"Test features not found at {test_feat_path}. "
            "Please run Phase 1D test extraction first."
        )

    print(f"Loading multimodal stimulus features from {args.feat_cache} ...")
    train_feats = np.load(train_feat_path)  # (1001, 1536)
    test_feats = np.load(test_feat_path)    # (102, 1536)
    print(f"  Train stimulus features shape: {train_feats.shape}")
    print(f"  Test stimulus features shape:  {test_feats.shape}")

    test_feats_tensor = torch.as_tensor(test_feats, dtype=torch.float32, device=args.device)

    # 2. Build multi-subject EEG dataset
    train_subjects = [int(s.strip()) for s in args.train_subjects.split(",") if s.strip()]
    dataset = EEGStimulusDataset(
        subjects=train_subjects,
        stim_feats=train_feats,
        trn_dir=args.trn_dir,
        augment=args.augment,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
        generator=g,
    )

    # 3. Pre-load test evaluation data for sub-01..04
    eval_subjects = [1, 2, 3, 4]
    test_cache = {}
    for sub in eval_subjects:
        try:
            sorted_eeg, sorted_gt = load_eval_test_data(sub, test_dir=args.tst_dir, gt_dir=args.gt_dir)
            test_cache[sub] = (
                torch.as_tensor(sorted_eeg, dtype=torch.float32),
                sorted_gt,
            )
        except Exception as e:
            print(f"  Warning: Could not load test set for sub-{sub:02d}: {e}")

    # 4. Instantiate contrastive model
    stim_dim = train_feats.shape[1]  # 1536
    model = MultimodalContrastiveModel(
        stim_dim=stim_dim,
        proj_dim=args.proj_dim,
        init_temp=args.init_temp,
        dropout=args.dropout,
        use_subject_adapter=args.use_adapter,
        num_subjects=6,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    print(f"\nModel initialized: EEGEncoder + StimulusProjector -> {args.proj_dim}-d joint space")
    print(f"  Subject Adapter: {args.use_adapter}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_avg_top1 = 0.0
    best_avg_top5 = 0.0
    best_epoch = 0

    print("\nStarting contrastive training ...")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for eeg_batch, stim_batch, stim_id_batch, sub_id_batch in dataloader:
            eeg_batch = eeg_batch.to(args.device)
            stim_batch = stim_batch.to(args.device)
            stim_ids = stim_id_batch.to(args.device)
            sub_ids = sub_id_batch.to(args.device) if args.use_adapter else None

            optimizer.zero_grad()
            loss, _ = model(eeg_batch, stim_batch, stim_ids=stim_ids, sub_ids=sub_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = total_loss / max(n_batches, 1)

        # Periodic evaluation on held-out test data
        if epoch % args.eval_interval == 0:
            sub_m, avg_m, _, _ = evaluate_all_test_subjects(
                model, test_cache, test_feats_tensor, args.device,
                anti_hub="none", use_adapter=args.use_adapter
            )

            current_top1 = avg_m["top1"]
            current_top5 = avg_m["top5"]
            current_median = avg_m["median_rank"]

            is_best = (current_top1 > best_avg_top1) or (current_top1 == best_avg_top1 and current_top5 > best_avg_top5)

            marker = "  [★ BEST]" if is_best else ""
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Test Avg Top-1: {current_top1:.4f} | "
                f"Top-5: {current_top5:.4f} | "
                f"Med Rank: {current_median:.1f}{marker}"
            )

            if is_best:
                best_avg_top1 = current_top1
                best_avg_top5 = current_top5
                best_epoch = epoch

                ckpt_path = os.path.join(args.model_dir, "phase2_best_contrastive.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "avg_metrics": avg_m,
                    "sub_metrics": sub_m,
                    "args": vars(args),
                }, ckpt_path)

    print("-" * 75)
    print(f"\nTraining Complete! Best Epoch: {best_epoch:02d}")
    print(f"  Best Test Avg Top-1: {best_avg_top1:.4f}  |  Avg Top-5: {best_avg_top5:.4f}")

    # Load and report full breakdown of best model
    best_ckpt = torch.load(os.path.join(args.model_dir, "phase2_best_contrastive.pt"), map_location=args.device)
    model.load_state_dict(best_ckpt["model_state"])

    print("\n" + "=" * 60)
    print("  Phase 2 Best Model — Raw Cosine Similarity")
    print("=" * 60)
    sub_m, avg_m, sub_sims, sub_ranks = evaluate_all_test_subjects(
        model, test_cache, test_feats_tensor, args.device,
        anti_hub="none", use_adapter=args.use_adapter
    )
    for sub, m in sub_m.items():
        print_metrics(m, label=f"sub-{sub:02d} [TEST]")
    print_metrics(avg_m, label="Phase 2 TEST Average (Raw)")

    # Save sub-01..sub-04 npy files (raw)
    video_ids = np.array(range(1001, 1103))
    for sub in sub_m.keys():
        np.save(os.path.join(args.model_dir, f"sub-{sub:02d}_ranks.npy"), sub_ranks[sub])
        np.save(os.path.join(args.model_dir, f"sub-{sub:02d}_similarity.npy"), sub_sims[sub])
        ranked_cand = video_ids[np.argsort(-sub_sims[sub], axis=1)]
        np.save(os.path.join(args.model_dir, f"sub-{sub:02d}_ranked_candidate_list.npy"), ranked_cand)

    print(f"\nSaved sub-01..04 .npy files to {args.model_dir}")

    # Also evaluate with CSLS anti-hubness
    print("\n" + "-" * 60)
    print(f"  With CSLS Anti-Hubness (k={args.csls_k}, lambda={args.hub_penalty}):")
    print("-" * 60)
    sub_m_csls, avg_m_csls, sims_csls, ranks_csls = evaluate_all_test_subjects(
        model, test_cache, test_feats_tensor, args.device,
        anti_hub="csls", csls_k=args.csls_k, hub_penalty=args.hub_penalty,
        use_adapter=args.use_adapter
    )
    for sub, m in sub_m_csls.items():
        print_metrics(m, label=f"sub-{sub:02d} [CSLS]")
    print_metrics(avg_m_csls, label="Phase 2 TEST Average (CSLS)")

    # Save CSLS version if requested
    if args.anti_hub == "csls":
        for sub in sub_m.keys():
            np.save(os.path.join(args.model_dir, f"sub-{sub:02d}_csls_ranks.npy"), ranks_csls[sub])
            np.save(os.path.join(args.model_dir, f"sub-{sub:02d}_csls_similarity.npy"), sims_csls[sub])

    # Save final model state
    final_path = os.path.join(args.model_dir, "phase2_final_contrastive.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved best model to:  {os.path.join(args.model_dir, 'phase2_best_contrastive.pt')}")
    print(f"Saved final model to: {final_path}")
    print("Done.")


if __name__ == "__main__":
    main()
