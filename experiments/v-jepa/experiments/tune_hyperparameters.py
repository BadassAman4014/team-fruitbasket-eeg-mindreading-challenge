"""Hyperparameter Tuning Pipeline on 900/100 Train-Val Split.

Trains contrastive models on stimuli 1–900 and evaluates on held-out stimuli 901–1000.
Tuning evaluates:
  1. InfoNCE batch size (64 vs 128)
  2. Subject Spatial Adapter (True vs False)
  3. Projection Dimension (256 vs 512)
  4. Initial Temperature (0.05 vs 0.07)
  5. Learning Rate (1e-4 vs 3e-4)
  6. Post-processing Anti-Hubness (CSLS k and lambda grid search)

Once the best configuration is identified on validation, it is saved to JSON
so it can be trained on the full 1000 stimuli for the final submission.

Usage
-----
    # Run full grid search:
    python experiments/tune_hyperparameters.py --device cuda --seed 42 --run_all

    # Or test a single specific configuration:
    python experiments/tune_hyperparameters.py --device cuda --seed 42 --batch_size 128 --use_adapter
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import EEGStimulusDataset, load_val_retrieval_data
from src.models.contrastive_eeg import MultimodalContrastiveModel
from src.metrics import compute_ranks, retrieval_metrics, print_metrics, apply_anti_hubness


def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive Hyperparameter Tuning")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for exact reproducibility")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per tuning run (peaks ~10-18)")
    parser.add_argument("--feat_cache", type=str, default="./cache/viclip_ast")
    parser.add_argument("--trn_dir", type=str, default="./trn")
    parser.add_argument("--out_dir", type=str, default="./models/tuning")
    parser.add_argument("--run_all", action="store_true", help="Run full tuning grid")

    # Single-run config overrides
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--init_temp", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--use_adapter", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_validation_retrieval(
    model: MultimodalContrastiveModel,
    val_data: dict,
    device: str,
    anti_hub: str = "none",
    csls_k: int = 10,
    hub_penalty: float = 1.0,
) -> dict:
    """Evaluate 100-way retrieval on held-out validation stimuli (901-1000)."""
    model.eval()
    val_eeg = val_data["val_eeg"].to(device)
    cand_feats = val_data["cand_feats"].to(device)
    val_subs = val_data["val_subs"].to(device)
    val_stims = val_data["val_stims"]          # (N_val,) true IDs in [901..1000]
    cand_ids = val_data["cand_ids"]            # [901, ..., 1000] (100 candidates)

    sim = model.compute_similarity(val_eeg, cand_feats, sub_ids=val_subs).cpu().numpy()  # (N_val, 100)

    if anti_hub != "none":
        sim = apply_anti_hubness(sim, method=anti_hub, k=csls_k, hub_penalty=hub_penalty)

    # For each query trial, find where the true stimulus ID ranks among the 100 candidates
    ranks = []
    for i, true_id in enumerate(val_stims):
        # Candidates are ordered by cand_ids [901..1000]
        # Column index of true_id is (true_id - 901)
        true_col = int(true_id - cand_ids[0])
        row_sims = sim[i]
        # Rank: how many candidates have higher similarity + 1
        rank = int(np.sum(row_sims > row_sims[true_col]) + 1)
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "median_rank": float(np.median(ranks)),
        "ranks": ranks,
        "sim": sim,
    }


def sweep_antihub_on_sim(sim: np.ndarray, val_stims: np.ndarray, cand_ids: np.ndarray) -> dict:
    """Instant grid search on validation similarity matrix for optimal CSLS k and lambda."""
    best_top1 = 0.0
    best_top5 = 0.0
    best_cfg = {"k": 10, "lambda": 1.0, "top1": 0.0, "top5": 0.0, "median": 50.0}

    k_values = [3, 5, 10, 15, 20]
    lambda_values = [0.25, 0.5, 0.75, 1.0, 1.25]

    for k in k_values:
        for lam in lambda_values:
            sim_adj = apply_anti_hubness(sim, method="csls", k=k, hub_penalty=lam)
            ranks = []
            for i, true_id in enumerate(val_stims):
                true_col = int(true_id - cand_ids[0])
                rank = int(np.sum(sim_adj[i] > sim_adj[i, true_col]) + 1)
                ranks.append(rank)
            ranks = np.array(ranks)
            top1 = float((ranks == 1).mean())
            top5 = float((ranks <= 5).mean())
            med = float(np.median(ranks))

            if (top1 > best_top1) or (top1 == best_top1 and top5 > best_top5):
                best_top1 = top1
                best_top5 = top5
                best_cfg = {"k": k, "lambda": lam, "top1": top1, "top5": top5, "median": med}

    return best_cfg


def train_single_config(cfg: Dict[str, Any], args: Any, train_feats: np.ndarray, val_data: dict) -> dict:
    """Train one configuration on stimuli 1-900, evaluate on 901-1000."""
    set_seed(args.seed)

    print("\n" + "=" * 70)
    print(f"  Testing Config: {cfg['name']}")
    print(f"  Params: Batch={cfg['batch_size']}, LR={cfg['lr']}, Proj={cfg['proj_dim']}, "
          f"Temp={cfg['init_temp']}, Adapter={cfg['use_adapter']}")
    print("=" * 70)

    # 1. Dataset on stimuli 1-900
    dataset = EEGStimulusDataset(
        subjects=[1, 2, 3, 4, 5, 6],
        stim_feats=train_feats,
        trn_dir=args.trn_dir,
        augment=True,
        stim_range=(1, 900),
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=g,
    )

    # 2. Model
    model = MultimodalContrastiveModel(
        stim_dim=train_feats.shape[1],
        proj_dim=cfg["proj_dim"],
        init_temp=cfg["init_temp"],
        dropout=cfg["dropout"],
        use_subject_adapter=cfg["use_adapter"],
        num_subjects=6,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_top1 = 0.0
    best_val_top5 = 0.0
    best_epoch = 0
    best_sim = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for eeg_batch, stim_batch, stim_id_batch, sub_id_batch in dataloader:
            eeg_batch = eeg_batch.to(args.device)
            stim_batch = stim_batch.to(args.device)
            stim_ids = stim_id_batch.to(args.device)
            sub_ids = sub_id_batch.to(args.device) if cfg["use_adapter"] else None

            optimizer.zero_grad()
            loss, _ = model(eeg_batch, stim_batch, stim_ids=stim_ids, sub_ids=sub_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = total_loss / max(n_batches, 1)

        # Validation evaluation on stimuli 901-1000
        val_res = evaluate_validation_retrieval(model, val_data, args.device, anti_hub="none")
        v_top1 = val_res["top1"]
        v_top5 = val_res["top5"]
        v_med = val_res["median_rank"]

        is_best = (v_top1 > best_val_top1) or (v_top1 == best_val_top1 and v_top5 > best_val_top5)
        marker = "  [★ BEST]" if is_best else ""
        print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Loss: {epoch_loss:.4f} | "
              f"Val Top-1: {v_top1*100:.2f}% | Top-5: {v_top5*100:.2f}% | Med Rank: {v_med:.1f}{marker}")

        if is_best:
            best_val_top1 = v_top1
            best_val_top5 = v_top5
            best_epoch = epoch
            best_sim = val_res["sim"]

    # Post-hoc CSLS sweep on best validation similarity matrix
    csls_opt = sweep_antihub_on_sim(best_sim, val_data["val_stims"], val_data["cand_ids"])
    print(f"\n  Optimal CSLS Post-Processing: k={csls_opt['k']}, lambda={csls_opt['lambda']}")
    print(f"  Val Top-1 (Raw): {best_val_top1*100:.2f}%  ->  Val Top-1 (CSLS): {csls_opt['top1']*100:.2f}%")
    print(f"  Val Top-5 (Raw): {best_val_top5*100:.2f}%  ->  Val Top-5 (CSLS): {csls_opt['top5']*100:.2f}%")

    result = {
        "name": cfg["name"],
        "config": cfg,
        "best_epoch": best_epoch,
        "val_top1_raw": best_val_top1,
        "val_top5_raw": best_val_top5,
        "val_top1_csls": csls_opt["top1"],
        "val_top5_csls": csls_opt["top5"],
        "val_median_csls": csls_opt["median"],
        "csls_k": csls_opt["k"],
        "csls_lambda": csls_opt["lambda"],
    }
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  Contrastive Hyperparameter Tuning (900/100 Train-Val Split)")
    print(f"  Device: {args.device} | Seed: {args.seed} | Max Epochs: {args.epochs}")
    print("=" * 70)

    # 1. Load train stimulus features
    train_feat_path = os.path.join(args.feat_cache, "viclip_ast_train_all.npy")
    if not os.path.exists(train_feat_path):
        raise FileNotFoundError(f"Missing {train_feat_path}")
    train_feats = np.load(train_feat_path)

    # 2. Pre-load validation set (stimuli 901-1000)
    print("\nLoading held-out validation set (stimuli 901-1000 across all subjects) ...")
    val_data = load_val_retrieval_data(
        subjects=[1, 2, 3, 4, 5, 6],
        stim_feats=train_feats,
        trn_dir=args.trn_dir,
        val_range=(901, 1000),
    )
    print(f"Validation set ready: {len(val_data['val_eeg'])} trials evaluating on 100 candidate stimuli.")

    # 3. Define Grid of Configurations
    if args.run_all:
        configs = [
            {
                "name": "Config_1_Baseline",
                "batch_size": 64,
                "lr": 3e-4,
                "proj_dim": 256,
                "init_temp": 0.07,
                "dropout": 0.25,
                "use_adapter": False,
            },
            {
                "name": "Config_2_Batch128",
                "batch_size": 128,
                "lr": 3e-4,
                "proj_dim": 256,
                "init_temp": 0.07,
                "dropout": 0.25,
                "use_adapter": False,
            },
            {
                "name": "Config_3_SubjectAdapter",
                "batch_size": 64,
                "lr": 3e-4,
                "proj_dim": 256,
                "init_temp": 0.07,
                "dropout": 0.25,
                "use_adapter": True,
            },
            {
                "name": "Config_4_Batch128_Adapter",
                "batch_size": 128,
                "lr": 3e-4,
                "proj_dim": 256,
                "init_temp": 0.07,
                "dropout": 0.25,
                "use_adapter": True,
            },
            {
                "name": "Config_5_Proj512",
                "batch_size": 64,
                "lr": 3e-4,
                "proj_dim": 512,
                "init_temp": 0.07,
                "dropout": 0.25,
                "use_adapter": False,
            },
            {
                "name": "Config_6_Temp05",
                "batch_size": 64,
                "lr": 3e-4,
                "proj_dim": 256,
                "init_temp": 0.05,
                "dropout": 0.25,
                "use_adapter": False,
            },
        ]
    else:
        configs = [{
            "name": "Custom_Config",
            "batch_size": args.batch_size,
            "lr": args.lr,
            "proj_dim": args.proj_dim,
            "init_temp": args.init_temp,
            "dropout": args.dropout,
            "use_adapter": args.use_adapter,
        }]

    # 4. Run tuning
    all_results = []
    for cfg in configs:
        res = train_single_config(cfg, args, train_feats, val_data)
        all_results.append(res)

    # 5. Summary Leaderboard
    all_results.sort(key=lambda x: (x["val_top1_csls"], x["val_top5_csls"]), reverse=True)

    print("\n" + "=" * 80)
    print("  HYPERPARAMETER TUNING LEADERBOARD (Held-Out 100-Stimulus Validation Split)")
    print("=" * 80)
    print(f"{'Config Name':<25} | {'Val Top-1':<10} | {'Val Top-5':<10} | {'Med Rank':<9} | {'Best Epoch':<10} | {'CSLS (k, lam)'}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['name']:<25} | {r['val_top1_csls']*100:6.2f}%   | {r['val_top5_csls']*100:6.2f}%   | {r['val_median_csls']:<9.1f} | {r['best_epoch']:<10} | ({r['csls_k']}, {r['csls_lambda']})")
    print("=" * 80)

    best_overall = all_results[0]
    print(f"\n🏆 WINNER: {best_overall['name']}")
    print(f"  Configuration: {best_overall['config']}")
    print(f"  Recommended Stopping Epoch on Full Data: {best_overall['best_epoch']}")
    print(f"  Optimal CSLS: k={best_overall['csls_k']}, lambda={best_overall['csls_lambda']}")

    # Save to JSON
    json_path = os.path.join(args.out_dir, "best_hparams.json")
    with open(json_path, "w") as f:
        json.dump(best_overall, f, indent=2)
    print(f"\nSaved winning recipe to: {json_path}")


if __name__ == "__main__":
    main()
