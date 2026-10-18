"""Evaluate test predictions and report metrics for a finished exp4 run.

Usage:
    python experiments/exp4_tuned_mlp_dino_audio/evaluate_test.py --run-dir experiments/exp4_tuned_mlp_dino_audio/runs/exp4_dinov2_video_only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp4_tuned_mlp_dino_audio.train import (
    Config, TunedMLPRetriever,
    load_dino_frames, load_test_eeg, normalize_trials,
    stack_dino_frames, stack_features, load_audio_map,
    apply_normalizer, encode_stimulus, encode_eeg,
    mild_csls, ranked_from_scores, metrics_from_scores,
    TEST_IDS, PUBLIC_SUBJECTS, WINDOW, N_CHANNELS
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate test set on finished exp4 run")
    p.add_argument("--run-dir", type=Path, required=True, help="Path to run directory")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    with open(run_dir / "config.json", "r") as f:
        cfg_dict = json.load(f)
    config = Config(**cfg_dict)

    device = torch.device(args.device)
    data_root = Path(config.data_root)

    norm = np.load(run_dir / "normalization.npz")
    dino_mean = norm["dino_mean"]
    dino_std = norm["dino_std"]
    audio_mean = norm["audio_mean"]
    audio_std = norm["audio_std"]

    heldout = np.load(run_dir / "heldout_outputs.npz")
    selected_lambda = float(heldout["selected_csls_lambda"])

    dino_test_map = load_dino_frames(
        data_root / config.dino_test_cache, TEST_IDS, config.dino_frames
    )
    test_dino_raw = stack_dino_frames(dino_test_map, TEST_IDS, "test DINO")
    test_dino = apply_normalizer(
        test_dino_raw, dino_mean[:, None, :], dino_std[:, None, :]
    )

    if config.audio_model != "none":
        audio_test_map = load_audio_map(
            data_root / "features_audio" / f"test_{config.audio_model}_features.npz"
        )
        test_audio_raw = stack_features(audio_test_map, TEST_IDS, "test audio")
        test_audio = apply_normalizer(test_audio_raw, audio_mean, audio_std)
        audio_dim = test_audio.shape[1]
    else:
        test_audio = np.zeros((len(TEST_IDS), 0), dtype=np.float32)
        audio_dim = 0

    dino_dim = test_dino.shape[2]

    subject_eeg = {}
    for subject in config.prediction_subjects:
        rows = load_test_eeg(data_root, subject)[:, :, WINDOW]
        subject_eeg[subject] = normalize_trials(rows)

    score_sums = {
        subject: np.zeros((len(subject_eeg[subject]), len(TEST_IDS)), dtype=np.float32)
        for subject in config.prediction_subjects
    }

    for seed in config.seeds:
        fullfit_path = run_dir / f"model_seed-{seed}_fullfit.pt"
        ckpt_path = fullfit_path if fullfit_path.exists() else run_dir / f"model_seed-{seed}.pt"
        print(f"Loading checkpoint for seed {seed} from {ckpt_path.name} ...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = TunedMLPRetriever(
            eeg_dim=N_CHANNELS * (WINDOW.stop - WINDOW.start),
            dino_dim=dino_dim,
            audio_dim=audio_dim,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            temperature=config.temperature,
            fusion=config.fusion,
            modality_dropout=config.modality_dropout,
            subject_affine=config.subject_affine,
            dino_pooling=config.dino_pooling,
            dino_frames=config.dino_frames,
            attention_hidden_dim=config.attention_hidden_dim,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        cand_emb = encode_stimulus(model, test_dino, test_audio, device, config.batch_size)
        for subject in config.prediction_subjects:
            s_ids = np.full(len(subject_eeg[subject]), subject, dtype=np.int64)
            q_emb = encode_eeg(model, subject_eeg[subject], s_ids, device, config.batch_size)
            score_sums[subject] += q_emb @ cand_emb.T

    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    public_metrics = {}
    print("\n" + "=" * 70)
    print(f"  Exp4 Video-Only Test Evaluation (CSLS lambda={selected_lambda})")
    print("=" * 70)
    print(f"  {'Subject':<10} | {'Top-1':<18} | {'Top-5':<18} | {'Median':<6}")
    print("-" * 70)

    for subject in config.prediction_subjects:
        raw = (score_sums[subject] / len(config.seeds)).astype(np.float32)
        adjusted = mild_csls(raw, config.csls_k, selected_lambda) if config.anti_hub == "csls" else raw.copy()
        rankings = ranked_from_scores(adjusted, TEST_IDS)

        np.savez_compressed(
            run_dir / f"sub-{subject:02d}_test_outputs.npz",
            test_row_indices=np.arange(len(raw), dtype=np.int64),
            candidate_ids=TEST_IDS,
            raw_scores=raw,
            adjusted_scores=adjusted,
            rankings=rankings,
            selected_csls_lambda=np.float32(selected_lambda),
        )

        np.save(submissions_dir / f"tuned_mlp_dino_{config.audio_model}_sub-{subject:02d}_ranked_candidate_list.npy", rankings)
        if subject in (5, 6):
            np.save(submissions_dir / f"sub-{subject:02d}_ranked_candidate_list.npy", rankings)

        if subject in PUBLIC_SUBJECTS:
            truth_path = data_root / "tst_stim" / f"sub-{subject:02d}_tst_stim.npy"
            if truth_path.exists():
                truth = np.asarray(np.load(truth_path), dtype=np.int64)
                m = metrics_from_scores(adjusted, TEST_IDS, truth)
                public_metrics[subject] = m
                t1_cnt = int(round(m["top1"] * len(truth)))
                t5_cnt = int(round(m["top5"] * len(truth)))
                print(f"  sub-{subject:02d}    | {m['top1']*100:5.2f}% ({t1_cnt:2d}/{len(truth)}) | {m['top5']*100:5.2f}% ({t5_cnt:2d}/{len(truth)}) | {m['median_rank']:4.1f}")

    if public_metrics:
        avg_top1 = np.mean([m["top1"] for m in public_metrics.values()]) * 100
        avg_top5 = np.mean([m["top5"] for m in public_metrics.values()]) * 100
        avg_med = np.mean([m["median_rank"] for m in public_metrics.values()])
        total_hits = sum(int(round(m["top1"] * 102)) for m in public_metrics.values())
        print("-" * 70)
        print(f"  AVERAGE    | {avg_top1:5.2f}% ({total_hits:2d}/408) | {avg_top5:5.2f}%      | {avg_med:4.1f}")
        print(f"  CHANCE     |  0.98% ( 4/408) |  4.90% (20/408)  | 51.5")
    print("=" * 70)
    print(f"\nAll test deliverables saved to: {submissions_dir}")


if __name__ == "__main__":
    main()
