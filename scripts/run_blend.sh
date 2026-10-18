#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "======================================================================"
echo "  Running 2-Way Simplex Blend (Exp4 + V-JEPA 2)"
echo "======================================================================"

python experiments/blend_exp4_vjepa.py \
  --exp4-run experiments/exp4_tuned_mlp_dino_audio/runs/exp4_dino_clap_retrain \
  --vjepa-dir experiments/v-jepa/submissions \
  --gt-dir tst_stim \
  --output-dir submissions
