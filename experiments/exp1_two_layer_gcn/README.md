# Experiment 1 — Two-layer GCN

This directory contains an isolated EEG-to-video retrieval experiment. It does not modify the original checkpoints, predictions, or submission files.

## Data and model alignment

For one subject, raw EEG is decimated from 500 Hz to 100 Hz and cropped to indices `5:60` (the organizer's 50–600 ms window), producing `(trials, 17 electrodes, 55 time features)`. Per-channel EEG statistics are fit only on the fitting split. The GCN is `55 → 64 → 32`, with ReLU, dropout 0.5 after layer 1, ReLU after layer 2, and global mean pooling over the 17 nodes. A linear regression head maps the pooled 32-D EEG embedding to the frozen video-feature dimension; MSE is used for training.

The default target is `cache/stim/vis_alexnet_k256.npz`. Its explicit `ids` array is used for every lookup, so feature rows are never assumed to match EEG rows implicitly. Candidate scoring follows the repository's existing convention: row-wise standardization, correlation matrix multiplication, descending score sort, and lookup of actual candidate video IDs through `src.metrics`.

No complete electrode names, montage, or coordinates exist in this repository. The default fallback graph is therefore a fixed binary channel-correlation k-NN graph computed from the fitting EEG split only; it does not claim anatomical distances or invent coordinates. `--graph identity` is available as a topology-free control. Both add self-loops and use symmetric GCN normalization `D^-1/2 (A+I) D^-1/2`.

Validation holds out exactly 102 video IDs from training data. Early stopping uses held-out retrieval MRR (then Top-5 and median rank for ties). Test labels are never loaded. The PCA + Ridge comparison uses exactly the same fitting trials, validation trials, and candidate IDs, following the existing baseline direction of frozen video features to measured EEG.

## Dependencies

Use an environment containing Python 3.10+, NumPy, SciPy, scikit-learn, joblib, and PyTorch. Install these according to your cluster's CUDA/module setup; do not replace a cluster-provided CUDA build of PyTorch with a generic wheel.

## Commands

Run from the repository root.

### 1. Verify imports

```bash
python -c "import torch,numpy,scipy,sklearn,joblib; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Smoke test first

The smoke run verifies `(B,17,55) → (B,17,64) → (B,17,32) → (B,32) → (B,256)`, runs two epochs on a reduced fitting/query set, still ranks the full 102 validation candidates, and fits the matching PCA + Ridge baseline.

```bash
python experiments/exp1_two_layer_gcn/run_experiment.py \
  --subject 1 \
  --device cuda \
  --smoke \
  --run-name sub-01-smoke-seed-42
```

### 3. Full experiment for one subject

```bash
python experiments/exp1_two_layer_gcn/run_experiment.py \
  --subject 1 \
  --device cuda \
  --max-epochs 200 \
  --patience 20 \
  --batch-size 32 \
  --seed 42 \
  --run-name sub-01-full-seed-42
```

Add `--predict-test` if you also want a `(102,102)` ranking of test rows against video IDs 1001–1102. This remains inside the run directory and never overwrites an official submission:

```bash
python experiments/exp1_two_layer_gcn/run_experiment.py \
  --subject 5 \
  --device cuda \
  --max-epochs 200 \
  --patience 20 \
  --batch-size 32 \
  --seed 42 \
  --predict-test \
  --run-name sub-05-full-seed-42
```

### 4. Run every subject sequentially

```bash
for subject in 1 2 3 4 5 6; do
  printf -v sub "%02d" "$subject"
  python experiments/exp1_two_layer_gcn/run_experiment.py \
    --subject "$subject" \
    --device cuda \
    --max-epochs 200 \
    --patience 20 \
    --batch-size 32 \
    --seed 42 \
    --predict-test \
    --run-name "sub-${sub}-full-seed-42"
done
```

For a SLURM array job, map the array index to `--subject "$SLURM_ARRAY_TASK_ID"` and give each task a unique run name.

## Outputs

Each run is saved below `experiments/exp1_two_layer_gcn/runs/<run-name>/`. Existing directories are never overwritten. Files include:

- `model.pt`: GCN weights, architecture configuration, and best epoch
- `config.json`, `history.json`, `metrics.json`: seed, settings, epoch history, train/validation metrics, and safeguards
- `normalization.npz`: train-only EEG and video-feature statistics
- `adjacency.npy`, `adjacency_normalized.npy`: fallback graph and normalized GCN graph
- `split.npz`: actual fitting/validation video and query IDs
- `gcn_outputs.npz`: scores, rankings, query/candidate IDs, and 32-D embeddings
- `pca_ridge_baseline.joblib`, `pca_ridge_outputs.npz`: matched baseline and outputs
- optional `gcn_test_outputs.npz` and `sub-XX_gcn_ranked_candidate_list.npy`: original test-row-order rankings using actual IDs 1001–1102
