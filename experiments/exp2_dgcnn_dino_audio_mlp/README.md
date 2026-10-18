# DGCNN + frozen DINO/audio + ranking MLP

This experiment implements direct multimodal retrieval:

```text
EEG (17×55) -> DGCNN (55->64->32) -> MLP -> L2-normalized 256-D query
DINOv2 (1536) -> existing branch MLP --+
                                          +-> learned fusion -> L2-normalized 256-D candidate
Audio          -> existing branch MLP --+
score(query, candidate) = cosine similarity
loss = symmetric temperature-scaled InfoNCE
```

DINOv2 and the audio foundation model are frozen: training consumes their existing cached embeddings and never loads or updates the extractors. Trainable components are the DGCNN, EEG projection, per-modality MLPs, fusion weights, and temperature. Feature normalization is fit only on the fitting split. Validation holds out 102 training-video IDs; test labels are never loaded.

CLAP is the default audio block. AST and BEATs can be selected with `--audio-model`.

## Required files

In addition to EEG `trn/` data (and `tst/` with `--predict-test`):

```text
features_dinov2/consolidated_encoder.layer.11.npz
features_test_dinov2/consolidated_encoder.layer.11.npz
features_audio/train_clap_features.npz
features_audio/test_clap_features.npz
```

For AST or BEATs, replace `clap` with `ast` or `beats` in the audio cache names.

Code dependencies that must exist on the cluster:

```text
experiments/exp2_dgcnn_dino_audio_mlp/run_experiment.py
experiments/exp1_two_layer_gcn/run_experiment.py
src/metrics.py
src/models/contrastive_eeg.py
src/models/__init__.py
```

## Verify caches

Run from the repository root:

```bash
ls -lh \
  features_dinov2/consolidated_encoder.layer.11.npz \
  features_test_dinov2/consolidated_encoder.layer.11.npz \
  features_audio/train_clap_features.npz \
  features_audio/test_clap_features.npz
```

## Smoke test

```bash
python experiments/exp2_dgcnn_dino_audio_mlp/run_experiment.py \
  --subject 1 \
  --audio-model clap \
  --graph-k 4 \
  --device cuda \
  --smoke \
  --run-name dgcnn-dino-clap-sub-01-smoke-seed-42
```

## Full run

```bash
python experiments/exp2_dgcnn_dino_audio_mlp/run_experiment.py \
  --subject 1 \
  --audio-model clap \
  --graph-k 4 \
  --device cuda \
  --max-epochs 100 \
  --patience 20 \
  --batch-size 32 \
  --seed 42 \
  --hubness-mode none \
  --predict-test \
  --run-name dgcnn-dino-clap-sub-01-full-seed-42
```

## All subjects

```bash
for subject in 1 2 3 4 5 6; do
  printf -v sub "%02d" "$subject"
  python experiments/exp2_dgcnn_dino_audio_mlp/run_experiment.py \
    --subject "$subject" \
    --audio-model clap \
    --graph-k 4 \
    --device cuda \
    --max-epochs 100 \
    --patience 20 \
    --batch-size 32 \
    --seed 42 \
    --hubness-mode none \
    --predict-test \
    --run-name "dgcnn-dino-clap-sub-${sub}-full-seed-42"
done
```

Every run is isolated under `experiments/exp2_dgcnn_dino_audio_mlp/runs/<run-name>/`. It saves weights, configuration, train-only normalization, split IDs, history, metrics, embeddings, score matrices, rankings, and optional original-row-order test rankings.
