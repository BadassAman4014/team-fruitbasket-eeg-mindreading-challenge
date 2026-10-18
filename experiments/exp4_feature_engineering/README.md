# Exp4 + Feature Engineering & Electrode Selection

This experiment extends **Exp4 (Tuned MLP DINO + CLAP Audio)** with:
1. **Electrode Selection & Pruning**:
   - Train on any subset of electrodes using `--channels <list|preset>`
   - Presets: `all17`, `core8` (`Oz, P8, PO8, O2, PO7, Pz, P7, P2`), `top12`, `occipital`, `parietal`
2. **EEG Feature Engineering**:
   - `--eeg-features raw`: Standard flattened time series voltages (default Exp4 baseline)
   - `--eeg-features bandpower`: Log spectral power in Delta (0.5–4Hz), Theta (4–8Hz), Alpha (8–13Hz), Beta (13–30Hz)
   - `--eeg-features both`: Concat of raw voltages + spectral band power
   - `--eeg-features erp_stats`: Raw voltages + ERP temporal window statistics (P100, N170, P300, Late visual window)
   - `--eeg-features full`: Concat of raw voltages + band-power + ERP statistics
3. **Electrode & Feature Importance Analyzer**:
   - Run `feature_importance.py` on any trained checkpoint to measure Permutation Feature Importance (PFI), Zero-out ablation, Gradient Saliency, and Temporal Window importance.
   - Or pass `--auto-importance` to `train.py` to print the leaderboard automatically after training.

---

## Usage Examples

### 1. Run Feature Importance on Existing Trained Exp4 Checkpoint
To discover which electrodes and time windows matter most for the Exp4 Tuned MLP:
```bash
python experiments/exp4_feature_engineering/feature_importance.py \
  --run-dir experiments/exp4_tuned_mlp_dino_audio/runs/exp6-source-all6-fullfit-20261018-155128 \
  --device cuda
```

### 2. Train Exp4 on Core-8 Visual Cortex Electrodes
```bash
RUN="exp4-fe-core8-$(date +%Y%m%d-%H%M%S)"

python experiments/exp4_feature_engineering/train.py \
  --run-name "$RUN" \
  --channels core8 \
  --eeg-features raw \
  --subjects 1,2,3,4,5,6 \
  --prediction-subjects 1,2,3,4,5,6 \
  --seeds 42,123,999 \
  --audio-model clap \
  --fusion concat \
  --loss multipositive \
  --dino-pooling mean \
  --hidden-dim 512 --embedding-dim 256 \
  --lr 3e-4 --weight-decay 0.01 --dropout 0.3 --temperature 0.07 \
  --batch-size 128 --epochs 30 \
  --anti-hub csls --csls-lambda auto --csls-k 10 \
  --device cuda --predict-test --report-public-subjects --final-full-fit
```

### 3. Train Exp4 with Engineered Spectral Band-Power Features
```bash
RUN="exp4-fe-bandpower-$(date +%Y%m%d-%H%M%S)"

python experiments/exp4_feature_engineering/train.py \
  --run-name "$RUN" \
  --channels all17 \
  --eeg-features both \
  --subjects 1,2,3,4,5,6 \
  --prediction-subjects 1,2,3,4,5,6 \
  --seeds 42,123,999 \
  --audio-model clap \
  --fusion concat \
  --loss multipositive \
  --dino-pooling mean \
  --hidden-dim 512 --embedding-dim 256 \
  --lr 3e-4 --weight-decay 0.01 --dropout 0.3 --temperature 0.07 \
  --batch-size 128 --epochs 30 \
  --anti-hub csls --csls-lambda auto --csls-k 10 \
  --device cuda --predict-test --report-public-subjects --final-full-fit
```

### 4. Self-Test
Verify all shapes, forward/backward passes, and feature engineering transformations:
```bash
python experiments/exp4_feature_engineering/train.py --self-test
python experiments/exp4_feature_engineering/feature_importance.py --self-test
```
