# Exp4: tuned MLP with frame-level DINO + audio

This is an isolated controlled ablation of `exp3_simple_mlp_dino_audio`. It reads frozen consolidated DINO frame caches plus frozen CLAP or AST caches; it never extracts or updates either foundation model. It does not write to exp1, exp2, or exp3.

## Fixed protocol

- Training subjects default to **1,2,3,4**, pooled across subjects.
- EEG is FIR-decimated to 100 Hz by the repository loader, sliced at indices **`5:60`** (17 × 55), and normalized independently per trial over channel and time.
- Stimulus IDs **1–900** train the model; **901–1000** alone select each seed checkpoint and the optional mild-CSLS lambda.
- Seeds default to **42,123,999** and raw cosine score matrices are averaged before CSLS.
- Test candidates are IDs **1001–1102**. Test rows are never reordered.
- `--prediction-subjects` is independent of `--subjects`, so training on 1–4 can still produce rankings for 1–6. Labels for subjects 5/6 are structurally excluded. Public labels for prediction subjects 1–4 are optional reporting only via `--report-public-subjects`, after every selection decision.
- Every run uses a new directory and refuses to overwrite an existing one. Submission copies are created only under that run directory.

The DINO loader requires exactly the configured frame IDs (default `f00`…`f07`) for every required stimulus. Train-only feature normalization is fitted on IDs 1–900. In attention mode, all normalized frozen frame vectors remain available to a trainable content+position attention pooler; DINO itself remains cached and frozen.

## Main controls

```text
--hidden-dim 512 --embedding-dim 256 --lr 3e-4 --weight-decay 1e-2
--dropout 0.3 --temperature 0.07 --batch-size 128
--loss {standard,multipositive}             # default: multipositive
--fusion {concat,separate}                  # default: separate
--subject-affine --subject-affine-reg 1e-4  # default: disabled
--dino-pooling {mean,attention}             # default: mean
--modality-dropout 0.15 --dino-frames 8 --attention-hidden-dim 64
--audio-model {clap,ast}
```

`standard` matches exp3’s diagonal symmetric InfoNCE while masking duplicate-video false negatives. `multipositive` uses every same-video item in the batch as a positive in both retrieval directions via masked `logsumexp`; singleton video IDs are valid. `concat` applies one MLP to pooled DINO plus audio. `separate` applies independent projection MLPs, learned global softmax fusion, and per-sample modality dropout. Subject affine calibration, when enabled, is a learned `(6,17,1)` scale/bias initialized to identity/zero and applied after trial normalization.

## Validation without caches

```bash
python experiments/exp4_tuned_mlp_dino_audio/train.py --help
python experiments/exp4_tuned_mlp_dino_audio/train.py --self-test
```

`--self-test` runs CPU forward/backward checks for concat/standard/mean and separate/multipositive/attention with subject affine, including duplicate and singleton positives. It loads no EEG or feature cache.

## Cluster smoke run

The following generic Slurm command assumes the repository environment is already activated and the consolidated DINO/audio caches are available at their default paths:

```bash
sbatch --job-name=exp4-smoke --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 \
  --wrap="python experiments/exp4_tuned_mlp_dino_audio/train.py --run-name smoke-separate-mp-mean --subjects 1,2,3,4 --seeds 42 --epochs 2 --smoke --device cuda"
```

Add `--predict-test --prediction-subjects 1,2,3,4,5,6` to generate test rankings. Add `--report-public-subjects` only when reporting public subjects 1–4 is wanted.

## Ordered controlled ablations

Run these in order. Each command changes one design choice relative to the preceding run and has a distinct output directory. Keep all other hyperparameters fixed.

```bash
# A. Baseline: exp3-style concat + standard InfoNCE + mean DINO pooling
sbatch --job-name=e4-a --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=06:00:00 \
  --wrap="python experiments/exp4_tuned_mlp_dino_audio/train.py --run-name a-concat-standard-mean --fusion concat --loss standard --dino-pooling mean --device cuda"

# B. Replace only the loss with symmetric multi-positive InfoNCE
sbatch --job-name=e4-b --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=06:00:00 \
  --wrap="python experiments/exp4_tuned_mlp_dino_audio/train.py --run-name b-concat-multipositive-mean --fusion concat --loss multipositive --dino-pooling mean --device cuda"

# C. Replace concat with independent learned DINO/audio fusion
sbatch --job-name=e4-c --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=06:00:00 \
  --wrap="python experiments/exp4_tuned_mlp_dino_audio/train.py --run-name c-separate-multipositive-mean --fusion separate --loss multipositive --dino-pooling mean --device cuda"

# D. Add per-subject affine calibration
sbatch --job-name=e4-d --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=06:00:00 \
  --wrap="python experiments/exp4_tuned_mlp_dino_audio/train.py --run-name d-separate-mp-affine-mean --fusion separate --loss multipositive --subject-affine --dino-pooling mean --device cuda"

# E. Replace mean pooling with eight-frame content+position attention
sbatch --job-name=e4-e --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=06:00:00 \
  --wrap="python experiments/exp4_tuned_mlp_dino_audio/train.py --run-name e-separate-mp-affine-attention --fusion separate --loss multipositive --subject-affine --dino-pooling attention --device cuda"
```

For final prediction, append `--predict-test --prediction-subjects 1,2,3,4,5,6`; optionally append `--report-public-subjects`. To use AST consistently, add `--audio-model ast` to every command.

## Run artifacts

Each fresh run contains `config.json`, `normalization.npz`, `split.npz`, one `model_seed-*.pt` per seed, `history.json`, `heldout_outputs.npz`, `metrics.json`, `model_metadata.json`, and `safeguards.json`. Prediction runs additionally contain per-subject test NPZ files with raw/adjusted scores and actual candidate-ID rankings, plus challenge-format `.npy` copies under that run’s `submissions/` directory. The metadata records selected lambda, fusion weights, attention details, split/query subjects, cache-freezing declarations, and label-leakage safeguards.
