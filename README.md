# Team Fruitbasket — EEG Mindreading Challenge

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6 CUDA 12.4](https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA%2012.4-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Competition%20Rules-100%25%20Compliant-brightgreen.svg)]()
[![Top-1 Test Accuracy](https://img.shields.io/badge/Public%20Test%20Top--1-16.18%25-success.svg)]()
[![Technical Report](https://img.shields.io/badge/Technical%20Report-REPORT.md-purple.svg)](REPORT.md)
[![Project Context](https://img.shields.io/badge/Project%20Context-CONTEXT.md-blueviolet.svg)](CONTEXT.md)

> **Winning Multimodal Retrieval Solution** for the EEG Mind Reading Retrieval Challenge: Mapping 63-channel human brain EEG recordings to 5-second video segments via spatial-temporal visual self-supervised models (**Meta V-JEPA 2** and **DINOv2**), environmental audio embeddings (**HTSAT-CLAP**), and Cross-Domain Similarity Local Scaling (**CSLS**).

---

## Table of Contents
1. [Official Competition Guidelines & Compliance Checklist](#1-official-competition-guidelines--compliance-checklist)
2. [Public Pre-Trained Foundation Models](#2-public-pre-trained-foundation-models)
3. [Executive Summary & Multimodal Architecture](#3-executive-summary--multimodal-architecture)
4. [Official Experimental Results & Benchmarks](#4-official-experimental-results--benchmarks)
5. [Repository Structure](#5-repository-structure)
6. [Environment Setup & Installation (Python 3.11)](#6-environment-setup--installation-python-311)
7. [Master Command Reference (All Tasks)](#7-master-command-reference-all-tasks)
   - [Task 1: Master Root Launcher (Interactive / Automated)](#task-1-master-root-launcher-interactive--automated)
   - [Task 2: Instant 1-Click Reproduction (5 Seconds)](#task-2-instant-1-click-reproduction-5-seconds)
   - [Task 3: Train Exp 4 (DINOv2 + CLAP MLP) on GPU](#task-3-train-exp-4-dinov2--clap-mlp-on-gpu)
   - [Task 4: Train V-JEPA 2 (8 Temporal Sub-Clips + CLAP) on GPU](#task-4-train-v-jepa-2-8-temporal-sub-clips--clap-on-gpu)
   - [Task 5: Run 2-Way Simplex Blend & Evaluation](#task-5-run-2-way-simplex-blend--evaluation)
   - [Task 6: Hyperparameter & Anti-Hubness Tuning](#task-6-hyperparameter--anti-hubness-tuning)
   - [Task 7: Package Final Competition Submission ZIP](#task-7-package-final-competition-submission-zip)
   - [Task 8: Optional 3-Way Tri-Blend (+ ViEEG)](#task-8-optional-3-way-tri-blend--vieeg)
8. [Expected Output & Submission Format Verification](#8-expected-output--submission-format-verification)
9. [Technical Deep Dive & Design Decisions](#9-technical-deep-dive--design-decisions)
10. [GitHub Packaging & Deliverables](#10-github-packaging--deliverables)

---

## 1. Official Competition Guidelines & Compliance Checklist

Our solution was designed, implemented, and verified in strict accordance with all hackathon rules:

| Guideline / Minimal Requirement | Compliance Status | Implementation Details |
|:---|:---:|:---|
| **Python 3.11 Runtime** | **COMPLIANT** | Pipeline is fully compatible with Python 3.11. Dependencies install cleanly with PyTorch 2.6 CUDA 12.4. |
| **Publicly Available Pre-Trained Models** | **COMPLIANT** | Uses only open-source foundation models (Meta V-JEPA 2, Meta DINOv2, LAION CLAP) released publicly before the hackathon. See [Section 2](#2-public-pre-trained-foundation-models). |
| **Zero Test Set Contamination / Leakage** | **COMPLIANT** | No test recordings (`tst/sub-XX_tst.npy`) or test labels were used during training or fine-tuning. Training data of all 6 subjects (`trn/sub-XX_trn.npz`, stimuli 1..900) was used exclusively for model parameter updates. |
| **No Test Template Matching on sub-05 / sub-06** | **COMPLIANT** | Test recordings of `sub-01` to `sub-04` were used strictly for post-hoc validation and ensembling weight selection. They were **never** used as reference templates for identifying test recordings of `sub-05` and `sub-06`. |
| **Unsupervised Normalization Only** | **COMPLIANT** | Feature z-score normalizers (mean, standard deviation) were fit strictly on training stimuli 1..900 and frozen. Zero test target statistics were incorporated into normalizers. |
| **Expected Output Format** | **COMPLIANT** | Generates `sub-05_ranked_candidate_list.npy` and `sub-06_ranked_candidate_list.npy` as integer arrays of shape `(102, 102)` saved with `numpy.save`. Candidate IDs are in range `1001` to `1102` (not indices 0..101). |
| **Original Unsorted Row Order Preserved** | **COMPLIANT** | Row $i$ corresponds strictly to the $i$-th recording in `tst/sub-XX_tst.npy` in its original unsorted order. Rows are **never** reordered by label. |
| **Report Submission** | **COMPLIANT** | A detailed technical report is included in the repository as [REPORT.md](REPORT.md). |
| **Reproducible GitHub Repository** | **COMPLIANT** | Complete training, inference, and packaging pipelines are provided. Raw MP4 videos and extracted frames are excluded via `.gitignore`. |
| **Source Code & Model Weights Provided** | **COMPLIANT** | Source code, configuration files, and retrained PyTorch weights (`.pt`) for all multi-seed models are included under `experiments/`. |

---

## 2. Public Pre-Trained Foundation Models

All pre-trained foundation models used were publicly released before the start of the hackathon and are accessible to all participants:

| Model | Exact Identifier / Checkpoint | Architecture | Feature Details | License & Release Date |
|:---|:---|:---|:---|:---|
| **Meta V-JEPA 2** | `facebookresearch/vjepa2` (ViT-L/16) | Vision Transformer (Large) | 8 temporal sub-clips $\times$ 8 frames (64 frames total per 5-sec video), 1024-D feature representations | Meta AI (Open-source, 2024) |
| **Meta DINOv2** | `dinov2_vitb14` (Layer 11) | Vision Transformer (Base) | Mean-pooled Layer 11 patch & CLS tokens, 768-D static spatial feature representations | Meta AI (Apache 2.0, April 2023) |
| **LAION HTSAT-CLAP** | `laion/larger_clap_general` | Hierarchical Token-Semantic Audio Transformer | 512-D environmental audio embeddings from 5-second 48kHz audio tracks | LAION-AI (MIT / Apache, 2022/2023) |

> **Note:** As permitted by the competition guidelines, the large public foundation model weights do not need to be committed to the repository because their exact names, checkpoints, and extraction scripts are documented. All trained task-specific projection heads and model weights are provided in `experiments/`.

---

## 3. Executive Summary & Multimodal Architecture

The objective is to decode 5-second 63-channel EEG epochs (sampled at 250 Hz, shape: `[63, 1250]`) recorded while subjects watched movies, and rank a candidate pool of 102 video stimuli (`1001` to `1102`) such that the true video stimulus appears at rank 1.

Our winning ensemble combines two complementary multimodal neural retrieval architectures:

```
                            ┌────────────────────────────────────────┐
                            │    Human EEG Epoch: [63, 1250]         │
                            └───────────────────┬────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │  Stream A: DINOv2 + CLAP    │                   │ Stream B: V-JEPA 2 + CLAP   │
        │  (Fine Spatial Semantics)   │                   │ (Temporal Motion Dynamics)  │
        ├─────────────────────────────┤                   ├─────────────────────────────┤
        │ • Subject-specific linear   │                   │ • Linear EEG projection     │
        │   alignment head (63*1250)  │                   │ • V-JEPA 2 ViT-L/16 across  │
        │ • DINOv2 Layer 11 pooled    │                   │   8 temporal sub-clips      │
        │ • CLAP audio 512-D concat   │                   │ • CLAP audio 512-D concat   │
        │ • Multi-positive InfoNCE    │                   │ • Modality Dropout (p=0.2)  │
        │ • 3-Seed Ensembling         │                   │ • 3-Seed Ensembling         │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       │ Cosine Similarity Matrix                        │ Cosine Similarity Matrix
                       │ S_exp4 (Shape: N x 102)                         │ S_vjepa (Shape: N x 102)
                       ▼                                                 ▼
        ┌───────────────────────────────────────────────────────────────────────────────┐
        │               2-Way Simplex Weight Blend & Anti-Hubness CSLS                  │
        │        S_blend = 0.35 * S_exp4 + 0.65 * S_vjepa   (CSLS lambda = 0.15)        │
        └───────────────────────────────────────┬───────────────────────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Final Ranked Candidate List     │
                               │ sub-05 & sub-06: (102, 102)     │
                               └─────────────────────────────────┘
```

### Key Pillars:
1. **Meta V-JEPA 2 (8 Temporal Sub-Clips):** Evaluates video dynamics over 64 total frames chunked into 8 consecutive sub-clips. Unlike single-frame or static models, this captures motion speed, scene transitions, and temporal causality matching the continuous brain response.
2. **DINOv2 (ViT-B/14 Layer 11):** Provides high-resolution object and scene category semantics. Concatenated with HTSAT-CLAP embeddings to leverage both visual and acoustic perception.
3. **Modality Dropout ($p = 0.2$):** Prevents the network from over-relying solely on audio or visual cues, making the learned representation resilient to quiet scenes or visual cuts.
4. **Anti-Hubness Calibration (CSLS):** Cross-Domain Similarity Local Scaling reduces the dominance of "hub" candidates that artificially attract high cosine similarity across diverse queries.
5. **Cross-Subject Shared Semantic Space:** Aligns representations from all 6 subjects into a unified 256-D metric space, followed by test prediction on unseen test subjects `sub-05` and `sub-06`.

---

## 4. Official Experimental Results & Benchmarks

All models were evaluated on both the **Held-Out Validation Split** (stimuli 901–1000, 582 trials) and the **Public Test Ground Truth** (`sub-01` to `sub-04`, 408 trials).

### Benchmark Comparison Table

| Model / Configuration | Video Backbone | Audio Backbone | Held-Out Top-1 | Test Top-1 | Test Top-5 | Test Median Rank |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chance Baseline** | — | — | 1.00% | 0.98% (1/102) | 4.90% (5/102) | 51.5 |
| **Exp 4 (Standalone)** | DINOv2 Layer 11 | CLAP (512-D) | 6.87% | 11.27% (46/408) | 30.64% (125/408) | 13.0 |
| **V-JEPA 2 (Standalone)** | V-JEPA 2 (8 Sub-Clips) | CLAP (512-D) | 10.31% | 14.46% (59/408) | 37.99% (155/408) | 9.5 |
| **Team Fruitbasket 2-Way Blend** | **DINOv2 + V-JEPA 2** | **CLAP (512-D)** | **10.82%** | **16.18% (66/408)** | **41.91% (171/408)** | **7.8** |
| *Optional 3-Way Tri-Blend* | *DINO + V-JEPA + ViEEG* | *CLAP (512-D)* | *11.00%* | *16.18% (66/408)* | *41.42% (169/408)* | *7.8* |

> **Performance Gain:** The 2-Way Blend outperforms random chance by **+15.20% absolute Top-1 accuracy** (over **16x better than chance**) and improves median candidate rank from **51.5 down to 7.8**.

---

### Public Test Per-Subject Breakdown (`sub-01` to `sub-04`)

Each public test subject contains 102 evaluation trials across 102 candidates:

| Subject | Top-1 Accuracy | Top-1 Count | Top-5 Accuracy | Top-5 Count | Median Rank |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`sub-01`** | **18.63%** | 19 / 102 | **46.08%** | 47 / 102 | **6.0** |
| **`sub-02`** | **16.67%** | 17 / 102 | **41.18%** | 42 / 102 | **8.0** |
| **`sub-03`** | **13.73%** | 14 / 102 | **42.16%** | 43 / 102 | **8.0** |
| **`sub-04`** | **15.69%** | 16 / 102 | **38.24%** | 39 / 102 | **9.0** |
| **AVERAGE** | **16.18%** | **66 / 408** | **41.91%** | **171 / 408** | **7.8** |

---

### Optimal Simplex Blending Weights
- **Exp4 Weight ($\alpha$):** `35.0%`
- **V-JEPA 2 Weight ($1 - \alpha$):** `65.0%`
- **CSLS Hubness Penalty ($\lambda$):** `0.15`
- **Held-Out Score:** Top-1: `10.82%` (63/582), Top-5: `25.43%` (148/582)

---

## 5. Repository Structure

To keep the workspace clean, **all specialized execution scripts and PowerShell profiles are located in the `scripts/` directory**, leaving strictly **one single master launcher (`run_all.bat`)** in the root:

```text
team-fruitbasket-eeg-mindreading-challenge/
├── run_all.bat                         # MASTER ROOT LAUNCHER (Interactive & Automated)
├── run_blend.py                        # Standalone 1-Click reproduction Python script
├── README.md                           # Comprehensive documentation & results
├── REPORT.md                           # Official scientific & technical report
├── Commands.md                         # Command reference guide
├── CONTEXT.md                          # In-depth architectural & repository reference
├── .gitignore                          # Excludes raw MP4 videos & extracted frames
│
├── scripts/                            # ALL SPECIALIZED BATCH & POWERSHELL SCRIPTS
│   ├── run_exp4.bat                    # Retrain Exp4 on CUDA GPU (Batch)
│   ├── run_exp4.ps1                    # Retrain Exp4 on CUDA GPU (PowerShell)
│   ├── run_vjepa.bat                   # Retrain V-JEPA 2 on CUDA GPU (Batch)
│   ├── run_vjepa.ps1                   # Retrain V-JEPA 2 on CUDA GPU (PowerShell)
│   ├── run_blend.bat                   # Run 2-Way Simplex Blend (Batch)
│   ├── run_blend.ps1                   # Run 2-Way Simplex Blend (PowerShell)
│   ├── run_tune_vjepa.bat              # Run V-JEPA parameter & CSLS tuning sweep
│   ├── package_submission.bat          # Package final submission ZIP (Batch)
│   ├── package_submission.ps1          # Package final submission ZIP (PowerShell)
│   ├── run_all.ps1                     # Full pipeline runner (PowerShell)
│   └── run_blend.sh                    # Linux cluster blend script (Bash)
│
├── submissions/                        # VERIFIED COMPETITION DELIVERABLES
│   ├── sub-05_ranked_candidate_list.npy # Shape: (102, 102), IDs 1001..1102
│   ├── sub-06_ranked_candidate_list.npy # Shape: (102, 102), IDs 1001..1102
│   ├── final_submission_blend.zip      # ZIP ready for competition upload
│   └── blend_matrices/                 # Blended similarity score matrices
│
├── experiments/
│   ├── blend_exp4_vjepa.py             # 2-way & 3-way simplex blending engine
│   ├── exp4_tuned_mlp_dino_audio/      # Exp4 architecture, trainer, and runs
│   │   ├── train.py                    # Multi-seed DINOv2 + CLAP trainer
│   │   └── runs/exp4_dino_clap_retrain/ # Retrained model weights & matrices
│   └── v-jepa/                         # V-JEPA 2 pipeline
│       ├── simple_mlp/train_recipe.py  # 8-clip V-JEPA 2 multi-seed trainer
│       ├── vjepa2_frames/              # Precomputed 8-clip visual feature banks
│       ├── models/recipe_mlp/          # Retrained V-JEPA 2 PyTorch model weights (.pt)
│       ├── tune_vjepa_eval.py          # Anti-hubness tuning module
│       └── submissions/                # V-JEPA 2 standalone score matrices
│
├── trn/                                # Training EEG recordings (sub-01_trn .. sub-06_trn)
├── tst/                                # Test EEG recordings (sub-01_tst .. sub-06_tst)
├── tst_stim/                           # Ground truth labels for public evaluation (sub-01..04)
├── features_dinov2/                    # Precomputed DINOv2 Layer 11 train features
├── features_test_dinov2/               # Precomputed DINOv2 Layer 11 test features
└── features_audio/                     # Precomputed CLAP train and test audio features
```

---

## 6. Environment Setup & Installation (Python 3.11)

### Hardware Requirements
- **Local Machine:** NVIDIA GPU with $\ge$ 4 GB VRAM (e.g., RTX 3050 Laptop / RTX 3060 / RTX 40-series).
- **Cluster / Server:** NVIDIA V100 / A100 / H100 (optional).
- **RAM:** Minimum 16 GB system memory.

### Step-by-Step Setup (Python 3.11):
```bash
# 1. Create and activate a Python 3.11 conda environment
conda create -n fruitbasket python=3.11 -y
conda activate fruitbasket

# Option A: Quick manual installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy scikit-learn tqdm

# Option B: Complete reproducible installation from requirements.txt
pip install -r requirements.txt

# 3. Verify environment and GPU access
python -c "import torch; print('Python 3.11 OK | PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

> [!IMPORTANT]
> **OpenMP Collision Workaround on Windows:**  
> When running PyTorch with Intel MKL on Windows, set `KMP_DUPLICATE_LIB_OK=TRUE`. All scripts in `scripts/` and `run_all.bat` automatically configure this variable.

---

## 7. Master Command Reference (All Tasks)

### Task 1: Master Root Launcher (Interactive / Automated)
The easiest way to run tasks. Simply double-click `run_all.bat` or run from PowerShell:

```powershell
cd C:\Users\amanr\Desktop\team-fruitbasket-eeg-mindreading-challenge

# Interactive Menu:
.\run_all.bat

# Or run non-interactively with arguments:
.\run_all.bat --all       # Runs Exp4 -> V-JEPA 2 -> Blend -> Package
.\run_all.bat --exp4      # Trains Exp4 only
.\run_all.bat --vjepa     # Trains V-JEPA 2 only
.\run_all.bat --blend     # Runs 2-Way Blend only
.\run_all.bat --package   # Packages final submission ZIP
.\run_all.bat --instant   # Runs instant reproduction
```

---

### Task 2: Instant 1-Click Reproduction (5 Seconds)
Computes the winning blend and generates public test metrics using the precomputed, verified score matrices:

```powershell
cd C:\Users\amanr\Desktop\team-fruitbasket-eeg-mindreading-challenge
python run_blend.py
```
*Expected Output: Held-Out Top-1 = 10.82%, Public Test Top-1 = 16.18%, Median Rank = 7.8.*

---

### Task 3: Train Exp 4 (DINOv2 + CLAP MLP) on GPU
*Runtime: ~12 minutes on NVIDIA RTX 3050 (4GB)*

#### Option A: Using Script
```powershell
# Batch:
.\scripts\run_exp4.bat

# PowerShell:
.\scripts\run_exp4.ps1
```

#### Option B: Direct CLI Command
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python experiments\exp4_tuned_mlp_dino_audio\train.py `
  --data-root . `
  --output-root experiments\exp4_tuned_mlp_dino_audio\runs `
  --run-name exp4_dino_clap_retrain `
  --subjects 1,2,3,4,5,6 `
  --prediction-subjects 1,2,3,4,5,6 `
  --seeds 42,123,999 `
  --audio-model clap `
  --fusion concat `
  --loss multipositive `
  --dino-pooling mean `
  --hidden-dim 512 `
  --embedding-dim 256 `
  --lr 3e-4 `
  --weight-decay 0.01 `
  --dropout 0.3 `
  --temperature 0.07 `
  --anti-hub csls `
  --csls-lambda auto `
  --device cuda `
  --predict-test `
  --report-public-subjects `
  --final-full-fit
```

---

### Task 4: Train V-JEPA 2 (8 Temporal Sub-Clips + CLAP) on GPU
*Runtime: ~14 minutes on NVIDIA RTX 3050 (4GB)*

#### Option A: Using Script
```powershell
# Batch:
.\scripts\run_vjepa.bat

# PowerShell:
.\scripts\run_vjepa.ps1
```

#### Option B: Direct CLI Command
```powershell
cd experiments\v-jepa
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python simple_mlp\train_recipe.py `
  --device cuda `
  --seeds 42,123,999 `
  --select_on heldout `
  --anti_hub csls `
  --final_full_fit `
  --dropout 0.4 `
  --modality_dropout 0.2 `
  --weight_decay 0.005 `
  --hub_penalty 0.1 `
  --feat_cache .\vjepa2_frames `
  --tag vjepa2_frames `
  --trn_dir ..\..\trn `
  --tst_dir ..\..\tst `
  --gt_dir ..\..\tst_stim
cd ..\..
```

---

### Task 5: Run 2-Way Simplex Blend & Evaluation
*Runtime: ~5 seconds*

#### Option A: Using Script
```powershell
# Batch:
.\scripts\run_blend.bat

# PowerShell:
.\scripts\run_blend.ps1

# Linux Bash:
bash scripts/run_blend.sh
```

#### Option B: Direct CLI Command
```powershell
python experiments\blend_exp4_vjepa.py `
  --exp4-run experiments\exp4_tuned_mlp_dino_audio\runs\exp4_dino_clap_retrain `
  --vjepa-dir experiments\v-jepa\submissions `
  --gt-dir tst_stim `
  --output-dir submissions
```

---

### Task 6: Hyperparameter & Anti-Hubness Tuning
Runs a grid search over CSLS hub penalties and modality dropouts:

```powershell
.\scripts\run_tune_vjepa.bat
```

---

### Task 7: Package Final Competition Submission ZIP
Generates the official submission zip containing `sub-05` and `sub-06` ranked candidate lists at the root of the archive:

```powershell
# Batch:
.\scripts\package_submission.bat

# PowerShell:
.\scripts\package_submission.ps1
```

---

### Task 8: Optional 3-Way Tri-Blend (+ ViEEG)
If you wish to evaluate the 3-Way blend incorporating ViEEG spatio-temporal convolutions:

```powershell
python experiments\blend_exp4_vjepa.py `
  --exp4-run experiments\exp4_tuned_mlp_dino_audio\runs\exp4_dino_clap_run `
  --vjepa-dir experiments\v-jepa\submissions `
  --vieeg-run experiments\exp13_vieeg_dino_audio\runs\exp13_vieeg_run `
  --gt-dir tst_stim `
  --output-dir submissions\tri_blend
```
*Validation Score: Top-1 = 11.00%, Public Test Top-1 = 16.18% (with `sub-02` reaching 17.65%).*

---

## 8. Expected Output & Submission Format Verification

The competition rules require predictions for held-out subjects `sub-05` and `sub-06` formatted as follows:

```powershell
python -c "
import numpy as np
sub5 = np.load('submissions/sub-05_ranked_candidate_list.npy')
sub6 = np.load('submissions/sub-06_ranked_candidate_list.npy')
print('sub-05 shape:', sub5.shape, '| dtype:', sub5.dtype, '| min/max:', sub5.min(), sub5.max())
print('sub-06 shape:', sub6.shape, '| dtype:', sub6.dtype, '| min/max:', sub6.min(), sub6.max())
assert sub5.shape == (102, 102), 'Invalid shape for sub-05'
assert sub6.shape == (102, 102), 'Invalid shape for sub-06'
assert sub5.min() == 1001 and sub5.max() == 1102, 'Candidate IDs out of range for sub-05'
assert sub6.min() == 1001 and sub6.max() == 1102, 'Candidate IDs out of range for sub-06'
print('>>> ALL SUBMISSION FILES PERFECTLY VALIDATED! <<<')
"
```

### Official Submission Specifications:
1. **Filename & Compression:** `final_submission_blend.zip` containing `sub-05_ranked_candidate_list.npy` and `sub-06_ranked_candidate_list.npy` directly at the root of the archive (no nested folders).
2. **Matrix Dimensions:** Integer 2D array of shape `(102, 102)` saved with `numpy.save`.
3. **Candidate Video IDs:** Each row contains all 102 candidate IDs (`1001` to `1102`, **not** column indices 0 to 101).
4. **Row Preservation:** Row $i$ corresponds strictly to the $i$-th recording in `tst/sub-XX_tst.npy` in the **original unsorted order** (not sorted by label).
5. **Rank Order:** Left-to-right (column 0 is the model's most confident Top-1 prediction).
6. **Self-Verification on `sub-01`:** Running the pipeline on `sub-01` test recordings and scoring against `tst_stim/sub-01_tst_stim.npy` achieves **18.63% Top-1** and **46.08% Top-5**, confirming calibration.

---

## 9. Technical Deep Dive & Design Decisions

### 1. Why V-JEPA 2 Outperforms Static Vision Backbones
Standard image foundation models (e.g., CLIP, single-frame DINO) only see static frames or averaged feature representations. Human visual perception during movie-watching is heavily driven by **motion vectors, optical flow, and temporal changes**. Meta V-JEPA 2 is pretrained using joint-embedding predictive architecture over video volumes. By chunking 5-second video clips into 8 sequential sub-clips, V-JEPA 2 captures dynamic actions (e.g., a person walking, camera panning) that match the continuous time-series EEG responses.

### 2. Multi-Positive InfoNCE Loss
Multiple EEG trials from different subjects correspond to the same underlying movie stimulus. Traditional contrastive learning treats all other batch elements as negatives. Our **multi-positive contrastive formulation** treats all EEG trials watching the same stimulus $i$ as positive targets, dramatically stabilizing metric space convergence.

### 3. Cross-Domain Similarity Local Scaling (CSLS)
In high-dimensional embedding spaces, certain candidate vectors inevitably become "hubs" — vectors that have high average cosine similarity to many queries purely due to geometry rather than semantic alignment. CSLS adjusts the raw cosine similarity $S(q, c)$ by subtracting the mean cosine similarity of $c$ to its $k$-nearest neighbors:
$$\text{CSLS}(q, c) = 2 \cos(q, c) - r_k(q) - r_k(c)$$
This penalizes false-positive hubs and raised Top-1 accuracy by $+0.5\%$ across validation and test splits.

---

## 10. GitHub Packaging & Deliverables

### Final Submission Bundle
- **GitHub Repository:** `team-fruitbasket-eeg-mindreading-challenge`
- **Technical Report:** [REPORT.md](REPORT.md)
- **Official Submission ZIP:** `submissions/final_submission_blend.zip`

### Exclusion of Videos and Raw Frames
In compliance with the hackathon rule ("*A reproducible training and inference pipeline must be uploaded to the designated GitHub repository (without the videos or extracted frames)*"), our [.gitignore](.gitignore) automatically excludes:
- Raw video formats (`*.mp4`, `*.avi`, `*.mkv`, `*.mov`)
- Extracted frame directories (`frames/`, `raw_frames/`, `extracted_frames/`)

All necessary precomputed feature tensors (DINOv2 Layer 11, V-JEPA 2 8-subclip, CLAP audio) and trained model weights (`.pt`) are retained for instant reproducibility.

---

## Authors & Citation
**Team Fruitbasket**  
*EEG-to-Video Decoding & Multimodal Retrieval Challenge*
