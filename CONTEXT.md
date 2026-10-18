# CONTEXT.md — Project Technical Reference Manual
## Team Fruitbasket — EEG Mindreading Challenge
**GitHub Repository:** `team-fruitbasket-eeg-mindreading-challenge`  
**Challenge:** Multimodal EEG-to-Video/Audio Retrieval Challenge  
**Environment:** Python 3.11 | PyTorch 2.6 (CUDA 12.4)  
**Deliverables:** `submissions/sub-05_ranked_candidate_list.npy`, `submissions/sub-06_ranked_candidate_list.npy`, `submissions/final_submission_blend.zip`

---

## 1. Executive Summary & Challenge Overview

### 1.1 The Challenge
The **EEG Mind Reading Retrieval Challenge** is an international machine learning competition in computational neuroscience and Brain-Computer Interfaces (BCIs). The task is to decode non-invasive human electroencephalography (EEG) signals recorded while human participants watched diverse 5-second audiovisual movie clips, and retrieve the exact clip being viewed from an unseen candidate pool of 102 possible stimuli.

### 1.2 Input / Output Specification
- **Input Query:** A 5-second multichannel EEG trial $\mathbf{x}_{\text{eeg}} \in \mathbb{R}^{63 \times 1250}$ acquired from 63 scalp electrodes sampled at 250 Hz.
- **Candidate Pool:** $\mathcal{C} = \{c_1, c_2, \dots, c_{102}\}$ containing 102 candidate movie segments (candidate IDs spanning $1001$ to $1102$).
- **Output Target:** A ranked candidate permutation list $\pi = (\pi_1, \pi_2, \dots, \pi_{102})$ sorted in descending order of predicted similarity such that the true stimulus $c^*$ appears at rank 1:
  $$\text{Rank}(c^*) = 1$$
- **Final Deliverable Format:** A 2D NumPy array of shape `(102, 102)` and dtype `int64` saved via `np.save`. Row $i$ corresponds strictly to the $i$-th unsorted EEG recording in `tst/sub-XX_tst.npy`, containing the 102 candidate IDs sorted from most likely to least likely.

### 1.3 Core Challenges & Scientific Difficulties
1. **Low Signal-to-Noise Ratio (SNR):** Non-invasive scalp potentials are microvolt-level superpositions of cortical currents severely attenuated by skull and scalp tissue, corrupted by ocular blinks, saccades, and muscular artifacts.
2. **Subject Heterogeneity:** Distinct anatomical cranial geometry, cortical folding patterns, and sensor impedances lead to substantial cross-subject distribution shifts.
3. **Dynamic Multimodal Perception:** Naturalistic movie viewing triggers simultaneous spatial visual processing (dorsal/ventral streams), dynamic scene motion perception, and auditory-evoked potentials (AEPs) across speech and soundtrack cues.
4. **Geometric Hubness:** In high-dimensional contrastive embedding spaces, certain candidate vectors become topological "hubs" that falsely appear close to multiple queries.

### 1.4 Benchmark Highlights
Our solution, **Team Fruitbasket 2-Way Simplex Blend**, achieves state-of-the-art retrieval performance across all public benchmark subjects (`sub-01` to `sub-04`, 408 trials):

| Metric | Random Chance Baseline | Exp 4 Standalone | V-JEPA 2 Standalone | **Team Fruitbasket 2-Way Blend** |
|:---|:---:|:---:|:---:|:---:|
| **Held-Out Validation Top-1** | 1.00% | 6.87% | 10.31% | **10.82%** (63 / 582) |
| **Public Test Top-1 Accuracy** | 0.98% (4 / 408) | 11.27% (46 / 408) | 14.46% (59 / 408) | **16.18% (66 / 408)** |
| **Public Test Top-5 Accuracy** | 4.90% (20 / 408) | 30.64% (125 / 408) | 37.99% (155 / 408) | **41.91% (171 / 408)** |
| **Median Candidate Rank** | 51.5 | 13.0 | 9.5 | **7.8** (Baseline: 51.5) |

---

## 2. Multimodal Architecture & Methodology

```text
                               ┌────────────────────────────────────────┐
                               │     EEG Recording: [63, 1250]          │
                               └───────────────────┬────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
          ┌─────────────────────────────┐                     ┌─────────────────────────────┐
          │  Stream A: DINOv2 + CLAP    │                     │  Stream B: V-JEPA 2 + CLAP  │
          │  (Exp4 Multi-Seed MLP)      │                     │  (8 Temporal Sub-Clips)     │
          ├─────────────────────────────┤                     ├─────────────────────────────┤
          │ • Subject Alignment Heads   │                     │ • Per-Subject Linear Heads  │
          │ • DINOv2 L11 (768-D)        │                     │ • V-JEPA 2 ViT-L (1024-D)   │
          │ • CLAP Audio (512-D)        │                     │ • CLAP Audio (512-D)        │
          │ • Multi-positive InfoNCE    │                     │ • Modality Dropout (p=0.2)  │
          │ • Full-fit Multi-seed (3)   │                     │ • Full-fit Multi-seed (3)   │
          └──────────────┬──────────────┘                     └──────────────┬──────────────┘
                         │                                                   │
                         │ Cosine Similarity S_exp4                          │ Cosine Similarity S_vjepa
                         ▼                                                   ▼
          ┌─────────────────────────────────────────────────────────────────────────────────┐
          │              2-Way Simplex Weight Blend & Anti-Hubness Calibration              │
          │           S_blend = 0.35 * S_exp4 + 0.65 * S_vjepa  (CSLS lambda = 0.15)        │
          └────────────────────────────────────────┬────────────────────────────────────────┘
                                                   ▼
                                  ┌─────────────────────────────────┐
                                  │ Final Ranked Permutation List   │
                                  │ sub-05 & sub-06: (102, 102)     │
                                  └─────────────────────────────────┘
```

### 2.1 Stream A: Exp4 (Static Semantics + Acoustic Alignment)
- **Visual Backbone:** Meta DINOv2 (`dinov2_vitb14`), extracting penultimate representations from **Layer 11** (768-D). Layer 11 captures fine-grained visual object semantics prior to final classification collapse.
- **Acoustic Backbone:** LAION HTSAT-CLAP (`630k-audioset-fusion-best`), extracting 512-D environmental audio representations covering speech rhythm, ambient music, and sound effects.
- **EEG Alignment Head:** Subject-specific linear projection:
  $$\mathbf{z}_{\text{eeg}}^{(s)} = W_{\text{eeg}}^{(s)} \mathbf{x}^{(s)} + \mathbf{b}_{\text{eeg}}^{(s)}, \quad \mathbf{z}_{\text{eeg}}^{(s)} \in \mathbb{R}^{256}$$
  This projects each subject's flattened $78,750$-D cortical vector into a shared 256-D metric space, preserving cross-subject semantic alignment.
- **Fusion:** Feature concatenation followed by Layer Normalization and an MLP projection head.

### 2.2 Stream B: V-JEPA 2 (Fine-Grained Video Dynamics + CLAP)
- **Visual Backbone:** Meta V-JEPA 2 (`vjepa2_vitl16`). To resolve video motion, camera transitions, and visual saccades across the 5-second interval, each clip is partitioned into **8 consecutive non-overlapping temporal sub-clips** (64 frames total).
- **Sub-Clip Pooling:** The resulting $(8, 1024)$ temporal tensor is temporally pooled and projected into the shared 256-D metric space:
  $$\mathbf{v}_{\text{vjepa}} = W_v \text{Pool}(\mathbf{F}_{\text{vjepa}}) \in \mathbb{R}^{256}$$
- **Acoustic Co-embedding:** Concatenated with CLAP audio embeddings (512-D).

### 2.3 Regularization & Training Objectives
1. **Modality Dropout ($p = 0.20$):**
   During training, visual and audio features are independently dropped with probability $p=0.20$:
   $$\mathbf{z}_{\text{stim}} = W_{\text{stim}} \left[ m_v \cdot \mathbf{v} \,\|\, m_a \cdot \mathbf{a} \right], \quad m_v, m_a \sim \text{Bernoulli}(0.80)$$
   This forces the EEG encoder to form robust representations of both modalities, preventing catastrophic reliance on single prominent audio cues.
2. **Multi-Positive Contrastive InfoNCE Loss:**
   Because multiple subjects view the same underlying stimulus $i$, standard contrastive pairing is generalized to multi-positive InfoNCE:
   $$\mathcal{L} = -\sum_{i \in \mathcal{B}} \frac{1}{|\mathcal{P}(i)|} \sum_{j \in \mathcal{P}(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \in \mathcal{B}} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}$$
   with temperature $\tau = 0.07$.
3. **Multi-Seed Ensembling:**
   Trained across seeds 42, 123, and 999 with `--final-full-fit` to ensure maximum stability and generalize across variance.

### 2.4 Cross-Domain Similarity Local Scaling (CSLS)
To eliminate hubness artifacts where generic movie segments dominate top ranks:
$$\text{CSLS}(\mathbf{q}, \mathbf{c}) = 2 \cos(\mathbf{q}, \mathbf{c}) - \lambda \cdot r_k(\mathbf{q}) - \lambda \cdot r_k(\mathbf{c})$$
where $r_k(\cdot)$ is the mean cosine similarity of a vector to its $k=10$ nearest neighbors in the target domain, and $\lambda = 0.15$ (optimized on held-out validation).

### 2.5 2-Way Simplex Blend
The final retrieval score matrix $S_{\text{blend}}$ is an optimal convex combination:
$$S_{\text{blend}} = 0.35 \cdot S_{\text{exp4}} + 0.65 \cdot S_{\text{vjepa}}$$
Selected by an exhaustive grid search over $\alpha \in [0.0, 1.0]$ evaluated strictly on the 582 held-out trials (stimuli 901–1000).

---

## 3. Directory Layout & Repository Map

```text
team-fruitbasket-eeg-mindreading-challenge/
│
├── run_all.bat                         # MASTER ROOT LAUNCHER (Interactive & Automated)
├── run_blend.py                        # Standalone 1-Click reproduction Python script (5s)
├── README.md                           # Main GitHub documentation & benchmarks
├── REPORT.md                           # Official technical & scientific report
├── Commands.md                         # Master command cheatsheet & CLI reference
├── CONTEXT.md                          # In-depth architectural & repository reference
├── environment.yml                     # Conda environment definition (Python 3.11)
├── requirements.txt                    # Pip dependencies (PyTorch CUDA 12.4 build)
├── .gitignore                          # Excludes raw MP4 videos & frame directories
│
├── scripts/                            # MODULAR BATCH & POWERSHELL SCRIPTS
│   ├── run_all.ps1                     # Full GPU pipeline runner (PowerShell)
│   ├── run_exp4.bat                    # Retrain Exp4 on CUDA GPU (Batch)
│   ├── run_exp4.ps1                    # Retrain Exp4 on CUDA GPU (PowerShell)
│   ├── run_vjepa.bat                   # Retrain V-JEPA 2 on CUDA GPU (Batch)
│   ├── run_vjepa.ps1                   # Retrain V-JEPA 2 on CUDA GPU (PowerShell)
│   ├── run_blend.bat                   # Run 2-Way Simplex Blend (Batch)
│   ├── run_blend.ps1                   # Run 2-Way Simplex Blend (PowerShell)
│   ├── run_blend.sh                    # Linux cluster blend script (Bash)
│   ├── run_tune_vjepa.bat              # Run V-JEPA parameter & CSLS tuning sweep
│   ├── package_submission.bat          # Package final submission ZIP (Batch)
│   ├── package_submission.ps1          # Package final submission ZIP (PowerShell)
│   └── generate_requirements.py        # Reproducibility utility to export requirements
│
├── trn/                                # MULTI-SUBJECT TRAINING EEG DATA
│   ├── sub-01_trn.npz                  # Subject 1 training EEG (trials across stimuli 1..1000)
│   ├── sub-02_trn.npz                  # Subject 2 training EEG
│   ├── sub-03_trn.npz                  # Subject 3 training EEG
│   ├── sub-04_trn.npz                  # Subject 4 training EEG
│   ├── sub-05_trn.npz                  # Subject 5 training EEG
│   └── sub-06_trn.npz                  # Subject 6 training EEG
│
├── tst/                                # MULTI-SUBJECT TEST EEG DATA (102 TRIALS EACH)
│   ├── sub-01_tst.npy                  # Public test EEG (102, 17/63, T)
│   ├── sub-02_tst.npy                  # Public test EEG
│   ├── sub-03_tst.npy                  # Public test EEG
│   ├── sub-04_tst.npy                  # Public test EEG
│   ├── sub-05_tst.npy                  # Unseen test EEG for official submission
│   └── sub-06_tst.npy                  # Unseen test EEG for official submission
│
├── tst_stim/                           # PUBLIC TEST EVALUATION LABELS
│   ├── sub-01_tst_stim.npy             # Ground truth stimulus IDs for sub-01 (102 values)
│   ├── sub-02_tst_stim.npy             # Ground truth stimulus IDs for sub-02
│   ├── sub-03_tst_stim.npy             # Ground truth stimulus IDs for sub-03
│   └── sub-04_tst_stim.npy             # Ground truth stimulus IDs for sub-04
│
├── features_dinov2/                    # PRECOMPUTED DINOv2 TRAINING FEATURES
│   └── consolidated_encoder.layer.11.npz  # DINOv2 Layer 11 embeddings (stimuli 1..1000)
│
├── features_test_dinov2/               # PRECOMPUTED DINOv2 TEST FEATURES
│   └── consolidated_encoder.layer.11.npz  # DINOv2 Layer 11 embeddings (stimuli 1001..1102)
│
├── features_audio/                     # PRECOMPUTED AUDIO FEATURE CACHES
│   ├── train_clap_features.npz         # LAION CLAP embeddings for train stimuli
│   ├── test_clap_features.npz          # LAION CLAP embeddings for candidate stimuli
│   ├── train_ast_features.npz          # AST baseline audio features
│   ├── test_ast_features.npz           # AST baseline audio features
│   ├── train_beats_features.npz        # BEATs baseline audio features
│   └── test_beats_features.npz         # BEATs baseline audio features
│
├── submissions/                        # COMPETITION DELIVERABLES & EVALUATION MATRICES
│   ├── sub-05_ranked_candidate_list.npy   # Official submission for Subject 5 (102, 102)
│   ├── sub-06_ranked_candidate_list.npy   # Official submission for Subject 6 (102, 102)
│   ├── final_submission_blend.zip         # Final verified ZIP bundle for portal upload
│   ├── blend_sub-01_similarity.npy        # Calibrated test similarity matrix for sub-01
│   ├── blend_sub-02_similarity.npy        # Calibrated test similarity matrix for sub-02
│   ├── blend_sub-03_similarity.npy        # Calibrated test similarity matrix for sub-03
│   ├── blend_sub-04_similarity.npy        # Calibrated test similarity matrix for sub-04
│   ├── blend_sub-05_similarity.npy        # Calibrated test similarity matrix for sub-05
│   └── blend_sub-06_similarity.npy        # Calibrated test similarity matrix for sub-06
│
└── experiments/                        # EXPERIMENTAL PIPELINES & ARCHITECTURES
    ├── blend_exp4_vjepa.py             # Simplex weight optimizer, score blender & evaluator
    │
    ├── exp4_tuned_mlp_dino_audio/      # STREAM A: DINOv2 + CLAP MLP
    │   ├── train.py                    # Complete PyTorch training & inference pipeline
    │   └── runs/
    │       ├── exp4_dino_clap_run/     # Precomputed held-out & test output matrices
    │       └── exp4_dino_clap_retrain/ # Retrained weights (model_seed-*.pt) & configs
    │
    ├── v-jepa/                         # STREAM B: V-JEPA 2 8-SUBCLIPS + CLAP
    │   ├── vjepa2_frames/              # Video & audio precomputed feature arrays (.npy)
    │   ├── simple_mlp/                 # Model architecture & train recipe
    │   │   ├── model_recipe.py         # Temporal pooling & projection architecture
    │   │   ├── train_recipe.py         # V-JEPA 2 training recipe script
    │   │   └── train.py                # Core training loop utilities
    │   ├── src/                        # Data loaders, decimation, and metric functions
    │   │   ├── data_utils.py           # EEG loading, decimation, and z-scoring
    │   │   ├── dataset.py              # PyTorch Dataset & evaluation loaders
    │   │   └── metrics.py              # Retrieval metrics & CSLS implementation
    │   ├── tune_vjepa_eval.py          # Grid sweep script for hyperparameters
    │   └── submissions/                # Precomputed V-JEPA score matrices
    │
    ├── exp1_two_layer_gcn/             # BASELINE: 2-Layer Graph Convolutional Network
    └── exp2_dgcnn_dino_audio_mlp/      # BASELINE: Dynamic Graph CNN
```

---

## 4. Dataset Specifications & Preprocessing Protocol

### 4.1 Training Dataset (`trn/`)
- Contains 6 `.npz` files (`sub-01_trn.npz` through `sub-06_trn.npz`).
- Keys are string representations of 1-based stimulus IDs (`"1"` to `"1000"`).
- Each value is a NumPy array representing the raw EEG recording of shape `(63, 1500)` or `(17, 1500)` sampled at 500 Hz.
- **Split Protocol:**
  - **Training Set (Stimuli 1–900):** 5,068 total trials across all 6 subjects used exclusively for fitting normalizers and optimizing neural network weights.
  - **Held-Out Validation Set (Stimuli 901–1000):** 582 total trials across all 6 subjects used strictly for checkpoint selection, CSLS tuning, and simplex blending weight optimization.

### 4.2 Test Dataset (`tst/`)
- Contains 6 `.npy` files (`sub-01_tst.npy` through `sub-06_tst.npy`).
- Shape: `(102, 63, 1250)` or `(102, 17, 1500)` representing 102 evaluation trials per subject.
- **Order Guarantee:** The row index $i \in \{0, \dots, 101\}$ is kept in its exact original unsorted order. **Rows are never reordered by stimulus ID.**

### 4.3 Public Test Stimulus Labels (`tst_stim/`)
- Contains 4 ground-truth arrays (`sub-01_tst_stim.npy` to `sub-04_tst_stim.npy`).
- Each contains 102 stimulus IDs within the range $[1001, 1102]$.
- **Usage:** Strictly used for post-hoc validation and ensembling performance verification. These labels are never accessed during model training or normalizer estimation.
- `sub-05` and `sub-06` labels are hidden and held out by the challenge organizers.

### 4.4 Preprocessing Steps
1. **Decimation:** Raw 500 Hz EEG is decimated by factor 5 to 100 Hz using `scipy.signal.decimate`, avoiding high-frequency noise and reducing parameter count.
2. **ERP Windowing:** A time window slice corresponding to $50\text{ ms} \dots 600\text{ ms}$ post-stimulus onset (`slice(5, 60)`) is extracted, capturing the primary visual evoked potential (VEP) and auditory evoked potential (AEP) peaks (P100, N170, P200).
3. **Trial Normalization:** Each trial is normalized by subtracting its temporal mean and dividing by standard deviation across channels.
4. **Subject Z-Scoring:** Scalp impedance variations across subjects are standardized using statistics estimated strictly on the training partition (stimuli 1–900).

---

## 5. Strict Compliance & Integrity Guarantees

Our pipeline adheres strictly to all hackathon rules:

| Guideline | Enforcement Mechanism |
|:---|:---|
| **Python 3.11 Runtime** | Tested and verified under standard Python 3.11 with PyTorch 2.6 (CUDA 12.4). |
| **Zero Test Leakage** | `sub-XX_tst.npy` and `tst_stim/` are completely excluded during training and normalization estimation. Normalizers are computed strictly on stimuli 1–900 and frozen. |
| **No Test Template Matching** | Test recordings of `sub-01..04` are never used as templates or reference prototypes to match test recordings of `sub-05..06`. |
| **Row Order Preservation** | Predictions maintain exact 1-to-1 correspondence with test recordings: row $i$ in `sub-05_ranked_candidate_list.npy` corresponds to row $i$ in `tst/sub-05_tst.npy`. |
| **Format Compliance** | Outputs are integer arrays of shape `(102, 102)` containing candidate IDs `1001` through `1102` (not zero-indexed `0..101`). |
| **Pretrained Model Legitimacy** | Uses only open-source foundation backbones (Meta V-JEPA 2, Meta DINOv2, LAION CLAP) released publicly before the hackathon. |
| **Clean Packaging** | Raw video files (`*.mp4`, `*.mkv`) and raw frame directories are excluded via `.gitignore`. Precomputed feature tensors and trained weights are retained. |

---

## 6. Execution & Reproduction Workflows

### 6.1 Environment Setup (Python 3.11 Conda)
```powershell
# 1. Create conda environment named "test" (or "fruitbasket")
conda create -n test python=3.11 -y
conda activate test

# 2. Install PyTorch with CUDA 12.4 wheels
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install required packages
pip install -r requirements.txt
```

### 6.2 Instant 1-Click Reproduction (5 Seconds)
Computes the winning blend and generates all public test metrics and submission files using the verified precomputed matrices:
```powershell
python run_blend.py
```
*Expected Output:*
- Held-Out Top-1: **10.82%** (CSLS $\lambda = 0.15$, 35% Exp4 + 65% V-JEPA 2)
- Public Test Top-1: **16.18%** (66 / 408)
- Public Test Top-5: **41.91%** (171 / 408)
- Median Rank: **7.8**
- Generated Deliverables: `submissions/sub-05_ranked_candidate_list.npy`, `submissions/sub-06_ranked_candidate_list.npy`

### 6.3 Full GPU Re-Training Pipeline (~40 Minutes)
To retrain Exp4, retrain V-JEPA 2, run the simplex blend sweep, and package the final submission:
```powershell
.\run_all.bat --all
```
*(Or run [run_all.bat](file:///c:/Users/amanr/Desktop/Team_Schubki/run_all.bat) interactively and select Option `1`)*

### 6.4 Modular Component Tasks
- **Train Exp4 Only:** `.\run_all.bat --exp4` or `.\scripts\run_exp4.bat`
- **Train V-JEPA 2 Only:** `.\run_all.bat --vjepa` or `.\scripts\run_vjepa.bat`
- **Run Simplex Blend Only:** `.\run_all.bat --blend` or `.\scripts\run_blend.bat`
- **Run Hyperparameter Sweep:** `.\scripts\run_tune_vjepa.bat`
- **Package Submission ZIP:** `.\run_all.bat --package` or `.\scripts\package_submission.bat`

---

## 7. Public Test Detailed Breakdown (Per-Subject)

The public test set comprises 408 trials across subjects `sub-01` to `sub-04` evaluated against the 102 candidate pool:

| Subject | Test Trials | Top-1 Accuracy | Top-1 Count | Top-5 Accuracy | Top-5 Count | Median Rank | Random Baseline |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`sub-01`** | 102 | **18.63%** | 19 / 102 | **46.08%** | 47 / 102 | **6.0** | 51.5 |
| **`sub-02`** | 102 | **16.67%** | 17 / 102 | **41.18%** | 42 / 102 | **8.0** | 51.5 |
| **`sub-03`** | 102 | **13.73%** | 14 / 102 | **42.16%** | 43 / 102 | **8.0** | 51.5 |
| **`sub-04`** | 102 | **15.69%** | 16 / 102 | **38.24%** | 39 / 102 | **9.0** | 51.5 |
| **Aggregate Mean** | **408** | **16.18%** | **66 / 408** | **41.91%** | **171 / 408** | **7.8** | **51.5** |

---

## 8. Troubleshooting & Windows Environment Tips

1. **`KMP_DUPLICATE_LIB_OK=TRUE` Workaround:**
   - On Windows with Intel MKL and PyTorch, an OpenMP runtime collision can occur when multiple DLLs load `libiomp5md.dll`.
   - **Resolution:** All batch scripts (`.bat`), PowerShell scripts (`.ps1`), and training Python scripts automatically set `KMP_DUPLICATE_LIB_OK=TRUE` at startup.
2. **Conda Python Environment Resolution:**
   - The launcher scripts automatically search for Python in the following priority:
     1. `%USERPROFILE%\.conda\envs\test\python.exe`
     2. `%USERPROFILE%\.conda\envs\fruitbasket\python.exe`
     3. Active system `python` in PATH
   - To force a specific Python executable, set `PYTHON_PATH` before calling the scripts:
     ```bat
     set PYTHON_PATH=C:\path\to\your\python.exe
     run_all.bat
     ```
3. **Array Shape and Dtype Verification:**
   - You can quickly verify the format of generated submission files at any time:
     ```powershell
     python -c "import numpy as np; f=np.load('submissions/sub-05_ranked_candidate_list.npy'); print('Shape:', f.shape, '| Dtype:', f.dtype, '| Min/Max:', f.min(), f.max(), '| Valid IDs:', set(f[0]) == set(range(1001, 1103)))"
     ```
   - *Expected output:* `Shape: (102, 102) | Dtype: int64 | Min/Max: 1001 1102 | Valid IDs: True`
