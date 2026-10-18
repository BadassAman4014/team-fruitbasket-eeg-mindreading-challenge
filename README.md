# Team Fruitbasket — EEG Mindreading Retrieval Challenge

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6 CUDA 12.4](https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA%2012.4-orange.svg)](https://pytorch.org/)
[![Top-1 Test Accuracy](https://img.shields.io/badge/Public%20Test%20Top--1-16.18%25-success.svg)]()
[![Technical Report](https://img.shields.io/badge/Technical%20Report-REPORT.md-purple.svg)](REPORT.md)
[![Project Context](https://img.shields.io/badge/Project%20Context-CONTEXT.md-blueviolet.svg)](CONTEXT.md)

Multimodal neural retrieval system decoding 5-second 63-channel human EEG trials ($\mathbb{R}^{63 \times 1250}$) into audiovisual movie stimuli from an unseen 102-candidate pool. Combines spatial-semantic representations (**Meta DINOv2 Layer 11**), temporal video dynamics (**Meta V-JEPA 2** across 8 temporal sub-clips), acoustic embeddings (**LAION HTSAT-CLAP**), and **Cross-Domain Similarity Local Scaling (CSLS)** metric calibration.

---

## 1. Benchmark Results

### Standalone Models vs. Simplex Blend
Evaluated across 408 public test trials (`sub-01` to `sub-04`) against the 102-candidate pool:

| Model / Configuration | Video Backbone | Audio Backbone | Held-Out Top-1 | Test Top-1 | Test Top-5 | Test Median Rank |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chance Baseline** | — | — | 1.00% | 0.98% (4/408) | 4.90% (20/408) | 51.5 |
| **Exp 4 (Standalone)** | DINOv2 Layer 11 | CLAP (512-D) | 6.87% | 11.27% (46/408) | 30.64% (125/408) | 13.0 |
| **V-JEPA 2 (Standalone)** | V-JEPA 2 (8 Sub-Clips) | CLAP (512-D) | 10.31% | 14.46% (59/408) | 37.99% (155/408) | 9.5 |
| **Team Fruitbasket 2-Way Blend** | **DINOv2 + V-JEPA 2** | **CLAP (512-D)** | **10.82%** | **16.18% (66/408)** | **41.91% (171/408)** | **7.8** |
| *Optional 3-Way Tri-Blend* | *DINO + V-JEPA + ViEEG* | *CLAP (512-D)* | *11.00%* | *16.18% (66/408)* | *41.42% (169/408)* | *7.8* |

> **Performance Gain:** The 2-Way Blend achieves **16.18% Top-1 Accuracy** (**>16× random chance**) and reduces median candidate rank from **51.5 down to 7.8**.

### Public Test Per-Subject Performance (102 Trials / Subject)

| Subject | Top-1 Accuracy | Top-1 Matches | Top-5 Accuracy | Top-5 Matches | Median Rank |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`sub-01`** | **18.63%** | 19 / 102 | **46.08%** | 47 / 102 | **6.0** |
| **`sub-02`** | **16.67%** | 17 / 102 | **41.18%** | 42 / 102 | **8.0** |
| **`sub-03`** | **13.73%** | 14 / 102 | **42.16%** | 43 / 102 | **8.0** |
| **`sub-04`** | **15.69%** | 16 / 102 | **38.24%** | 39 / 102 | **9.0** |
| **Overall Mean** | **16.18%** | **66 / 408** | **41.91%** | **171 / 408** | **7.8** |

---

## 2. Technical Methodology & Architecture

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
          │ • Subject Alignment Heads   │                     │ • Subject Linear Heads      │
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

### 2.1 Subject-Specific Cortical Alignment
To handle cross-subject anatomical variations and impedance shifts, raw EEG signals $\mathbf{x}^{(s)} \in \mathbb{R}^{63 \times 1250}$ are flattened ($78,750$-D) and projected through subject-specific linear heads:
$$\mathbf{z}_{\text{eeg}}^{(s)} = W_{\text{eeg}}^{(s)} \mathbf{x}^{(s)} + \mathbf{b}_{\text{eeg}}^{(s)}, \quad \mathbf{z}_{\text{eeg}}^{(s)} \in \mathbb{R}^{256}$$
This maps all 6 subjects into a unified 256-D metric space for joint training.

### 2.2 Complementary Multimodal Backbones
1. **Static Visual Semantics (Stream A — Exp4):**
   - **Backbone:** Meta DINOv2 (`dinov2_vitb14`), extracting penultimate representations from **Layer 11** (768-D). Preserves high-level visual object categorization.
   - **Audio:** LAION HTSAT-CLAP (512-D).
2. **Dynamic Spatio-Temporal Modeling (Stream B — V-JEPA 2):**
   - **Backbone:** Meta V-JEPA 2 (`vjepa2_vitl16`). Partitions the 5-second video into **8 consecutive non-overlapping temporal sub-clips** (64 frames total), producing an $(8, 1024)$ temporal feature tensor that captures continuous scene dynamics, motion, and visual transitions.
   - **Audio:** Co-embedded with CLAP (512-D).

### 2.3 Regularization & Training Objectives
- **Modality Dropout ($p = 0.20$):** Randomly zeros visual or audio features during training to eliminate unimodal bias:
  $$\mathbf{z}_{\text{stim}} = W_{\text{stim}} \left[ m_v \cdot \mathbf{v} \,\|\, m_a \cdot \mathbf{a} \right], \quad m_v, m_a \sim \text{Bernoulli}(0.80)$$
- **Multi-Positive InfoNCE Loss:** Generalizes contrastive learning across trials where different subjects viewed the identical stimulus:
  $$\mathcal{L} = -\sum_{i \in \mathcal{B}} \frac{1}{|\mathcal{P}(i)|} \sum_{j \in \mathcal{P}(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \in \mathcal{B}} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}, \quad \tau = 0.07$$

### 2.4 Anti-Hubness Calibration & Simplex Ensembling
- **CSLS (Cross-Domain Similarity Local Scaling):** Corrects high-dimensional cosine hubness artifacts:
  $$\text{CSLS}(\mathbf{q}, \mathbf{c}) = 2 \cos(\mathbf{q}, \mathbf{c}) - \lambda \cdot r_k(\mathbf{q}) - \lambda \cdot r_k(\mathbf{c}), \quad \lambda = 0.15, \; k = 10$$
- **Convex Simplex Ensembling:** The optimal weights were selected strictly on the 582 held-out trials (stimuli 901–1000):
  $$S_{\text{blend}} = 0.35 \cdot S_{\text{exp4}} + 0.65 \cdot S_{\text{vjepa}}$$

---

## 3. Reproduction & Execution

### 3.1 Environment Setup (Python 3.11)
```bash
# Create and activate environment
conda create -n test python=3.11 -y
conda activate test

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

---

### 3.2 1-Click Reproduction
Computes the calibrated 2-Way Blend and generates public test metrics and submission files using verified precomputed matrices:

```bash
python run_blend.py
```
*Expected Output:* Held-out Top-1 = **10.82%**, Public Test Top-1 = **16.18%**, Median Rank = **7.8**.

---

### 3.3 Full GPU Retraining Pipeline (~40 Minutes)
Retrains Exp4, retrains V-JEPA 2, performs the held-out simplex blend sweep, and packages the final competition deliverables:

**Windows (Batch):**
```cmd
run_all.bat --all
```

**Windows (PowerShell):**
```powershell
.\scripts\run_all.ps1
```

---

### 3.4 Modular Execution Commands

| Task | Command | Description |
|:---|:---|:---|
| **Train Exp4 Only** | `.\scripts\run_exp4.bat` | Retrains DINOv2 + CLAP MLP on CUDA GPU (~12 min) |
| **Train V-JEPA 2 Only** | `.\scripts\run_vjepa.bat` | Retrains 8-subclip V-JEPA 2 + CLAP on CUDA GPU (~25 min) |
| **Run Simplex Blend** | `.\scripts\run_blend.bat` | Computes optimal blending sweep and held-out metrics |
| **Hyperparameter Sweep** | `.\scripts\run_tune_vjepa.bat` | Sweeps CSLS penalty $\lambda$ and dropout rates |
| **Package Submission** | `.\scripts\package_submission.bat` | Validates arrays and generates `final_submission_blend.zip` |

---

## 4. Repository Layout

```text
team-fruitbasket-eeg-mindreading-challenge/
├── run_all.bat                         # Master root launcher (Interactive & CLI flags)
├── run_blend.py                        # Standalone 1-Click reproduction script (5 seconds)
├── README.md                           # Technical overview & reproduction guide
├── REPORT.md                           # Official scientific & engineering report
├── CONTEXT.md                          # In-depth architectural & implementation reference
├── environment.yml                     # Conda environment definition
├── requirements.txt                    # Pip dependencies (PyTorch CUDA 12.4 build)
│
├── scripts/                            # Modular execution scripts
│   ├── run_all.ps1                     # Full GPU pipeline runner (PowerShell)
│   ├── run_exp4.bat / .ps1             # Exp4 GPU training scripts
│   ├── run_vjepa.bat / .ps1            # V-JEPA 2 GPU training scripts
│   ├── run_blend.bat / .ps1 / .sh      # Simplex blend runners
│   ├── run_tune_vjepa.bat              # Parameter tuning sweep
│   └── package_submission.bat / .ps1   # Submission ZIP packager
│
├── trn/                                # Multi-subject training EEG (sub-01_trn .. sub-06_trn)
├── tst/                                # Multi-subject test EEG (sub-01_tst .. sub-06_tst)
├── tst_stim/                           # Ground truth labels for public test verification
├── features_dinov2/                    # Precomputed DINOv2 Layer 11 features (stimuli 1..1000)
├── features_test_dinov2/               # Precomputed DINOv2 Layer 11 features (stimuli 1001..1102)
├── features_audio/                     # Precomputed CLAP, AST, and BEATs audio caches
│
├── submissions/                        # Output ranked lists & submission deliverables
│   ├── sub-05_ranked_candidate_list.npy  # Shape: (102, 102), IDs: 1001..1102
│   ├── sub-06_ranked_candidate_list.npy  # Shape: (102, 102), IDs: 1001..1102
│   └── final_submission_blend.zip        # Official submission bundle
│
└── experiments/                        # Core model code, architectures, and checkpoints
    ├── blend_exp4_vjepa.py             # Score-level simplex blender & metric evaluator
    ├── exp4_tuned_mlp_dino_audio/      # Exp4 architecture, training script & checkpoints
    └── v-jepa/                         # V-JEPA 2 sub-clip recipe, dataset loaders & metrics
```
