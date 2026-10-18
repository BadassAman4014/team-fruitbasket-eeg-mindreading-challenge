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
