# Team Fruitbasket — EEG Mindreading Challenge — Master Command Reference

This document contains all commands required to reproduce, train, blend, evaluate, and package the final winning solutions with **zero external dependencies**.

---

## 1. Quick Start: 1-Click Instant Reproduction (5 Seconds)

Uses the precomputed held-out and test score matrices already packaged in the repository. No GPU training needed.

```bash
cd ~/team-fruitbasket-eeg-mindreading-challenge

# Execute 1-click reproduction
python run_blend.py
```

### Expected Output:
- **Held-out Split (Stimuli 901–1000):** Top-1 = **10.82%** (Weights: 35.0% Exp4, 65.0% V-JEPA2, CSLS $\lambda = 0.15$)
- **Public Test (`sub-01` to `sub-04`):**
  - `sub-01`: Top-1 = **18.63%** | Top-5 = **45.10%** | Median = **6.0**
  - `sub-02`: Top-1 = **16.67%** | Top-5 = **41.18%** | Median = **8.0**
  - `sub-03`: Top-1 = **13.73%** | Top-5 = **42.16%** | Median = **8.0**
  - `sub-04`: Top-1 = **15.69%** | Top-5 = **39.22%** | Median = **9.0**
  - **AVERAGE:** Top-1 = **16.18%** (66/408) | Top-5 = **41.91%** (171/408) | Median = **7.8**
- **Deliverables Generated:**
  - `submissions/sub-05_ranked_candidate_list.npy` (102, 102)
  - `submissions/sub-06_ranked_candidate_list.npy` (102, 102)

---

## 2. Complete Re-Training from Scratch (On GPU)

### Master Root Launcher (Interactive / Automated)
Only **one launcher script** is kept in the root folder:

```powershell
cd C:\Users\amanr\Desktop\team-fruitbasket-eeg-mindreading-challenge

# Interactive Mode:
.\run_all.bat

# Automated Non-Interactive Mode:
.\run_all.bat --all
```

---

### Step A: Train Exp 4 (DINOv2 + CLAP MLP)
*Runtime: ~12 minutes on 1x CUDA GPU*

#### Local Windows Script:
```powershell
# Using Batch:
.\scripts\run_exp4.bat

# Using PowerShell:
.\scripts\run_exp4.ps1
```

#### Linux Cluster / Direct Python:
```bash
python experiments/exp4_tuned_mlp_dino_audio/train.py \
  --data-root . \
  --output-root experiments/exp4_tuned_mlp_dino_audio/runs \
  --run-name exp4_dino_clap_retrain \
  --subjects 1,2,3,4,5,6 \
  --prediction-subjects 1,2,3,4,5,6 \
  --seeds 42,123,999 \
  --audio-model clap \
  --fusion concat \
  --loss multipositive \
  --dino-pooling mean \
  --hidden-dim 512 --embedding-dim 256 \
  --lr 3e-4 --weight-decay 0.01 --dropout 0.3 --temperature 0.07 \
  --anti-hub csls --csls-lambda auto \
  --device cuda \
  --predict-test --report-public-subjects \
  --final-full-fit
```

---

### Step B: Train V-JEPA 2 (8 Temporal Sub-Clips + CLAP MLP)
*Runtime: ~14 minutes on 1x CUDA GPU*

#### Local Windows Script:
```powershell
# Using Batch:
.\scripts\run_vjepa.bat

# Using PowerShell:
.\scripts\run_vjepa.ps1
```

#### Linux Cluster / Direct Python:
```bash
cd experiments/v-jepa
python simple_mlp/train_recipe.py \
  --device cuda \
  --seeds 42,123,999 \
  --select_on heldout \
  --anti_hub csls \
  --final_full_fit \
  --dropout 0.4 \
  --modality_dropout 0.2 \
  --weight_decay 0.005 \
  --hub_penalty 0.1 \
  --feat_cache ./vjepa2_frames \
  --tag vjepa2_frames \
  --trn_dir ../../trn \
  --tst_dir ../../tst \
  --gt_dir ../../tst_stim
cd ../..
```

---

### Step C: Blend the Freshly Trained Models
*Runtime: ~5 seconds*

#### Local Windows Script:
```powershell
# Using Batch:
.\scripts\run_blend.bat

# Using PowerShell:
.\scripts\run_blend.ps1
```

#### Linux Cluster / Direct Python:
```bash
python experiments/blend_exp4_vjepa.py \
  --exp4-run experiments/exp4_tuned_mlp_dino_audio/runs/exp4_dino_clap_retrain \
  --vjepa-dir experiments/v-jepa/submissions \
  --gt-dir tst_stim \
  --output-dir submissions
```

---

### Step D: Package Final Competition Submission ZIP

#### Local Windows Script:
```powershell
# Using Batch:
.\scripts\package_submission.bat

# Using PowerShell:
.\scripts\package_submission.ps1
```

#### Linux Cluster:
```bash
cd submissions
zip -j final_submission_blend.zip sub-05_ranked_candidate_list.npy sub-06_ranked_candidate_list.npy
unzip -l final_submission_blend.zip
```

---

## 3. Optional: 3-Way Tri-Blend with ViEEG (11.00% Validation Top-1)

If you wish to evaluate the 3-Way blend (DINOv2 + V-JEPA2 + ViEEG Spatio-Temporal Convolutions):

```bash
python experiments/blend_exp4_vjepa.py \
  --exp4-run experiments/exp4_tuned_mlp_dino_audio/runs/exp4_dino_clap_run \
  --vjepa-dir experiments/v-jepa/submissions \
  --vieeg-run experiments/exp13_vieeg_dino_audio/runs/exp13_vieeg_run \
  --gt-dir tst_stim \
  --output-dir submissions/tri_blend
```
- **Held-out Split:** Top-1 = **11.00%** (35% Exp4, 60% V-JEPA2, 5% ViEEG)
- **Public Test:** Top-1 = **16.18%**, Top-5 = **41.42%**, Median = **7.8** (with `sub-02` reaching **17.65%**).

---

## 4. Directory Layout Reference

```text
team-fruitbasket-eeg-mindreading-challenge/
├── run_all.bat                         # ONLY script in root (interactive & automated launcher)
├── run_blend.py                        # Standalone 1-Click reproduction Python script
├── README.md                           # Comprehensive documentation & results
├── Commands.md                         # This command cheatsheet
├── CONTEXT.md                          # In-depth architectural & repository reference
├── scripts/                            # All modular batch & powershell scripts
│   ├── run_exp4.bat / .ps1
│   ├── run_vjepa.bat / .ps1
│   ├── run_blend.bat / .ps1
│   ├── run_tune_vjepa.bat
│   ├── package_submission.bat / .ps1
│   ├── run_all.ps1
│   └── run_blend.sh
├── trn/                                # Training EEG recordings (sub-01_trn .. sub-06_trn)
├── tst/                                # Test EEG recordings (sub-01_tst .. sub-06_tst)
├── tst_stim/                           # Ground truth labels for public test evaluation (sub-01..04)
├── features_dinov2/                    # DINOv2 Layer 11 training video features
├── features_test_dinov2/               # DINOv2 Layer 11 test video features
├── features_audio/                     # CLAP train & test audio feature caches
├── submissions/                        # Final competition outputs
│   ├── sub-05_ranked_candidate_list.npy  (Shape: 102x102, candidate IDs in 1001..1102)
│   ├── sub-06_ranked_candidate_list.npy  (Shape: 102x102, candidate IDs in 1001..1102)
│   └── final_submission_blend.zip      # ZIP ready for competition upload
└── experiments/
    ├── blend_exp4_vjepa.py             # Simplex weight optimizer & score blender
    ├── exp4_tuned_mlp_dino_audio/      # Exp4 architecture, training script, & precomputed runs
    └── v-jepa/                         # V-JEPA 2 recipe, features cache, and similarity matrices
```
