#!/usr/bin/env python3
"""1-Click Reproduction Script for Team Fruitbasket's Winning 2-Way Blend.

Ensemble:
  35.0% Exp4 (DINOv2 + CLAP MLP) + 65.0% V-JEPA 2 (8 Temporal Sub-Clips + CLAP)
  Anti-hubness: CSLS (lambda = 0.15)

Outputs:
  - submissions/sub-05_ranked_candidate_list.npy (102, 102)
  - submissions/sub-06_ranked_candidate_list.npy (102, 102)
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    script = root / "experiments" / "blend_exp4_vjepa.py"
    exp4_run = root / "experiments" / "exp4_tuned_mlp_dino_audio" / "runs" / "exp4_dino_clap_run"
    vjepa_dir = root / "experiments" / "v-jepa" / "submissions"
    gt_dir = root / "tst_stim"
    if not gt_dir.exists():
        gt_dir = root / "inputs" / "tst_stim"
    if not gt_dir.exists():
        gt_dir = root.parent / "tst_stim"
    out_dir = root / "submissions"

    cmd = [
        sys.executable,
        str(script),
        "--exp4-run", str(exp4_run),
        "--vjepa-dir", str(vjepa_dir),
        "--gt-dir", str(gt_dir),
        "--output-dir", str(out_dir),
    ]

    print("=" * 80)
    print("  TEAM FRUITBASKET: Running 2-Way Blend Reproduction")
    print("  Command: " + " ".join(cmd))
    print("=" * 80)
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
