$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location (Join-Path $RepoRoot "experiments\v-jepa")

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$TestEnv = Join-Path $HOME ".conda\envs\test\python.exe"
$FruitbasketEnv = Join-Path $HOME ".conda\envs\fruitbasket\python.exe"
if (Test-Path $TestEnv) {
    $PythonPath = $TestEnv
} elseif (Test-Path $FruitbasketEnv) {
    $PythonPath = $FruitbasketEnv
} else {
    $PythonPath = "python"
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Starting V-JEPA 2 Training (8 Temporal Sub-Clips + CLAP on GPU)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

& $PythonPath simple_mlp\train_recipe.py `
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

Write-Host "`n=== V-JEPA 2 Training Completed Successfully ===" -ForegroundColor Green
