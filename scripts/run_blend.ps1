$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

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
Write-Host "  Running 2-Way Simplex Blend (Exp4 GPU + V-JEPA 2 GPU)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

& $PythonPath experiments\blend_exp4_vjepa.py `
  --exp4-run experiments\exp4_tuned_mlp_dino_audio\runs\exp4_dino_clap_retrain `
  --vjepa-dir experiments\v-jepa\submissions `
  --gt-dir tst_stim `
  --output-dir submissions

Write-Host "`n=== 2-Way Blend Completed Successfully ===" -ForegroundColor Green
