$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

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

$RunName = "exp4_dino_clap_retrain"
$RunDir = Join-Path $RepoRoot "experiments\exp4_tuned_mlp_dino_audio\runs\$RunName"
if (Test-Path $RunDir) {
    Write-Host "Cleaning existing run directory: $RunDir" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $RunDir
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Starting Exp4 Training (DINOv2 + CLAP MLP on GPU CUDA)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

& $PythonPath experiments\exp4_tuned_mlp_dino_audio\train.py `
  --data-root . `
  --output-root experiments\exp4_tuned_mlp_dino_audio\runs `
  --run-name $RunName `
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

Write-Host "`n=== Exp4 Training Completed Successfully ===" -ForegroundColor Green
