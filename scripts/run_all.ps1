# PowerShell script to run the full GPU pipeline end-to-end
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  TEAM FRUITBASKET: FULL GPU PIPELINE (POWERSHELL)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$ScriptsDir = Join-Path $RepoRoot "scripts"

Write-Host "`n[1/4] Running Exp4 Training..." -ForegroundColor Yellow
& (Join-Path $ScriptsDir "run_exp4.ps1")

Write-Host "`n[2/4] Running V-JEPA 2 Training..." -ForegroundColor Yellow
& (Join-Path $ScriptsDir "run_vjepa.ps1")

Write-Host "`n[3/4] Running 2-Way Blend..." -ForegroundColor Yellow
& (Join-Path $ScriptsDir "run_blend.ps1")

Write-Host "`n[4/4] Packaging Final Submission ZIP..." -ForegroundColor Yellow
& (Join-Path $ScriptsDir "package_submission.ps1")

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "  ALL EXPERIMENTS COMPLETED ON GPU! SUBMISSION ZIP READY IN submissions/" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
