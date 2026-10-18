@echo off
setlocal
cd /d "%~dp0.."

if not defined PYTHON_PATH (
    if exist "%USERPROFILE%\.conda\envs\test\python.exe" (
        set PYTHON_PATH=%USERPROFILE%\.conda\envs\test\python.exe
    ) else if exist "%USERPROFILE%\.conda\envs\fruitbasket\python.exe" (
        set PYTHON_PATH=%USERPROFILE%\.conda\envs\fruitbasket\python.exe
    ) else (
        set PYTHON_PATH=python
    )
)

echo ======================================================================
echo   Running 2-Way Simplex Blend (Exp4 GPU + V-JEPA 2 GPU)
echo ======================================================================

"%PYTHON_PATH%" experiments\blend_exp4_vjepa.py ^
  --exp4-run experiments\exp4_tuned_mlp_dino_audio\runs\exp4_dino_clap_retrain ^
  --vjepa-dir experiments\v-jepa\submissions ^
  --gt-dir tst_stim ^
  --output-dir submissions

echo.
echo === 2-Way Blend Completed Successfully ===
