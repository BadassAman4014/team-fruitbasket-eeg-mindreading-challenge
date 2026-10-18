@echo off
setlocal
cd /d "%~dp0.."
set KMP_DUPLICATE_LIB_OK=TRUE
if not defined PYTHON_PATH (
    if exist "%USERPROFILE%\.conda\envs\test\python.exe" (
        set PYTHON_PATH=%USERPROFILE%\.conda\envs\test\python.exe
    ) else if exist "%USERPROFILE%\.conda\envs\fruitbasket\python.exe" (
        set PYTHON_PATH=%USERPROFILE%\.conda\envs\fruitbasket\python.exe
    ) else (
        set PYTHON_PATH=python
    )
)

set RUN_NAME=exp4_dino_clap_retrain
set RUN_DIR=experiments\exp4_tuned_mlp_dino_audio\runs\%RUN_NAME%

if exist "%RUN_DIR%" (
    echo Cleaning existing run directory: %RUN_DIR%
    rmdir /s /q "%RUN_DIR%"
)

echo ======================================================================
echo   Starting Exp4 Training (DINOv2 + CLAP MLP on GPU CUDA)
echo ======================================================================

"%PYTHON_PATH%" experiments\exp4_tuned_mlp_dino_audio\train.py ^
  --data-root . ^
  --output-root experiments\exp4_tuned_mlp_dino_audio\runs ^
  --run-name %RUN_NAME% ^
  --subjects 1,2,3,4,5,6 ^
  --prediction-subjects 1,2,3,4,5,6 ^
  --seeds 42,123,999 ^
  --audio-model clap ^
  --fusion concat ^
  --loss multipositive ^
  --dino-pooling mean ^
  --hidden-dim 512 ^
  --embedding-dim 256 ^
  --lr 3e-4 ^
  --weight-decay 0.01 ^
  --dropout 0.3 ^
  --temperature 0.07 ^
  --anti-hub csls ^
  --csls-lambda auto ^
  --device cuda ^
  --predict-test ^
  --report-public-subjects ^
  --final-full-fit

echo.
echo === Exp4 Training Completed Successfully ===
