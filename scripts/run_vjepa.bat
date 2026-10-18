@echo off
setlocal
cd /d "%~dp0..\experiments\v-jepa"

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

echo ======================================================================
echo   Starting V-JEPA 2 Training (8 Temporal Sub-Clips + CLAP on GPU)
echo ======================================================================

"%PYTHON_PATH%" simple_mlp\train_recipe.py ^
  --device cuda ^
  --seeds 42,123,999 ^
  --select_on heldout ^
  --anti_hub csls ^
  --final_full_fit ^
  --dropout 0.4 ^
  --modality_dropout 0.2 ^
  --weight_decay 0.005 ^
  --hub_penalty 0.1 ^
  --feat_cache .\vjepa2_frames ^
  --tag vjepa2_frames ^
  --trn_dir ..\..\trn ^
  --tst_dir ..\..\tst ^
  --gt_dir ..\..\tst_stim

echo.
echo === V-JEPA 2 Training Completed Successfully ===
