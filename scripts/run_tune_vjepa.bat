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
echo   Running V-JEPA Hyperparameter & Anti-Hubness Tuning Sweep
echo ======================================================================
"%PYTHON_PATH%" tune_vjepa_eval.py
echo === Tuning Sweep Completed ===
