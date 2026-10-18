@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
title Team Fruitbasket - Master Pipeline Launcher

echo ======================================================================
echo       TEAM FRUITBASKET: EEG MIND READING RETRIEVAL PIPELINE
echo ======================================================================
echo.

if not "%1"=="" (
    set choice=%1
    goto process_choice
)

echo Select execution task:
echo   [1] Full GPU Pipeline (Train Exp4 -> Train V-JEPA 2 -> Blend -> Package)
echo   [2] Train Exp4 Only (DINOv2 Layer 11 + CLAP Audio MLP)
echo   [3] Train V-JEPA 2 Only (8 Temporal Sub-Clips + CLAP MLP)
echo   [4] Run 2-Way Simplex Blend (Exp4 + V-JEPA 2)
echo   [5] Package Final Competition Submission ZIP
echo   [6] Instant 1-Click Reproduction (Precomputed Score Matrices)
echo   [0] Exit
echo.
set /p choice="Enter choice [1-6, default: 1]: "
if "%choice%"=="" set choice=1

:process_choice
if "%choice%"=="1" goto full_pipeline
if "%choice%"=="--all" goto full_pipeline
if "%choice%"=="2" goto run_exp4
if "%choice%"=="--exp4" goto run_exp4
if "%choice%"=="3" goto run_vjepa
if "%choice%"=="--vjepa" goto run_vjepa
if "%choice%"=="4" goto run_blend
if "%choice%"=="--blend" goto run_blend
if "%choice%"=="5" goto run_package
if "%choice%"=="--package" goto run_package
if "%choice%"=="6" goto run_instant
if "%choice%"=="--instant" goto run_instant
if "%choice%"=="0" exit /b 0

echo Invalid choice: %choice%. Running full pipeline by default...
goto full_pipeline

:full_pipeline
echo.
echo ======================================================================
echo   [1/4] Retraining Exp4 (DINOv2 + CLAP MLP) on GPU CUDA...
echo ======================================================================
call "%~dp0scripts\run_exp4.bat"
if errorlevel 1 (
    echo [ERROR] Exp4 training failed!
    if "%1"=="" pause
    exit /b 1
)

echo.
echo ======================================================================
echo   [2/4] Retraining V-JEPA 2 (8 Sub-Clips + CLAP) on GPU CUDA...
echo ======================================================================
call "%~dp0scripts\run_vjepa.bat"
if errorlevel 1 (
    echo [ERROR] V-JEPA 2 training failed!
    if "%1"=="" pause
    exit /b 1
)

echo.
echo ======================================================================
echo   [3/4] Blending Exp4 + V-JEPA 2 (Optimal Held-Out Simplex Sweep)...
echo ======================================================================
call "%~dp0scripts\run_blend.bat"
if errorlevel 1 (
    echo [ERROR] 2-Way Blend failed!
    if "%1"=="" pause
    exit /b 1
)

echo.
echo ======================================================================
echo   [4/4] Packaging Competition Submission ZIP...
echo ======================================================================
call "%~dp0scripts\package_submission.bat"

echo.
echo ======================================================================
echo   ALL EXPERIMENTS COMPLETED ON GPU!
echo   Final ZIP ready at: submissions\final_submission_blend.zip
echo ======================================================================
if "%1"=="" pause
exit /b 0

:run_exp4
call "%~dp0scripts\run_exp4.bat"
if "%1"=="" pause
exit /b 0

:run_vjepa
call "%~dp0scripts\run_vjepa.bat"
if "%1"=="" pause
exit /b 0

:run_blend
call "%~dp0scripts\run_blend.bat"
if "%1"=="" pause
exit /b 0

:run_package
call "%~dp0scripts\package_submission.bat"
if "%1"=="" pause
exit /b 0

:run_instant
if not defined PYTHON_PATH (
    if exist "%USERPROFILE%\.conda\envs\test\python.exe" (
        set PYTHON_PATH=%USERPROFILE%\.conda\envs\test\python.exe
    ) else if exist "%USERPROFILE%\.conda\envs\fruitbasket\python.exe" (
        set PYTHON_PATH=%USERPROFILE%\.conda\envs\fruitbasket\python.exe
    ) else (
        set PYTHON_PATH=python
    )
)
"%PYTHON_PATH%" run_blend.py
if "%1"=="" pause
exit /b 0
