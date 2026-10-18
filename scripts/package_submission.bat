@echo off
setlocal
cd /d "%~dp0.."

echo ======================================================================
echo   Packaging Final Competition Submission ZIP
echo ======================================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0package_submission.ps1"
echo === Packaging Complete ===
