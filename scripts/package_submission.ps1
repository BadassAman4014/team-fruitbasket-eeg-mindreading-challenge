# PowerShell script to package final competition submission zip
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$SubDir = Join-Path $RepoRoot "submissions"
$ZipPath = Join-Path $SubDir "final_submission_blend.zip"

if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}

Write-Host "Creating competition zip package at: $ZipPath" -ForegroundColor Cyan

# Package with files at the root of the ZIP
Set-Location $SubDir
Compress-Archive -Path "sub-05_ranked_candidate_list.npy", "sub-06_ranked_candidate_list.npy" -DestinationPath $ZipPath -Force

Write-Host "Verifying ZIP contents..." -ForegroundColor Green
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
foreach ($entry in $zip.Entries) {
    Write-Host "  -> $($entry.FullName) ($($entry.Length) bytes)"
}
$zip.Dispose()

Write-Host "`nSubmission ZIP ready for upload: $ZipPath" -ForegroundColor Green
