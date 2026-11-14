# Builds standalone Windows executable: dist\FrFruitStudio.exe
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing dependencies with pip..."
    python -m pip install -e ".[desktop]" pyinstaller
} else {
    Write-Host "Syncing environment with uv..."
    uv sync --extra desktop
    uv pip install pyinstaller
}

uv run pyinstaller --noconfirm --clean windows_app\frfruit_studio.spec
Write-Host ""
Write-Host "Build completed: dist\FrFruitStudio.exe"
