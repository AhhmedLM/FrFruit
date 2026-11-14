@echo off
setlocal
title FrFruit Studio Workstation
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m frfruit.desktop.win_shell --workspace "%CD%" %*
) else (
    python -m frfruit.desktop.win_shell --workspace "%CD%" %*
)
endlocal
