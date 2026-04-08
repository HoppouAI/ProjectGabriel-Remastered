@echo off
chcp 65001 > nul 2>&1
setlocal

title ProjectGabriel
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo   Virtual environment not found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo   Starting ProjectGabriel...
echo   Press Ctrl+C to stop.
echo.

python supervisor.py
