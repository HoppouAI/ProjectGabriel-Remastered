@echo off
chcp 65001 > nul 2>&1
setlocal

title ProjectGabriel - Configurator
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo   Virtual environment not found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python configurator.py
pause
