@echo off
chcp 65001 > nul 2>&1
setlocal

title ProjectGabriel

reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f > nul 2>&1
for /f %%a in ('echo prompt $E^| cmd /q /v:on') do set "E=%%a"

set "R=%E%[0m"
set "G=%E%[92m"
set "RE=%E%[91m"
set "D=%E%[2m"

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo %RE%  Virtual environment not found. Run setup.bat first.%R%
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo %G%  Starting ProjectGabriel...%R%
echo %D%  Press Ctrl+C to stop.%R%
echo.

python supervisor.py
