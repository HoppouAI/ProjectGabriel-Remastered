@echo off
chcp 65001 > nul 2>&1
setlocal

:: Enable ANSI colors
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f > nul 2>&1
for /f %%a in ('echo prompt $E^| cmd /q /v:on') do set "E=%%a"

set "R=%E%[0m"
set "G=%E%[92m"
set "RE=%E%[91m"
set "D=%E%[2m"

title Social Server
cd /d "%~dp0"

if not exist "node_modules" (
    echo %RE%  Dependencies not installed. Run install.bat first.%R%
    echo.
    pause
    exit /b 1
)

if not exist "config.yml" (
    echo %RE%  config.yml not found. Run install.bat first, then edit config.yml.%R%
    echo.
    pause
    exit /b 1
)

echo %G%  Starting Social Server...%R%
echo %D%  Press Ctrl+C to stop.%R%
echo.

node server.js
