@echo off
chcp 65001 > nul 2>&1
setlocal

:: Enable ANSI colors
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f > nul 2>&1
for /f %%a in ('echo prompt $E^| cmd /q /v:on') do set "E=%%a"

set "R=%E%[0m"
set "G=%E%[92m"
set "Y=%E%[93m"
set "C=%E%[96m"
set "D=%E%[2m"
set "B=%E%[1m"
set "RE=%E%[91m"

title Social Server Setup
cd /d "%~dp0"
cls

echo.
echo %C%%B%  ====================================================%R%
echo %C%%B%       ProjectGabriel Social Server Setup%R%
echo %C%%B%  ====================================================%R%
echo.

:: Check for Node.js
echo %B%  [1/3]%R%%B% Checking for Node.js...%R%
echo.

where node > nul 2>&1
if %errorlevel% neq 0 (
    echo %RE%        Node.js not found. Please install Node.js 18 or newer.%R%
    echo %D%        Download from: https://nodejs.org/%R%
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('node -v') do set "NODE_VER=%%v"
echo %D%        Found Node.js %NODE_VER%%R%
echo.

:: Install dependencies
echo %B%  [2/3]%R%%B% Installing dependencies...%R%
echo.

call npm install
if %errorlevel% neq 0 (
    echo.
    echo %RE%        npm install failed. Check the output above for errors.%R%
    echo.
    pause
    exit /b 1
)
echo.

:: Copy config if it doesn't exist
echo %B%  [3/3]%R%%B% Setting up config...%R%
echo.

if exist "config.yml" (
    echo %D%        config.yml already exists, skipping.%R%
) else (
    copy config.yml.example config.yml > nul
    echo %G%        Created config.yml from example.%R%
    echo %Y%        Open config.yml and set your admin_key and API keys.%R%
)
echo.

:: Done
echo %C%%B%  ====================================================%R%
echo %G%%B%       Setup complete!%R%
echo %C%%B%  ====================================================%R%
echo.
echo %D%  Next steps:%R%
echo %D%    1. Edit config.yml with your settings%R%
echo %D%    2. Run  run.bat  to start the server%R%
echo.
pause
