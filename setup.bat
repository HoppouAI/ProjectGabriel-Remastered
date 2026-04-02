@echo off
chcp 65001 > nul 2>&1
setlocal enabledelayedexpansion

:: Enable ANSI colors
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f > nul 2>&1
for /f %%a in ('echo prompt $E^| cmd /q /v:on') do set "E=%%a"

set "R=%E%[0m"
set "G=%E%[92m"
set "Y=%E%[93m"
set "C=%E%[96m"
set "W=%E%[97m"
set "D=%E%[2m"
set "B=%E%[1m"
set "RE=%E%[91m"

title ProjectGabriel Setup
cd /d "%~dp0"
cls

echo.
echo %C%%B%  ====================================================%R%
echo %C%%B%       Project Gabriel - Remaster Setup Wizard%R%
echo %C%%B%  ====================================================%R%
echo.
echo %D%  This will set up a virtual environment and install%R%
echo %D%  all dependencies for you.%R%
echo.
echo  Press any key to start...
pause > nul
echo.

:: =======================================================
:: Step 1 - UV
:: =======================================================
echo %W%%B%  [1/5]%R%%B% Setting up UV package manager...%R%
echo.

if not exist "bin" mkdir "bin"

if exist "bin\uv.exe" (
    echo %D%        UV already installed in bin\%R%
) else (
    echo %D%        Installing UV to local bin folder...%R%
    set "UV_INSTALL_DIR=%~dp0bin"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if not exist "bin\uv.exe" (
        echo.
        echo %RE%  Error: UV download failed. Check your internet connection and try again.%R%
        echo.
        pause
        exit /b 1
    )
    echo %G%        UV installed.%R%
)

set "PATH=%~dp0bin;%PATH%"
echo.

:: =======================================================
:: Step 2 - Virtual environment
:: =======================================================
echo %W%%B%  [2/5]%R%%B% Creating Python 3.12 virtual environment...%R%
echo.

uv venv --python 3.12
if %errorlevel% neq 0 (
    echo.
    echo %RE%  Error: Could not create virtual environment.%R%
    echo %D%  Make sure Python 3.12 is installed: https://www.python.org/downloads/%R%
    echo.
    pause
    exit /b 1
)
echo.

:: =======================================================
:: Step 3 - Hardware selection
:: =======================================================
echo %W%%B%  [3/5]%R%%B% Hardware selection...%R%
echo.
echo %W%        Select your hardware:%R%
echo.
echo %Y%          1.  NVIDIA GPU - installs CUDA PyTorch for better vision performance%R%
echo %Y%          2.  CPU Only%R%
echo.
choice /C 12 /M "        Enter your choice"
set "GPU_CHOICE=%errorlevel%"
echo.

:: =======================================================
:: Step 4 - Install dependencies
:: =======================================================
echo %W%%B%  [4/5]%R%%B% Installing dependencies...%R%
echo %D%        This can take a few minutes the first time.%R%
echo.

uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo %RE%  Error: Package installation failed. See output above for details.%R%
    echo.
    pause
    exit /b 1
)
echo.
echo %G%        All packages installed.%R%
echo.

if "%GPU_CHOICE%"=="1" goto install_cuda
goto setup_config

:install_cuda
echo %D%        Uninstalling default CPU torch...%R%
uv pip uninstall torch torchvision torchaudio
echo.
echo %D%        Installing CUDA PyTorch, this will take a few minutes...%R%
echo.
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
if %errorlevel% neq 0 (
    echo.
    echo %Y%        CUDA install failed. You can retry manually later:%R%
    echo %D%        .venv\Scripts\activate%R%
    echo %D%        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126%R%
) else (
    echo.
    echo %G%        CUDA PyTorch installed.%R%
)
echo.

:setup_config
:: =======================================================
:: Step 5 - Config files
:: =======================================================
echo %W%%B%  [5/5]%R%%B% Setting up config files...%R%
echo.

set "NEED_KEY=0"

if not exist "config.yml" (
    copy /y "config.yml.example" "config.yml" > nul
    echo %G%        Created config.yml%R%
    set "NEED_KEY=1"
) else (
    echo %D%        config.yml already exists, skipping.%R%
)

for %%f in (prompts appends personalities) do (
    if not exist "config\prompts\%%f.yml" (
        if exist "config\prompts\%%f.yml.example" (
            copy /y "config\prompts\%%f.yml.example" "config\prompts\%%f.yml" > nul
            echo %G%        Created config\prompts\%%f.yml%R%
        )
    )
)

if not exist "config\voices.yml" (
    if exist "config\voices.yml.example" (
        copy /y "config\voices.yml.example" "config\voices.yml" > nul
        echo %G%        Created config\voices.yml%R%
    )
)
echo.

:: =======================================================
:: Done
:: =======================================================
echo %C%%B%  ====================================================%R%
echo %G%%B%           Setup complete!%R%
echo %C%%B%  ====================================================%R%
echo.
echo %W%  To run Gabriel:%R%
echo %D%    .venv\Scripts\activate%R%
echo %D%    python supervisor.py%R%
echo.

if "%NEED_KEY%"=="1" (
    echo %Y%  Open config.yml and add your Gemini API key before starting.%R%
    echo.
)

pause
        echo %G%        Created config\voices.yml%R%
    )
)
echo.

:: =======================================================
:: Done
:: =======================================================
echo %C%%B%  ====================================================%R%
echo %G%%B%           Setup complete!%R%
echo %C%%B%  ====================================================%R%
echo.
echo %W%  To run Gabriel:%R%
echo %D%    .venv\Scripts\activate%R%
echo %D%    python supervisor.py%R%
echo.

if "%NEED_KEY%"=="1" (
    echo %Y%  Open config.yml and add your Gemini API key before starting.%R%
    echo.
)

pause


exit /b 0
