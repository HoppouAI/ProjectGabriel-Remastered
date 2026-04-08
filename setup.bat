@echo off
chcp 65001 > nul 2>&1
setlocal enabledelayedexpansion

title ProjectGabriel Setup
cd /d "%~dp0"
cls

echo.
echo   ====================================================
echo        Project Gabriel - Remaster Setup Wizard
echo   ====================================================
echo.
echo   This will set up a virtual environment and install
echo   all dependencies for you.
echo.
echo  Press any key to start...
pause > nul
echo.

:: =======================================================
:: Step 1 - UV
:: =======================================================
echo   [1/5] Setting up UV package manager...
echo.

if not exist "bin" mkdir "bin"

if exist "bin\uv.exe" (
    echo        UV already installed in bin\
) else (
    echo        Installing UV to local bin folder...
    set "UV_INSTALL_DIR=%~dp0bin"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if not exist "bin\uv.exe" (
        echo.
        echo   Error: UV download failed. Check your internet connection and try again.
        echo.
        pause
        exit /b 1
    )
    echo        UV installed.
)

set "PATH=%~dp0bin;%PATH%"
echo.

:: =======================================================
:: Step 2 - Virtual environment
:: =======================================================
echo   [2/5] Creating Python 3.12 virtual environment...
echo.

uv venv --python 3.12
if %errorlevel% neq 0 (
    echo.
    echo   Error: Could not create virtual environment.
    echo   Make sure Python 3.12 is installed: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo.

:: =======================================================
:: Step 3 - Hardware selection
:: =======================================================
echo   [3/5] Hardware selection...
echo.
echo        Select your hardware:
echo.
echo          1.  NVIDIA GPU - installs CUDA PyTorch for better vision performance
echo          2.  CPU Only
echo.
choice /C 12 /M "        Enter your choice"
set "GPU_CHOICE=%errorlevel%"
echo.

:: =======================================================
:: Step 4 - Install dependencies
:: =======================================================
echo   [4/5] Installing dependencies...
echo        This can take a few minutes the first time.
echo.

uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo   Error: Package installation failed. See output above for details.
    echo.
    pause
    exit /b 1
)
echo.
echo        All packages installed.
echo.

if "%GPU_CHOICE%"=="1" goto install_cuda
goto setup_config

:install_cuda
echo        Uninstalling default CPU torch...
uv pip uninstall torch torchvision torchaudio
echo.
echo        Installing CUDA PyTorch, this will take a few minutes...
echo.
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
if %errorlevel% neq 0 (
    echo.
    echo        CUDA install failed. You can retry manually later:
    echo        .venv\Scripts\activate
    echo        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
) else (
    echo.
    echo        CUDA PyTorch installed.
)
echo.

:setup_config
:: =======================================================
:: Step 5 - Config files
:: =======================================================
echo   [5/5] Setting up config files...
echo.

set "NEED_KEY=0"

if not exist "config.yml" (
    copy /y "config.yml.example" "config.yml" > nul
    echo        Created config.yml
    set "NEED_KEY=1"
) else (
    echo        config.yml already exists, skipping.
)

for %%f in (prompts appends personalities) do (
    if not exist "config\prompts\%%f.yml" (
        if exist "config\prompts\%%f.yml.example" (
            copy /y "config\prompts\%%f.yml.example" "config\prompts\%%f.yml" > nul
            echo        Created config\prompts\%%f.yml
        )
    )
)

if not exist "config\voices.yml" (
    if exist "config\voices.yml.example" (
        copy /y "config\voices.yml.example" "config\voices.yml" > nul
        echo        Created config\voices.yml
    )
)
echo.

:: =======================================================
:: Done
:: =======================================================
echo   ====================================================
echo           Setup complete!
echo   ====================================================
echo.
echo   To run Gabriel:
echo     .venv\Scripts\activate
echo     python supervisor.py
echo.

if "%NEED_KEY%"=="1" (
    echo   Open config.yml and add your Gemini API key before starting.
    echo.
)

pause

exit /b 0
