@echo off
chcp 65001 > nul 2>&1
setlocal enabledelayedexpansion

title ProjectGabriel Setup
cd /d "%~dp0"
cls

echo.
echo   ====================================================
echo        Project Gabriel - Setup
echo   ====================================================
echo.

:: ---- Step 1: UV ----
echo   [1/4] UV package manager...

if not exist "bin" mkdir "bin"

if not exist "bin\uv.exe" (
    echo        Downloading UV...
    set "UV_INSTALL_DIR=%~dp0bin"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if not exist "bin\uv.exe" (
        echo   ERROR: UV download failed. Check your internet and try again.
        pause
        exit /b 1
    )
)
set "PATH=%~dp0bin;%PATH%"
echo        OK
echo.

:: ---- Step 2: venv ----
echo   [2/4] Creating Python 3.12 environment...

uv venv --python 3.12
if %errorlevel% neq 0 (
    echo.
    echo   ERROR: Could not create venv. UV will auto-download
    echo   Python 3.12 if you have it in your PATH or py launcher.
    echo   Otherwise install it from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo        OK
echo.

:: ---- Step 3: hardware ----
echo   [3/4] Hardware selection
echo.
echo        1. CPU only
echo        2. NVIDIA GPU
echo.
choice /C 12 /M "        Choice"
set "GPU=%errorlevel%"
echo.

:: ---- Step 4: deps ----
echo   [4/4] Installing dependencies...
echo        This takes a few minutes the first time.

uv sync
if %errorlevel% neq 0 (
    echo.
    echo   ERROR: Package install failed. See output above.
    pause
    exit /b 1
)

if "%GPU%"=="2" goto cuda_torch
goto torch_done

:cuda_torch
echo.
echo        Swapping in CUDA PyTorch ^(CUDA 12.8^)...
uv pip install --index-url https://download.pytorch.org/whl/cu128 ^
    torch torchvision torchaudio --reinstall
if %errorlevel% neq 0 (
    echo        WARNING: CUDA torch install failed. See output above.
    goto torch_done
)

echo        Testing that this build actually runs on your GPU...
.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'no cuda device'; x = torch.rand(64, 64, device='cuda'); (x @ x).sum().item(); print('        CUDA OK:', torch.cuda.get_device_name(0))"
if %errorlevel% equ 0 goto torch_done

echo.
echo        The CUDA 12.8 build doesn't work on this GPU, trying CUDA 12.6...
uv pip install --index-url https://download.pytorch.org/whl/cu126 ^
    torch torchvision torchaudio --reinstall
if %errorlevel% neq 0 (
    echo        WARNING: CUDA torch install failed. See output above.
    goto torch_done
)

echo        Testing that this build actually runs on your GPU...
.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'no cuda device'; x = torch.rand(64, 64, device='cuda'); (x @ x).sum().item(); print('        CUDA OK:', torch.cuda.get_device_name(0))"
if %errorlevel% equ 0 goto torch_done

echo.
echo        WARNING: neither CUDA build works on this machine.
echo        Falling back to CPU torch so everything still runs.
uv pip install torch torchvision torchaudio --reinstall

:torch_done

:: ---- config files ----
echo.
echo        Setting up config files...

if not exist "data" mkdir "data"

for %%f in (prompts appends personalities) do (
    if not exist "config\prompts\%%f.yml" (
        if exist "config\prompts\%%f.yml.example" (
            copy /y "config\prompts\%%f.yml.example" "config\prompts\%%f.yml" > nul
            echo           config\prompts\%%f.yml
        )
    )
)
if not exist "config\voices.yml" (
    if exist "config\voices.yml.example" (
        copy /y "config\voices.yml.example" "config\voices.yml" > nul
        echo           config\voices.yml
    )
)

:: ---- done ----
echo.
echo   ====================================================
echo        Setup complete
echo   ====================================================
echo.

if exist "config.yml" (
    echo   config.yml already exists, skipping wizard.
    echo.
    echo   Run:  run.bat
    echo.
    pause
    exit /b 0
)

echo   Launching configuration wizard...
.venv\Scripts\python.exe configurator.py

echo.
echo   Run:  run.bat
echo.
pause
exit /b 0
