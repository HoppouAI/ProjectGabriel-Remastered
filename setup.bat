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

title ProjectGabriel Installer
cd /d "%~dp0"
cls

echo.
echo %C%%B%  +-----------------------------------------------+%R%
echo %C%%B%  ^|          Project Gabriel - Setup              ^|%R%
echo %C%%B%  +-----------------------------------------------+%R%
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

set "UV_DIR=%~dp0.uv"
set "UV=%UV_DIR%\uv.exe"

if exist "%UV%" (
    echo %D%        UV already present, skipping download.%R%
) else (
    echo %D%        Downloading UV to .uv folder, please wait...%R%
    if not exist "%UV_DIR%" mkdir "%UV_DIR%"

    set "UV_INSTALL_DIR=%UV_DIR%"
    set "UV_INSTALLER_NO_MODIFY_PATH=1"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression" > nul 2>&1

    if not exist "%UV%" (
        if exist "%UV_DIR%\bin\uv.exe" (
            set "UV=%UV_DIR%\bin\uv.exe"
        ) else (
            echo.
            echo %RE%  Error: UV download failed.%R%
            echo %D%  Check your internet connection and try again.%R%
            echo.
            pause
            exit /b 1
        )
    )

    echo %G%        UV downloaded.%R%
)
echo.

:: =======================================================
:: Step 2 - Virtual environment
:: =======================================================
echo %W%%B%  [2/5]%R%%B% Creating Python 3.12 virtual environment...%R%
echo.

if exist "%~dp0.venv\Scripts\python.exe" (
    echo %D%        Virtual environment already exists, skipping.%R%
) else (
    "%UV%" venv --python 3.12
    if !errorlevel! neq 0 (
        echo.
        echo %RE%  Error: Could not create virtual environment.%R%
        echo %D%  Make sure Python 3.12 is installed: https://www.python.org/downloads/%R%
        echo.
        pause
        exit /b 1
    )
    echo.
    echo %G%        Virtual environment created.%R%
)
echo.

:: =======================================================
:: Step 3 - Install dependencies
:: =======================================================
echo %W%%B%  [3/5]%R%%B% Installing dependencies...%R%
echo %D%        This can take a few minutes the first time.%R%
echo.

"%UV%" pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if !errorlevel! neq 0 (
    echo.
    echo %RE%  Error: Package installation failed. See the output above for details.%R%
    echo.
    pause
    exit /b 1
)
echo.
echo %G%        All packages installed.%R%
echo.

:: =======================================================
:: Step 4 - NVIDIA GPU check
:: =======================================================
echo %W%%B%  [4/5]%R%%B% Checking for NVIDIA GPU...%R%
echo.

set "NVIDIA=0"
nvidia-smi > nul 2>&1
if !errorlevel! equ 0 set "NVIDIA=1"

if "!NVIDIA!"=="1" (
    echo %G%        NVIDIA GPU detected.%R%
    echo.
    echo %W%        Install CUDA-accelerated PyTorch?%R%
    echo %D%        Recommended if you plan to use person or face tracking.%R%
    echo.
    echo %Y%          [1]  Yes, install CUDA version%R%
    echo %Y%          [2]  No, keep the standard CPU version%R%
    echo.
    set /p "GPUC=        Enter choice (1 or 2): "
    echo.

    if "!GPUC!"=="1" (
        echo %D%        Removing CPU PyTorch first...%R%
        "%UV%" pip uninstall --python ".venv\Scripts\python.exe" torch torchvision torchaudio -y > nul 2>&1
        echo %D%        Installing CUDA PyTorch, this may take a few minutes...%R%
        echo.
        "%UV%" pip install --python ".venv\Scripts\python.exe" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
        if !errorlevel! neq 0 (
            echo.
            echo %Y%        CUDA install failed. You can retry it manually later with:%R%
            echo %D%        .venv\Scripts\activate%R%
            echo %D%        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126%R%
        ) else (
            echo.
            echo %G%        CUDA PyTorch installed.%R%
        )
    ) else (
        echo %D%        Keeping CPU-only PyTorch.%R%
    )
) else (
    echo %D%        No NVIDIA GPU found, keeping CPU-only PyTorch.%R%
)
echo.

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
echo %C%%B%  +-----------------------------------------------+%R%
echo %G%%B%  ^|           Setup complete!                    ^|%R%
echo %C%%B%  +-----------------------------------------------+%R%
echo.
echo %W%  To run Gabriel:%R%
echo %D%    .venv\Scripts\activate%R%
echo %D%    python supervisor.py%R%
echo.

if "!NEED_KEY!"=="1" (
    echo %Y%  Open config.yml and add your Gemini API key before starting!%R%
    echo.
)

pause
