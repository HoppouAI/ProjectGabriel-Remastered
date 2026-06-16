#!/usr/bin/env bash
# ProjectGabriel setup for linux and macos. windows users run setup.bat instead.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo
echo "  ===================================================="
echo "       Project Gabriel - Setup"
echo "  ===================================================="
echo

# ---- preflight: native libs the python wheels link against ----
# pyaudio builds against portaudio, opencv needs libGL at runtime, etc. we cant
# install these for you (needs sudo and varies by distro) so just show the hint.
pkg_hint() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "sudo apt install -y portaudio19-dev libsndfile1 libgl1 ffmpeg python3-dev"
    elif command -v dnf >/dev/null 2>&1; then
        echo "sudo dnf install -y portaudio-devel libsndfile mesa-libGL ffmpeg python3-devel"
    elif command -v pacman >/dev/null 2>&1; then
        echo "sudo pacman -S --needed portaudio libsndfile mesa ffmpeg"
    elif command -v zypper >/dev/null 2>&1; then
        echo "sudo zypper install -y portaudio-devel libsndfile Mesa-libGL1 ffmpeg python3-devel"
    else
        echo "(install portaudio, libsndfile, libGL and ffmpeg dev packages for your distro)"
    fi
}

if [ "$(uname -s)" = "Linux" ]; then
    echo "  [system] If a build fails, you probably need these native libs:"
    echo "           $(pkg_hint)"
    echo
fi

# ---- Step 1: uv ----
echo "  [1/4] UV package manager..."
mkdir -p bin

if [ ! -x "bin/uv" ]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "  ERROR: curl is required to download uv. Install curl and re-run."
        exit 1
    fi
    echo "        Downloading UV..."
    export UV_INSTALL_DIR="$SCRIPT_DIR/bin"
    export UV_NO_MODIFY_PATH=1
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$SCRIPT_DIR/bin:$PATH"
UV="$SCRIPT_DIR/bin/uv"
if [ ! -x "$UV" ]; then
    # installer layout differed, fall back to whatever uv is on PATH
    if command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
    else
        echo "  ERROR: UV install failed. Check your internet and try again."
        exit 1
    fi
fi
echo "        OK"
echo

# ---- Step 2: venv ----
echo "  [2/4] Creating Python 3.12 environment..."
if ! "$UV" venv --python 3.12; then
    echo
    echo "  ERROR: Could not create venv. UV auto-downloads Python 3.12 when it"
    echo "  can, otherwise install it from https://www.python.org/downloads/"
    echo
    exit 1
fi
echo "        OK"
echo

# ---- Step 3: hardware ----
echo "  [3/4] Hardware selection"
echo
echo "        1. CPU only"
echo "        2. NVIDIA GPU (CUDA 12.6)"
echo
echo "        (AMD ROCm users: pick 1, then install ROCm torch by hand)"
echo
read -rp "        Choice [1/2]: " GPU
echo

# ---- Step 4: deps ----
echo "  [4/4] Installing dependencies..."
echo "        This takes a few minutes the first time."
if ! "$UV" sync; then
    echo
    echo "  ERROR: Package install failed. See output above."
    echo "  A native lib is often the cause:  $(pkg_hint)"
    exit 1
fi

if [ "${GPU:-1}" = "2" ]; then
    echo
    echo "        Swapping in CUDA PyTorch..."
    if "$UV" pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch torchvision torchaudio --reinstall; then
        echo "        CUDA PyTorch installed."
    else
        echo "        WARNING: CUDA torch failed, CPU torch will be used."
    fi
fi

# ---- config files ----
echo
echo "        Setting up config files..."
mkdir -p data

for f in prompts appends personalities; do
    if [ ! -f "config/prompts/$f.yml" ] && [ -f "config/prompts/$f.yml.example" ]; then
        cp "config/prompts/$f.yml.example" "config/prompts/$f.yml"
        echo "           config/prompts/$f.yml"
    fi
done
if [ ! -f "config/voices.yml" ] && [ -f "config/voices.yml.example" ]; then
    cp "config/voices.yml.example" "config/voices.yml"
    echo "           config/voices.yml"
fi

# ---- done ----
echo
echo "  ===================================================="
echo "       Setup complete"
echo "  ===================================================="
echo

if [ -f "config.yml" ]; then
    echo "  config.yml already exists, skipping wizard."
    echo
    echo "  Run:  ./run.sh"
    echo
    exit 0
fi

echo "  Launching configuration wizard..."
"$UV" run python configurator.py

echo
echo "  Run:  ./run.sh"
echo
exit 0
