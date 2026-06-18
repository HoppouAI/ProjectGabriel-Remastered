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
# pyaudio compiles a C extension (needs a compiler) and links against portaudio,
# opencv wants libGL at runtime, etc. we cant install these for you (needs sudo
# and varies by distro) so just show the hint. build-essential / base-devel pull
# in gcc+make, without them pyaudio dies with "command 'cc' failed".
pkg_hint() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "sudo apt install -y build-essential portaudio19-dev libsndfile1 libgl1 ffmpeg python3-dev"
    elif command -v dnf >/dev/null 2>&1; then
        echo "sudo dnf install -y gcc gcc-c++ make portaudio-devel libsndfile mesa-libGL ffmpeg python3-devel"
    elif command -v pacman >/dev/null 2>&1; then
        echo "sudo pacman -S --needed base-devel portaudio libsndfile mesa ffmpeg"
    elif command -v zypper >/dev/null 2>&1; then
        echo "sudo zypper install -y gcc gcc-c++ make portaudio-devel libsndfile Mesa-libGL1 ffmpeg python3-devel"
    elif command -v nix-shell >/dev/null 2>&1; then
        echo "nix-shell shell.nix"
    else
        echo "(install a C compiler plus portaudio, libsndfile, libGL and ffmpeg dev packages for your distro)"
    fi
}

if [ "$(uname -s)" = "Linux" ]; then
    # the nix shell (shell.nix) bundles the C compiler and native libs the
    # wheels need. force it when there is no system cc (eg NixOS, or any
    # minimal distro without build tools) and nix-shell is available. on
    # normal distros with a compiler, fall through to the package-manager
    # hint below instead.
    if ! command -v cc >/dev/null 2>&1 && command -v nix-shell >/dev/null 2>&1 && [ -z "${IN_NIX_SHELL:-}" ]; then
        echo "  [system] No C compiler found. The nix shell provides one plus the"
        echo "           native libs (portaudio, libsndfile, ...) the build needs."
        if [ -f "$SCRIPT_DIR/shell.nix" ]; then
            echo
            echo "  Enter the nix shell first, then re-run this script:"
            echo "    nix-shell shell.nix"
        else
            echo
            echo "  No shell.nix found. See the README for nix shell setup instructions."
        fi
        echo
        exit 1
    fi
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
if [ "${GPU:-1}" = "2" ]; then
    TORCH_EXTRA="cu126"
    echo "  [4/4] Installing dependencies (CUDA PyTorch)..."
else
    TORCH_EXTRA="cpu"
    echo "  [4/4] Installing dependencies (CPU PyTorch)..."
fi
echo "        This takes a few minutes the first time."
if ! "$UV" sync --extra "$TORCH_EXTRA"; then
    echo
    echo "  ERROR: Package install failed. See output above."
    echo "  A native lib is often the cause:  $(pkg_hint)"
    exit 1
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
# use the venv python directly, not `uv run`, which would re-sync the default
# (extra-less) env and uninstall the torch build we just selected.
.venv/bin/python configurator.py

echo
echo "  Run:  ./run.sh"
echo
exit 0
