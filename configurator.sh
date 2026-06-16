#!/usr/bin/env bash
# Open the ProjectGabriel configurator on linux/macos.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -x ".venv/bin/python" ]; then
    echo "  Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

exec .venv/bin/python configurator.py
