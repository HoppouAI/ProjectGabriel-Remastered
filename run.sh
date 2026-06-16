#!/usr/bin/env bash
# Launch ProjectGabriel on linux/macos. windows users run run.bat instead.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -x ".venv/bin/python" ]; then
    echo "  Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

echo "  Starting ProjectGabriel..."
echo "  Press Ctrl+C to stop."
echo

exec .venv/bin/python supervisor.py
