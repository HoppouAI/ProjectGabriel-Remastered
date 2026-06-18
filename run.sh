#!/usr/bin/env bash
# Launch ProjectGabriel on linux/macos. windows users run run.bat instead.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# The native libs (portaudio, libGL, ...) the venv links against are provided
# by the nix shell (shell.nix). run.sh is non-interactive, so if nix-shell is
# available and we arent already in one, re-exec into it. IN_NIX_SHELL is set
# by nix-shell and prevents a re-exec loop. works on any distro with nix.
if command -v nix-shell >/dev/null 2>&1 && [ -z "${IN_NIX_SHELL:-}" ] && [ -f "$SCRIPT_DIR/shell.nix" ]; then
    exec nix-shell "$SCRIPT_DIR/shell.nix" --run "cd '$SCRIPT_DIR' && exec '$SCRIPT_DIR/run.sh'"
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "  Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# if the Gabriel virtual sinks are loaded (scripts/pipewire-setup.sh up) and the
# user hasnt already pinned a pulse target, route our audio through them so
# VRChat and the AI can hear each other without any pavucontrol fiddling.
if command -v pactl >/dev/null 2>&1; then
    if pactl list short sinks 2>/dev/null | grep -q "Gabriel_Mic"; then
        : "${PULSE_SINK:=Gabriel_Mic}"
        : "${PULSE_SOURCE:=Gabriel_Ears.monitor}"
        export PULSE_SINK PULSE_SOURCE
        echo "  Audio routed through Gabriel_Mic / Gabriel_Ears (PipeWire sinks found)."
    fi
fi

echo "  Starting ProjectGabriel..."
echo "  Press Ctrl+C to stop."
echo

exec .venv/bin/python supervisor.py
