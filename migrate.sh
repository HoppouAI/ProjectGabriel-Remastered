#!/usr/bin/env bash
# Run the memory migration on linux/macos.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/bin/python" ]; then
    .venv/bin/python scripts/migrate_memories.py
else
    python3 scripts/migrate_memories.py
fi
