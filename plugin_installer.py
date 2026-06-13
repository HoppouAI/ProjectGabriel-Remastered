"""ProjectGabriel - Plugin Installer entry point.

Launches the TUI for browsing, installing, and removing plugins from
the public plugin repo (or a local folder). Run after setup.bat has
created the venv. Most users will launch this via plugins.bat.
"""
import sys

from src.plugin_installer.tui import run


if __name__ == "__main__":
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        sys.exit(0)
