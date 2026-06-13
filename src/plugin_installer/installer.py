"""Install / remove plugins and their pip dependencies.

Runs `bin/uv.exe pip install ...` against the active venv to grab
plugin requirements. uv auto detects `.venv/` so we just shell out and
let it work.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.plugin_installer.sources import PluginInfo, PluginSource


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_DIR = PROJECT_ROOT / "plugins"
UV_BIN = PROJECT_ROOT / "bin" / ("uv.exe" if sys.platform == "win32" else "uv")


@dataclass
class InstallResult:
    name: str
    ok: bool
    copied: bool = False
    deps_ok: bool = True
    skipped_deps: bool = False
    error: Optional[str] = None
    log: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.log is None:
            self.log = []


# ---- pip install via uv --------------------------------------------------


def uv_available() -> bool:
    return UV_BIN.exists()


def install_requirements(
    reqs: list[str],
    on_line: Optional[Callable[[str], None]] = None,
) -> tuple[bool, list[str]]:
    """Run `uv pip install <reqs>`. Returns (ok, captured lines).

    If uv is missing we report that as a non-fatal warning (deps_ok =
    False) and let the caller decide what to do, the plugin folder is
    still copied either way.
    """
    if not reqs:
        return True, []
    if not UV_BIN.exists():
        msg = f"uv not found at {UV_BIN}, skipping pip install (run setup.bat)"
        if on_line:
            on_line(msg)
        return False, [msg]

    cmd = [str(UV_BIN), "pip", "install", *reqs]
    if on_line:
        on_line("$ " + " ".join(cmd))

    captured: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as e:
        msg = f"failed to launch uv: {e}"
        if on_line:
            on_line(msg)
        return False, [msg]

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        captured.append(line)
        if on_line:
            on_line(line)
    rc = proc.wait()
    return rc == 0, captured


# ---- install / remove ----------------------------------------------------


def install_plugin(
    info: PluginInfo,
    source: PluginSource,
    *,
    install_deps: bool = True,
    on_line: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """Materialize the plugin into ./plugins/ and pip install its reqs."""
    res = InstallResult(name=info.name, ok=False)
    log = res.log

    def emit(line: str) -> None:
        log.append(line)
        if on_line:
            on_line(line)

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

    emit(f"copying {info.name} v{info.version} from {info.source_label}")
    try:
        dest = source.materialize(info, PLUGINS_DIR)
    except Exception as e:
        res.error = f"copy failed: {e}"
        emit(res.error)
        return res
    res.copied = True
    emit(f"  -> {dest}")

    if info.requirements:
        if install_deps:
            emit(f"installing {len(info.requirements)} requirement(s) via uv")
            ok, _ = install_requirements(info.requirements, on_line=on_line)
            res.deps_ok = ok
            if not ok:
                emit("  requirement install failed (plugin folder is still in place)")
        else:
            res.skipped_deps = True
            emit("skipped requirements per user choice")
    else:
        emit("no pip requirements")

    res.ok = res.copied and (res.deps_ok or res.skipped_deps)
    return res


def remove_plugin(name: str, on_line: Optional[Callable[[str], None]] = None) -> bool:
    """Delete plugins/<name>/. Does not touch installed pip deps because
    they may be shared with the host or other plugins."""
    target = PLUGINS_DIR / name
    if not target.exists():
        if on_line:
            on_line(f"plugin '{name}' is not installed")
        return False
    try:
        shutil.rmtree(target)
        if on_line:
            on_line(f"removed {target}")
        return True
    except Exception as e:
        if on_line:
            on_line(f"could not remove {target}: {e}")
        return False
