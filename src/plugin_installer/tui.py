"""Terminal UI for the plugin installer.

Renders a full screen picker, lets the user multi select plugins to
install or remove, and shells out to the installer module for the
actual work. Pure stdlib + colorama (already a host dep).
"""
from __future__ import annotations

import shutil
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Optional

from src.plugin_installer.installer import (
    PLUGINS_DIR,
    UV_BIN,
    install_plugin,
    remove_plugin,
    uv_available,
)
from src.plugin_installer.keys import get_key
from src.plugin_installer.sources import (
    DEFAULT_BRANCH,
    DEFAULT_REPO,
    GitHubSource,
    LocalSource,
    PluginInfo,
    PluginSource,
    SourceError,
    read_installed,
)


# ANSI helpers. Same conventions as src/cli.py
class C:
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    INV = "\033[7m"
    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAG = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


CLEAR = "\033[2J\033[H"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
ALT_SCREEN_ON = "\033[?1049h"
ALT_SCREEN_OFF = "\033[?1049l"

# Read the real host api version straight from the loader so this never
# drifts out of sync when PLUGIN_API_VERSION gets bumped. Falls back to a
# current default if the host package isnt importable for some reason.
try:
    from src.plugins.loader import PLUGIN_API_VERSION as HOST_API_VERSION
except Exception:
    HOST_API_VERSION = 4


def _enable_ansi() -> None:
    if sys.platform == "win32":
        try:
            from colorama import just_fix_windows_console
            just_fix_windows_console()
        except ImportError:
            pass


def _term_size() -> tuple[int, int]:
    try:
        s = shutil.get_terminal_size((100, 30))
        return s.columns, s.lines
    except Exception:
        return 100, 30


def _truncate(s: str, n: int) -> str:
    if n <= 0:
        return ""
    return s if len(s) <= n else s[: max(0, n - 1)] + "\u2026"


# --- app state ------------------------------------------------------------


class App:
    def __init__(self) -> None:
        self.sources: list[PluginSource] = [GitHubSource()]
        self.source_idx: int = 0
        self.plugins: list[PluginInfo] = []
        self.installed: dict[str, PluginInfo] = {}
        self.selected: set[str] = set()
        self.cursor: int = 0
        self.filter: str = ""
        self.message: str = ""
        self.message_kind: str = "info"  # info | warn | error
        self.dirty: bool = True

    # ---- source / data wiring -----------------------------------------

    @property
    def source(self) -> PluginSource:
        return self.sources[self.source_idx]

    def add_source(self, src: PluginSource, switch: bool = True) -> None:
        self.sources.append(src)
        if switch:
            self.source_idx = len(self.sources) - 1
        self.selected.clear()
        self.cursor = 0
        self.filter = ""
        self.dirty = True

    def cycle_source(self) -> None:
        if len(self.sources) <= 1:
            self.set_message("only one source loaded. Press 'L' to add a local folder.", "warn")
            return
        self.source_idx = (self.source_idx + 1) % len(self.sources)
        self.selected.clear()
        self.cursor = 0
        self.filter = ""
        self.refresh()

    def refresh(self) -> None:
        self.installed = read_installed(PLUGINS_DIR)
        try:
            self.plugins = self.source.list_plugins()
            self.set_message(f"loaded {len(self.plugins)} plugin(s) from {self.source.label}", "info")
        except SourceError as e:
            self.plugins = []
            self.set_message(str(e), "error")
        if self.cursor >= len(self.visible_plugins()):
            self.cursor = max(0, len(self.visible_plugins()) - 1)
        self.dirty = True

    # ---- helpers -----------------------------------------------------

    def visible_plugins(self) -> list[PluginInfo]:
        if not self.filter:
            return self.plugins
        f = self.filter.lower()
        return [
            p for p in self.plugins
            if f in p.name.lower() or f in p.description.lower()
        ]

    def current(self) -> Optional[PluginInfo]:
        vis = self.visible_plugins()
        if not vis:
            return None
        if self.cursor >= len(vis):
            self.cursor = len(vis) - 1
        return vis[self.cursor]

    def set_message(self, msg: str, kind: str = "info") -> None:
        self.message = msg
        self.message_kind = kind
        self.dirty = True

    # ---- actions -----------------------------------------------------

    def toggle(self) -> None:
        cur = self.current()
        if cur is None:
            return
        if cur.name in self.selected:
            self.selected.remove(cur.name)
        else:
            self.selected.add(cur.name)
        self.dirty = True

    def select_all(self) -> None:
        self.selected = {p.name for p in self.visible_plugins()}
        self.dirty = True

    def select_none(self) -> None:
        self.selected.clear()
        self.dirty = True

    def move(self, delta: int) -> None:
        vis = self.visible_plugins()
        if not vis:
            return
        self.cursor = max(0, min(len(vis) - 1, self.cursor + delta))
        self.dirty = True

    # ---- rendering ---------------------------------------------------

    def draw(self) -> None:
        if not self.dirty:
            return
        cols, rows = _term_size()
        cols = max(60, cols)
        rows = max(20, rows)

        out: list[str] = [CLEAR]
        out.append(self._header(cols))
        out.append("")

        list_rows = max(8, rows - 14)  # leave room for footer + details
        list_lines = self._render_list(cols, list_rows)
        out.extend(list_lines)
        # pad list area so the detail block always sits at the same line
        for _ in range(list_rows - len(list_lines)):
            out.append("")

        out.append("")
        out.extend(self._render_details(cols))
        out.append("")
        out.append(self._render_message(cols))
        out.append(self._footer(cols))

        sys.stdout.write("\n".join(out))
        sys.stdout.flush()
        self.dirty = False

    def _header(self, cols: int) -> str:
        title = "  ProjectGabriel Plugin Installer"
        right = f"plugins/  {PLUGINS_DIR}  "
        bar = "\u2550" * cols
        head = title + " " * max(0, cols - len(title) - len(right)) + right
        head = _truncate(head, cols)
        src_line = f"  Source [{self.source_idx + 1}/{len(self.sources)}]: {self.source.label}"
        if self.filter:
            src_line += f"   filter: '{self.filter}'"
        src_line = _truncate(src_line, cols)
        return f"{C.BOLD}{C.CYAN}{bar}{C.RST}\n{C.BOLD}{head}{C.RST}\n{C.DIM}{src_line}{C.RST}\n{C.CYAN}{bar}{C.RST}"

    def _render_list(self, cols: int, max_rows: int) -> list[str]:
        vis = self.visible_plugins()
        if not vis:
            return [f"  {C.DIM}(no plugins to show){C.RST}"]

        # window the list around the cursor so it stays visible
        if self.cursor < 0:
            self.cursor = 0
        if self.cursor >= len(vis):
            self.cursor = len(vis) - 1
        # simple windowing
        if len(vis) <= max_rows:
            start = 0
        else:
            half = max_rows // 2
            start = max(0, min(self.cursor - half, len(vis) - max_rows))
        end = min(len(vis), start + max_rows)

        lines: list[str] = []
        name_w = max((len(p.name) for p in vis), default=10)
        name_w = min(name_w, 22)
        ver_w = 8

        for i in range(start, end):
            p = vis[i]
            checked = "x" if p.name in self.selected else " "
            inst = self.installed.get(p.name)
            status = ""
            if inst is not None:
                if inst.version != p.version:
                    status = f"{C.YELLOW}upd {inst.version}->{p.version}{C.RST}"
                else:
                    status = f"{C.GREEN}installed{C.RST}"
            api_warn = ""
            if p.api_version > HOST_API_VERSION:
                api_warn = f" {C.YELLOW}!api{p.api_version}{C.RST}"

            cursor_mark = ">" if i == self.cursor else " "
            cursor_color = C.BOLD + C.CYAN if i == self.cursor else ""
            cursor_reset = C.RST if i == self.cursor else ""
            chk_color = C.GREEN if checked == "x" else C.GRAY

            name_disp = _truncate(p.name, name_w).ljust(name_w)
            ver_disp = _truncate(p.version, ver_w).ljust(ver_w)
            desc_room = max(10, cols - (4 + 1 + 1 + 3 + 1 + name_w + 1 + ver_w + 1 + 12))
            desc = _truncate(p.short_desc, desc_room)

            line = (
                f" {cursor_color}{cursor_mark}{cursor_reset} "
                f"{chk_color}[{checked}]{C.RST} "
                f"{C.WHITE}{name_disp}{C.RST}  "
                f"{C.DIM}{ver_disp}{C.RST}  "
                f"{desc}"
            )
            tail = []
            if status:
                tail.append(status)
            if api_warn:
                tail.append(api_warn.strip())
            if tail:
                line += "  " + " ".join(tail)
            lines.append(line)

        if start > 0:
            lines.insert(0, f"  {C.DIM}\u2191 {start} more above{C.RST}")
        if end < len(vis):
            lines.append(f"  {C.DIM}\u2193 {len(vis) - end} more below{C.RST}")
        return lines

    def _render_details(self, cols: int) -> list[str]:
        cur = self.current()
        if cur is None:
            return [f"{C.DIM}  no plugin highlighted{C.RST}"]
        inst = self.installed.get(cur.name)
        out: list[str] = []
        head = f"  {C.BOLD}{cur.name}{C.RST}  v{cur.version}  "
        head += f"{C.DIM}by {cur.author}  api v{cur.api_version}{C.RST}"
        if inst is not None:
            if inst.version != cur.version:
                head += f"  {C.YELLOW}(installed: v{inst.version}){C.RST}"
            else:
                head += f"  {C.GREEN}(installed){C.RST}"
        if cur.api_version > HOST_API_VERSION:
            head += f"  {C.YELLOW}WARNING: targets api v{cur.api_version}, host is v{HOST_API_VERSION}{C.RST}"
        out.append(head)

        wrap_width = max(40, cols - 4)
        desc_lines = textwrap.wrap(cur.description.strip() or "(no description)", wrap_width) or [""]
        for ln in desc_lines[:3]:
            out.append(f"  {C.DIM}{ln}{C.RST}")

        if cur.requirements:
            reqs = ", ".join(cur.requirements)
            reqs = _truncate(reqs, cols - 14)
            out.append(f"  {C.DIM}Requires:{C.RST} {reqs}")
        else:
            out.append(f"  {C.DIM}Requires: (none){C.RST}")
        return out

    def _render_message(self, cols: int) -> str:
        if not self.message:
            return ""
        col = {"error": C.RED, "warn": C.YELLOW, "info": C.GREEN}.get(self.message_kind, C.WHITE)
        return f"  {col}{_truncate(self.message, cols - 4)}{C.RST}"

    def _footer(self, cols: int) -> str:
        sel = len(self.selected)
        bar = "\u2500" * cols
        line1 = (
            f"  {C.BOLD}\u2191/\u2193{C.RST} move   "
            f"{C.BOLD}Space{C.RST} toggle   "
            f"{C.BOLD}A{C.RST} all   {C.BOLD}N{C.RST} none   "
            f"{C.BOLD}/{C.RST} filter   {C.BOLD}L{C.RST} local folder   "
            f"{C.BOLD}S{C.RST} switch source"
        )
        line2 = (
            f"  {C.BOLD}Enter{C.RST} install ({sel} sel)   "
            f"{C.BOLD}R{C.RST} remove highlighted   "
            f"{C.BOLD}F5{C.RST} reload   "
            f"{C.BOLD}Q{C.RST} quit"
        )
        return f"{C.DIM}{bar}{C.RST}\n{line1}\n{line2}"


# ---- prompts -------------------------------------------------------------


def _read_line(prompt: str) -> str:
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()
    try:
        return input(prompt)
    finally:
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()


def _confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n] " if default_yes else " [y/N] "
    ans = _read_line(prompt + suffix).strip().lower()
    if not ans:
        return default_yes
    return ans.startswith("y")


# ---- screens that pause the picker --------------------------------------


def _exit_alt() -> None:
    sys.stdout.write(SHOW_CURSOR + ALT_SCREEN_OFF)
    sys.stdout.flush()


def _enter_alt() -> None:
    sys.stdout.write(ALT_SCREEN_ON + HIDE_CURSOR + CLEAR)
    sys.stdout.flush()


def _press_any() -> None:
    print()
    print(f"{C.DIM}press any key to continue...{C.RST}")
    get_key()


def _do_install(app: App) -> None:
    targets: list[PluginInfo] = []
    if app.selected:
        # honor selection order from the visible list
        sel_names = app.selected
        for p in app.plugins:
            if p.name in sel_names:
                targets.append(p)
    else:
        cur = app.current()
        if cur is None:
            app.set_message("nothing to install", "warn")
            return
        targets = [cur]

    _exit_alt()
    print()
    print(f"{C.BOLD}Install plan{C.RST}")
    print(f"{C.DIM}{'-' * 40}{C.RST}")
    total_reqs: list[str] = []
    for p in targets:
        suffix = ""
        inst = app.installed.get(p.name)
        if inst is not None:
            if inst.version != p.version:
                suffix = f"  {C.YELLOW}(replacing v{inst.version}){C.RST}"
            else:
                suffix = f"  {C.GREEN}(reinstall same version){C.RST}"
        if p.api_version > HOST_API_VERSION:
            suffix += f"  {C.YELLOW}!api v{p.api_version}{C.RST}"
        reqs = ", ".join(p.requirements) if p.requirements else "no pip deps"
        print(f"  - {C.WHITE}{p.name}{C.RST} v{p.version}{suffix}")
        print(f"      {C.DIM}{reqs}{C.RST}")
        total_reqs.extend(p.requirements)

    if not uv_available():
        print()
        print(f"  {C.YELLOW}note:{C.RST} {UV_BIN} not found. pip dependencies will be SKIPPED.")
        print(f"  Run setup.bat first if you want deps installed for you.")

    print()
    install_deps = True
    if total_reqs and uv_available():
        install_deps = _confirm("Install pip requirements with uv?", True)
    elif not uv_available():
        install_deps = False

    if not _confirm("Proceed with install?", True):
        print(f"{C.DIM}cancelled.{C.RST}")
        _press_any()
        _enter_alt()
        app.set_message("install cancelled", "warn")
        app.dirty = True
        return

    print()
    ok_count = 0
    for i, p in enumerate(targets, 1):
        print(f"{C.BOLD}[{i}/{len(targets)}]{C.RST} {p.name} v{p.version}")
        res = install_plugin(p, app.source, install_deps=install_deps, on_line=lambda l: print(f"  {l}"))
        if res.ok:
            ok_count += 1
            print(f"  {C.GREEN}done{C.RST}")
        else:
            print(f"  {C.RED}failed: {res.error or 'see above'}{C.RST}")
        print()

    summary = f"installed {ok_count}/{len(targets)} plugin(s). Restart Gabriel to load them."
    print(summary)
    if ok_count < len(targets):
        print(f"{C.YELLOW}some installs failed, check logs above.{C.RST}")
    _press_any()
    _enter_alt()

    app.selected.clear()
    app.refresh()
    app.set_message(summary, "info" if ok_count == len(targets) else "warn")


def _do_remove(app: App) -> None:
    cur = app.current()
    if cur is None:
        app.set_message("nothing highlighted", "warn")
        return
    if cur.name not in app.installed:
        app.set_message(f"'{cur.name}' is not installed", "warn")
        return

    _exit_alt()
    print()
    print(f"{C.BOLD}Remove plugin{C.RST}")
    print(f"{C.DIM}{'-' * 40}{C.RST}")
    print(f"  {C.WHITE}{cur.name}{C.RST} v{app.installed[cur.name].version}")
    print(f"  {C.DIM}{PLUGINS_DIR / cur.name} will be deleted.{C.RST}")
    print(f"  {C.DIM}pip dependencies are NOT removed (they may be shared).{C.RST}")
    print()
    if not _confirm(f"Remove '{cur.name}'?", False):
        print(f"{C.DIM}cancelled.{C.RST}")
        _press_any()
        _enter_alt()
        app.set_message("remove cancelled", "warn")
        app.dirty = True
        return

    ok = remove_plugin(cur.name, on_line=lambda l: print(f"  {l}"))
    print()
    _press_any()
    _enter_alt()
    if ok:
        app.set_message(f"removed '{cur.name}'", "info")
    else:
        app.set_message(f"could not remove '{cur.name}'", "error")
    app.refresh()


def _prompt_filter(app: App) -> None:
    _exit_alt()
    try:
        text = _read_line("filter (empty to clear): ").strip()
    finally:
        _enter_alt()
    app.filter = text
    app.cursor = 0
    if text:
        app.set_message(f"filter set to '{text}'", "info")
    else:
        app.set_message("filter cleared", "info")
    app.dirty = True


def _prompt_local(app: App) -> None:
    _exit_alt()
    try:
        path_str = _read_line("path to local plugin / plugins folder: ").strip().strip('"').strip("'")
    finally:
        _enter_alt()
    if not path_str:
        app.set_message("no path entered", "warn")
        return
    path = Path(path_str).expanduser()
    try:
        src = LocalSource(path)
        # validate by listing
        _ = src.list_plugins()
    except SourceError as e:
        app.set_message(str(e), "error")
        return
    app.add_source(src, switch=True)
    app.refresh()


def _prompt_repo(app: App) -> None:
    _exit_alt()
    try:
        repo = _read_line(f"repo (default {DEFAULT_REPO}): ").strip() or DEFAULT_REPO
        branch = _read_line(f"branch (default {DEFAULT_BRANCH}): ").strip() or DEFAULT_BRANCH
    finally:
        _enter_alt()
    src = GitHubSource(repo=repo, branch=branch)
    app.add_source(src, switch=True)
    app.refresh()


# ---- main loop -----------------------------------------------------------


def run() -> int:
    _enable_ansi()
    app = App()

    # banner before alt screen
    print(f"{C.BOLD}{C.CYAN}ProjectGabriel Plugin Installer{C.RST}")
    print(f"{C.DIM}fetching plugin list from {DEFAULT_REPO}@{DEFAULT_BRANCH}...{C.RST}")
    app.refresh()

    sys.stdout.write(ALT_SCREEN_ON + HIDE_CURSOR)
    sys.stdout.flush()
    try:
        return _loop(app)
    finally:
        for s in app.sources:
            try:
                s.close()
            except Exception:
                pass
        sys.stdout.write(SHOW_CURSOR + ALT_SCREEN_OFF)
        sys.stdout.flush()


def _loop(app: App) -> int:
    while True:
        try:
            app.draw()
        except Exception:
            traceback.print_exc()
            return 1

        try:
            key = get_key()
        except (KeyboardInterrupt, EOFError):
            return 0

        if key in ("q", "Q", "ESC"):
            return 0
        elif key == "UP":
            app.move(-1)
        elif key == "DOWN":
            app.move(1)
        elif key == "PGUP":
            app.move(-10)
        elif key == "PGDN":
            app.move(10)
        elif key == "HOME":
            app.cursor = 0
            app.dirty = True
        elif key == "END":
            app.cursor = max(0, len(app.visible_plugins()) - 1)
            app.dirty = True
        elif key == "SPACE":
            app.toggle()
        elif key in ("a", "A"):
            app.select_all()
        elif key in ("n", "N"):
            app.select_none()
        elif key == "ENTER":
            _do_install(app)
        elif key in ("r", "R"):
            _do_remove(app)
        elif key in ("s", "S"):
            app.cycle_source()
        elif key in ("l", "L"):
            _prompt_local(app)
        elif key in ("g", "G"):
            _prompt_repo(app)
        elif key == "/":
            _prompt_filter(app)
        elif key == "?":
            app.set_message(
                "keys: arrows move, space toggle, A all, N none, Enter install, R remove, S switch, L local, G repo, / filter, Q quit",
                "info",
            )
        else:
            # ignore unknown keys silently
            pass


if __name__ == "__main__":
    sys.exit(run())
