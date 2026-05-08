"""CLI formatting utilities for ProjectGabriel."""
import logging
import sys
import platform


def _enable_ansi():
    """Enable ANSI escape code processing on Windows."""
    if sys.platform == "win32":
        try:
            from colorama import just_fix_windows_console
            just_fix_windows_console()
        except ImportError:
            pass
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")


class C:
    """ANSI color/style codes."""
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"
    B_GREEN = "\033[92m"
    B_YELLOW = "\033[93m"
    B_RED = "\033[91m"
    B_CYAN = "\033[96m"
    B_WHITE = "\033[97m"
    B_MAGENTA = "\033[95m"


class ColoredFormatter(logging.Formatter):
    """Compact colored log formatter."""
    LEVELS = {
        logging.DEBUG:    (C.GRAY,           "DEBUG"),
        logging.INFO:     (C.B_GREEN,        " INFO"),
        logging.WARNING:  (C.B_YELLOW,       " WARN"),
        logging.ERROR:    (C.B_RED,          "ERROR"),
        logging.CRITICAL: (C.B_RED + C.BOLD, "FATAL"),
    }

    def format(self, record):
        color, label = self.LEVELS.get(record.levelno, (C.RST, record.levelname[:5]))
        ts = self.formatTime(record, "%H:%M:%S")
        return (
            f"{C.DIM}{ts}{C.RST} "
            f"{color}{label}{C.RST} "
            f"{C.CYAN}{record.name}{C.RST}  "
            f"{record.getMessage()}"
        )


def setup_logging(level=logging.INFO):
    """Replace default logging with colored output."""
    _enable_ansi()
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    root.addHandler(handler)

    # quiet down chatty third party libs that the everyday user does not
    # care about. selfbot internals (TLS fingerprint, user agent string,
    # PyNaCl missing, "Logging in using static token") just spam scrollback.
    # httpx logs every single HTTP call at INFO which floods the terminal
    # when the embedding server is busy, drop it to WARNING.
    for noisy in ("discord.http", "discord.client"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    for chatty in ("httpx", "httpcore"):
        logging.getLogger(chatty).setLevel(logging.WARNING)


_W = 49


def print_startup_info(config):
    """Print configuration summary and component status."""
    _kv("App Name", config.app_name, C.B_CYAN)
    _kv("Model", config.model, C.B_YELLOW)
    _kv("Voice", config.voice, C.B_MAGENTA)
    _kv("TTS", config.tts_provider.replace("_", " ").title(), C.B_WHITE)
    _kv("OSC", f"{config.osc_ip}:{config.osc_port}")
    _kv("Music", config.music_dir)
    print()

    components = [
        ("Tracker", config.tracker_enabled),
        ("Face Tracker", config.face_tracker_enabled),
        ("Wanderer", config.wanderer_enabled),
        ("Memory", config.memory_enabled),
        ("Emotions", config.emotion_enabled),
        ("Vision", config.vision_enabled),
        ("VRChat API", bool(config.vrchat_api_username)),
        ("OBS Overlay", config.obs_enabled),
        ("Music Gen", config.music_gen_enabled),
        ("Web Search", config.web_search_enabled),
        ("Discord Bot", getattr(config, "discord_bot_enabled", False)),
        ("Social", getattr(config, "social_enabled", False)),
    ]

    parts = []
    for name, on in components:
        if on:
            parts.append(f"{C.B_GREEN}\u25cf{C.RST} {name}")
        else:
            parts.append(f"{C.DIM}\u25cb {name}{C.RST}")

    for i in range(0, len(parts), 4):
        print(f"  {'   '.join(parts[i:i + 4])}")
    print()

    # Plugin status, integrated into the same banner block
    _print_plugins_block()

    print(f"  {C.DIM}{'\u2500' * _W}{C.RST}")
    print()


def _print_plugins_block(plugins_dir: str = "plugins"):
    """Inline helper used inside print_startup_info to render the
    Plugins sub-section. Reads plugin.yml + config/tools.yml so it
    stays accurate. Emits nothing if there are no plugin folders."""
    import yaml as _yaml
    from pathlib import Path

    pdir = Path(plugins_dir)
    if not pdir.is_dir():
        return

    tools_path = Path("config/tools.yml")
    plugin_tool_map: dict = {}
    if tools_path.exists():
        try:
            with open(tools_path, "r", encoding="utf-8") as f:
                tdata = _yaml.safe_load(f) or {}
            plugin_tool_map = tdata.get("plugin_tools") or {}
        except Exception:
            plugin_tool_map = {}

    rows: list[tuple[str, bool, int, int, str, str]] = []
    for entry in sorted(pdir.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        manifest = entry / "plugin.yml"
        if not manifest.exists():
            continue
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                m = _yaml.safe_load(f) or {}
        except Exception:
            m = {}
        name = m.get("name") or entry.name
        enabled = bool(m.get("enabled", True))
        version = str(m.get("version") or "").strip()
        author = str(m.get("author") or "").strip()
        sub = plugin_tool_map.get(name) or {}
        if isinstance(sub, dict):
            total = len(sub)
            on_count = sum(1 for v in sub.values() if bool(v))
        else:
            total = 0
            on_count = 0
        rows.append((name, enabled, on_count, total, version, author))

    if not rows:
        return

    try:
        from src.plugins.loader import get_plugin_issues, get_plugin_log_path
        issues_map = get_plugin_issues()
        log_path = get_plugin_log_path()
    except Exception:
        issues_map = {}
        log_path = None

    print(f"  {C.DIM}Plugins{C.RST}")
    for name, enabled, on_count, total, version, author in rows:
        if enabled:
            dot = f"{C.B_GREEN}\u25cf{C.RST}"
            label = f"{C.B_WHITE}{name}{C.RST}"
        else:
            dot = f"{C.DIM}\u25cb{C.RST}"
            label = f"{C.DIM}{name}{C.RST}"
        meta_bits = []
        if version:
            meta_bits.append(f"v{version}")
        if author:
            meta_bits.append(f"by {author}")
        if total > 0:
            meta_bits.append(f"{on_count}/{total} tools")
        else:
            meta_bits.append("no tools")
        meta = f"  {C.DIM}({' \u2022 '.join(meta_bits)}){C.RST}"

        # warning / error tally with the first message inline. when there
        # are more than one of either we tack on a "check plugins.log"
        # hint pointing at the daily plugin log file.
        issues = issues_map.get(name, []) or []
        warns = [it for it in issues if it.get("level") == "warning"]
        errs = [it for it in issues if it.get("level") == "error"]
        status_bits = []
        if errs:
            n = len(errs)
            first = str(errs[0].get("message", "")).strip()
            text = f"{n} error{'s' if n != 1 else ''}"
            if first:
                text += f" ({first})"
            status_bits.append(f"{C.B_RED}{text}{C.RST}")
        if warns:
            n = len(warns)
            first = str(warns[0].get("message", "")).strip()
            text = f"{n} warning{'s' if n != 1 else ''}"
            if first:
                text += f" ({first})"
            status_bits.append(f"{C.B_YELLOW}{text}{C.RST}")
        status = f"  {' '.join(status_bits)}" if status_bits else ""
        if status and (len(warns) + len(errs)) > 1:
            log_name = log_path.name if log_path else "plugins.log"
            status += f" {C.DIM}check {log_name}{C.RST}"

        print(f"  {dot} {label}{meta}{status}")
    print()


def print_plugins_info(plugins_dir: str = "plugins"):
    """Backwards compat: prints the plugins block as a standalone section
    with its own divider. Prefer print_startup_info now since it inlines."""
    _print_plugins_block(plugins_dir)
    print(f"  {C.DIM}{'\u2500' * _W}{C.RST}")
    print()


def _kv(key, value, color=C.B_WHITE):
    print(f"  {C.DIM}{key:<12}{C.RST} {color}{value}{C.RST}")
