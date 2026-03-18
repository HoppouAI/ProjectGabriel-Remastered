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


_W = 49


def print_startup_info(config):
    """Print configuration summary and component status."""
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
    print(f"  {C.DIM}{'\u2500' * _W}{C.RST}")
    print()


def _kv(key, value, color=C.B_WHITE):
    print(f"  {C.DIM}{key:<12}{C.RST} {color}{value}{C.RST}")
