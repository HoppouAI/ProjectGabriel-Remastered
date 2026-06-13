"""Cross platform single keypress reader for the plugin installer TUI.

Returns a string token per call. Special keys come back as named tokens
('UP', 'DOWN', 'LEFT', 'RIGHT', 'ENTER', 'ESC', 'BACKSPACE', 'TAB',
'SPACE', 'PGUP', 'PGDN', 'HOME', 'END', 'DELETE'). Plain printable keys
come back as the literal character.
"""
from __future__ import annotations

import sys

_NAMED = {
    "\r": "ENTER",
    "\n": "ENTER",
    "\x1b": "ESC",
    "\x7f": "BACKSPACE",
    "\x08": "BACKSPACE",
    "\t": "TAB",
    " ": "SPACE",
}


if sys.platform == "win32":
    import msvcrt

    _WIN_SPECIAL = {
        "H": "UP",
        "P": "DOWN",
        "K": "LEFT",
        "M": "RIGHT",
        "I": "PGUP",
        "Q": "PGDN",
        "G": "HOME",
        "O": "END",
        "S": "DELETE",
    }

    def get_key() -> str:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            return _WIN_SPECIAL.get(ch2, "?")
        return _NAMED.get(ch, ch)

else:  # posix fallback, mostly here so devs on linux can poke at this too
    import select
    import termios
    import tty

    def _read_one(fd: int, timeout: float = 0.0) -> str:
        if timeout > 0:
            r, _, _ = select.select([fd], [], [], timeout)
            if not r:
                return ""
        return sys.stdin.read(1)

    def get_key() -> str:  # type: ignore[no-redef]
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # could be ESC alone or the start of a CSI sequence
                ch2 = _read_one(fd, 0.05)
                if ch2 == "":
                    return "ESC"
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return {
                        "A": "UP",
                        "B": "DOWN",
                        "C": "RIGHT",
                        "D": "LEFT",
                        "H": "HOME",
                        "F": "END",
                        "5": "PGUP",
                        "6": "PGDN",
                        "3": "DELETE",
                    }.get(ch3, "?")
                return "ESC"
            return _NAMED.get(ch, ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
