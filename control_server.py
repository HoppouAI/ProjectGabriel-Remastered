"""Backwards compat shim. Real implementation lives in src/web/.

Old code does `from control_server import app, shared_state, add_console_log`
and we want that to keep working without touching every caller.
"""
from src.web import add_console_log, app, run_control_server, shared_state

__all__ = ["app", "shared_state", "add_console_log", "run_control_server"]
