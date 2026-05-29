"""Web layer for the Gabriel control panel. Exports the FastAPI app + shared state."""
from .shared import shared_state, console_logs, add_console_log
from .app import app, run_control_server

__all__ = ["app", "shared_state", "console_logs", "add_console_log", "run_control_server"]
