import logging
import psutil
from datetime import datetime
from google.genai import types

logger = logging.getLogger(__name__)

_start_time = datetime.now()


class DiscordSystemTool:
    """System information tools for the Discord bot."""

    def __init__(self, handler):
        self.handler = handler

    def declarations(self):
        return [
            types.FunctionDeclaration(
                name="getSystemInfo",
                description="Get system info including uptime, memory usage, and bot status.\n**Invocation Condition:** Call when asked about bot status, uptime, or system info.",
                parameters={"type": "OBJECT", "properties": {}, "required": []},
            ),
        ]

    async def handle(self, name, args):
        if name != "getSystemInfo":
            return None

        uptime = datetime.now() - _start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        mem = psutil.virtual_memory()
        return {
            "result": "ok",
            "uptime": f"{hours}h {minutes}m {seconds}s",
            "memory_used_mb": round(mem.used / 1024 / 1024),
            "memory_percent": mem.percent,
            "cpu_percent": psutil.cpu_percent(),
        }
