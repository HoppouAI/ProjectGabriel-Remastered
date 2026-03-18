import asyncio
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

CHATBOX_CHAR_LIMIT = 144


class IdleChatbox:
    """Displays a customizable banner in VRChat chatbox when the AI is idle."""

    def __init__(self, osc, config):
        self._osc = osc
        self._config = config
        self._running = False
        self._task = None
        self._session_start = time.time()

    @property
    def enabled(self):
        return self._config.get("vrchat", "idle_chatbox", "enabled", default=False)

    def start(self):
        if not self.enabled or self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._update_loop())
        logger.debug("Idle chatbox started")

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.debug("Idle chatbox stopped")

    def _format_active_time(self):
        elapsed = int(time.time() - self._session_start)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def _format_clock(self):
        now = datetime.now()
        hour = now.hour % 12 or 12
        ampm = "AM" if now.hour < 12 else "PM"
        return f"{hour}:{now.minute:02d} {ampm}"

    def _format_banner(self):
        cfg = self._config
        banner = cfg.get("vrchat", "idle_chatbox", "banner", default="")
        divider_char = cfg.get("vrchat", "idle_chatbox", "divider", default="\u2500")
        divider_length = cfg.get("vrchat", "idle_chatbox", "divider_length", default=20)
        lines = cfg.get("vrchat", "idle_chatbox", "lines", default=[])

        divider = str(divider_char) * int(divider_length)

        parts = []
        if banner:
            parts.append(str(banner))
        parts.append(divider)

        for line in lines[:3]:
            if line:
                parts.append(str(line))

        parts.append(divider)
        parts.append(f"Active: {self._format_active_time()} | {self._format_clock()}")

        text = "\n".join(parts)
        if len(text) > CHATBOX_CHAR_LIMIT:
            text = text[:CHATBOX_CHAR_LIMIT]
        return text

    async def _update_loop(self):
        interval = self._config.get("vrchat", "idle_chatbox", "update_interval", default=30)
        try:
            while self._running:
                text = self._format_banner()
                self._osc.send_chatbox(text)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
