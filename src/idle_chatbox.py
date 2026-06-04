import asyncio
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

CHATBOX_CHAR_LIMIT = 144
# while mapping is active we want the timer to visibly tick, but we
# also dont want to hammer VRChat's chatbox endpoint. 10s is well above
# the 1.5s hard limit and matches the voxel-count cache window.
MAPPING_REFRESH_SECONDS = 10


class IdleChatbox:
    """Displays a customizable banner in VRChat chatbox when the AI is idle."""

    def __init__(self, osc, config):
        self._osc = osc
        self._config = config
        self._running = False
        self._task = None
        self._session_start = time.time()
        self._mapping_service = None
        self._mapping_started_at = None
        self._mapping_cache = None
        self._mapping_cache_at = 0.0

    def set_mapping_service(self, mapping_service):
        """Wire a MappingService so the banner can show live mapping stats."""
        self._mapping_service = mapping_service

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

    def _format_elapsed(self, seconds):
        seconds = int(max(0, seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _get_mapping_snapshot(self):
        """Pull mapping state with a 10s cache so we dont touch the
        graph lock more than necessary. Returns None when theres nothing
        special to show. Otherwise a dict with "kind":

            "navigate" -> {"waypoint": str}  (driving to a saved waypoint)
            "explore"  -> {"world", "voxels", "started_at"}  (auto mapping)

        We only surface the mapping banner when the explorer is actually
        exploring on its own (state["explore"]). Mapping merely being
        "running" (e.g. passive trail recording, or a one-off goto) does
        not count. Waypoint navigation takes priority over the explore
        banner."""
        ms = self._mapping_service
        if ms is None:
            return None
        now = time.time()
        if self._mapping_cache is not None and (now - self._mapping_cache_at) < MAPPING_REFRESH_SECONDS:
            return self._mapping_cache
        try:
            state = ms.get_state()
        except Exception:
            return self._mapping_cache  # serve stale on transient failure

        running = bool(state.get("running"))
        follow = state.get("follow") or {}
        label = str(follow.get("label") or "")
        # waypoint navigation: follow active with a "wp:<name>" label
        if running and follow.get("active") and label.startswith("wp:"):
            self._mapping_started_at = None
            snap = {"kind": "navigate", "waypoint": label[3:] or "waypoint"}
            self._mapping_cache = snap
            self._mapping_cache_at = now
            return snap

        # autonomous exploration banner only when the explorer is self-driving
        if running and bool(state.get("explore")):
            if self._mapping_started_at is None:
                self._mapping_started_at = now
            counts = state.get("counts") or {}
            snap = {
                "kind": "explore",
                "world": state.get("world_name") or state.get("world") or "unknown world",
                "voxels": int(counts.get("total", 0)),
                "started_at": self._mapping_started_at,
            }
            self._mapping_cache = snap
            self._mapping_cache_at = now
            return snap

        # nothing special to show
        self._mapping_started_at = None
        self._mapping_cache = None
        self._mapping_cache_at = now
        return None

    def _format_banner(self):
        cfg = self._config
        banner = cfg.get("vrchat", "idle_chatbox", "banner", default="")
        divider_char = cfg.get("vrchat", "idle_chatbox", "divider", default="\u2500")
        divider_length = cfg.get("vrchat", "idle_chatbox", "divider_length", default=14)
        lines = cfg.get("vrchat", "idle_chatbox", "lines", default=[])

        divider = str(divider_char) * int(divider_length)
        mapping = self._get_mapping_snapshot()

        parts = []
        if banner:
            parts.append(str(banner))
        parts.append(divider)

        if mapping is not None and mapping.get("kind") == "navigate":
            parts.append(f"Navigating to: {mapping['waypoint']}")
        elif mapping is not None and mapping.get("kind") == "explore":
            elapsed = self._format_elapsed(time.time() - mapping["started_at"])
            parts.append(f"Mapping: {mapping['world']}")
            parts.append(f"{mapping['voxels']} voxels, {elapsed}")
        else:
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
        idle_interval = self._config.get("vrchat", "idle_chatbox", "update_interval", default=30)
        try:
            while self._running:
                text = self._format_banner()
                self._osc.send_chatbox(text)
                # tick faster while actively mapping so the elapsed
                # counter feels live; fall back to the configured idle
                # cadence otherwise.
                interval = MAPPING_REFRESH_SECONDS if self._mapping_cache is not None else idle_interval
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

