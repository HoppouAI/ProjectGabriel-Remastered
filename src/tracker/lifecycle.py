"""Tracker lifecycle: start/stop follow, distance, reset state, active property."""

import logging
import threading
import time

logger = logging.getLogger("src.tracker")


class LifecycleMixin:
    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        if not value and self._active:
            self.stopfollow()
        self._active = value

    def startfollow(self, mode="auto"):
        """Start following a player visible on screen."""
        if self._active and self._thread and self._thread.is_alive():
            return {"result": "ok", "message": "already following"}

        self._active = True
        self._reset_state()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="player-tracker"
        )
        self._thread.start()
        return {"result": "ok", "message": f"started following (mode={mode})"}

    def stopfollow(self):
        if not self._active:
            return {"result": "ok", "message": "not following"}

        self._active = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        return {"result": "ok", "message": "stopped following"}

    def setfollowdistance(self, value):
        """Desired follow distance as target bounding-box area fraction (0.01-0.5)."""
        value = max(0.01, min(0.5, float(value)))
        self._cfg["target_area"] = value
        self._save_config()
        return {"result": "ok", "message": f"follow distance set to {value:.3f}"}

    def _reset_state(self):
        self._locked_id = None
        self._lock_lost_time = None
        self._current_target_area = 0.0
        self._smoothed_look_h = 0.0
        self._smoothed_look_v = 0.0
        self._smoothed_forward = 0.0
        self._sprinting = False
        self._first_frame = True
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()
