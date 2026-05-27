"""Autonomous wandering using avatar-mounted VRCRaycasts.

Replaces the old depth-model based wanderer. Reads engine-truth distance
readings published by VRCRaycast components on the avatar (see
unity_assets/AVATAR_SETUP.md) via VRChatOSC.raycast_state and steers
purely from those, no screen capture and no neural net.

Public API (kept stable for callers):
    Wanderer(config, osc=None)
    .active                  -> bool property
    .preload()               -> no-op now, kept so main.py doesnt break
    .start() / .stop()       -> dict result
    .pause() / .resume()
    .on_speech_activity()
    .on_ai_speaking()
    ._face_tracker_ref       -> assigned by main.py
    ._emotion_system_ref     -> assigned by main.py
"""

from __future__ import annotations

import threading

from .config import DEFAULT_CFG
from .decision import DecisionMixin
from .loop import LoopMixin
from .map_mode import MapModeMixin
from .osc_out import OscMixin
from .pause import PauseMixin
from .sensors import SensorsMixin


class Wanderer(
    SensorsMixin,
    OscMixin,
    PauseMixin,
    DecisionMixin,
    MapModeMixin,
    LoopMixin,
):
    """Raycast-driven VRChat wanderer."""

    def __init__(self, config, osc=None):
        self.config = config
        self.osc = osc
        self._active = False
        self._thread = None
        self._lock = threading.Lock()

        # external refs wired up by main.py
        self._face_tracker_ref = None
        self._emotion_system_ref = None
        self._mapping_service_ref = None

        # map-mode state
        self._map_state = "idle"        # idle | following | dwell
        self._map_visits: dict = {}     # serial -> last visit timestamp
        self._map_dwell_until: float = 0.0
        self._map_follow_start: float = 0.0
        self._map_target_cell = None
        self._map_last_pick_failed: float = 0.0

        # pause / resume state
        self._paused = False
        self._auto_paused = False
        self._resume_timer = None

        # navigation state
        self._smoothed_turn = 0.0
        self._smoothed_forward = 0.0
        self._last_straight_time = 0.0
        self._committed_turn_dir = 0.0
        self._committed_turn_until = 0.0
        self._stuck_frames = 0
        self._stuck_turn_dir = 1.0
        self._current_action = "idle"
        # DropFwd is only trusted as a ledge sensor after we have seen it
        # hit the ground at least once. otherwise a missing/misconfigured
        # DropFwd ray would make us think we are always on a cliff.
        self._dropfwd_ever_hit = False
        # rolling history of forward clearance for predictive steering
        self._clearance_history = []  # list of (timestamp, clearance)
        # recent wall hits, for escalation to u-turn when we keep bouncing
        self._recent_wall_hits = []  # list of timestamps

        # config dict, with optional yaml overrides under wanderer.*
        self._cfg = dict(DEFAULT_CFG)
        try:
            user_overrides = self.config.get("wanderer", default={}) or {}
            if isinstance(user_overrides, dict):
                for k, v in user_overrides.items():
                    if k in self._cfg and isinstance(v, (int, float, bool)):
                        self._cfg[k] = v
        except Exception:
            pass

        self._resume_delay = float(self._cfg["auto_resume_seconds"])

    @property
    def active(self):
        return self._active

    def preload(self):
        # kept so main.py call site doesnt break, raycasts dont need preloading
        return
