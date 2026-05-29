"""PlayerTracker, composed from mixins."""

import threading
import time

from .capture import CaptureMixin
from .config import DEFAULT_CFG
from .config_io import ConfigIOMixin
from .debug import DebugMixin
from .detection import DetectionMixin
from .lifecycle import LifecycleMixin
from .loop import LoopMixin
from .model import ModelMixin
from .osc_out import OscMixin


class PlayerTracker(
    ConfigIOMixin,
    ModelMixin,
    CaptureMixin,
    DetectionMixin,
    DebugMixin,
    OscMixin,
    LifecycleMixin,
    LoopMixin,
):
    """Detects and follows players in VRChat using screen capture, YOLO, and OSC."""

    def __init__(self, config, osc=None):
        self.config = config
        self.osc = osc
        self.model = None
        self._active = False
        self._thread = None
        self._camera = None
        self._use_half = False
        self._first_frame = True
        self._preload_ready = threading.Event()
        self._vision_debug = False  # flipped True when vision debug server is running
        self._next_cache_cleanup = 0.0
        self._next_tracker_reset = 0.0

        # tracking state
        self._locked_id = None
        self._lock_lost_time = None
        self._current_target_area = 0.0
        self._smoothed_look_h = 0.0
        self._smoothed_look_v = 0.0
        self._smoothed_forward = 0.0
        self._sprinting = False

        self._cfg = dict(DEFAULT_CFG)
        self._load_config()

        # fps metrics
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()
        now = time.perf_counter()
        self._next_cache_cleanup = now + float(self._cfg.get("cache_cleanup_interval", 300.0))
        self._next_tracker_reset = now + float(self._cfg.get("tracker_reset_interval", 1800.0))
