"""The PlayerTracker class -- composes the mixins, owns the main loop."""

import json
import logging
import threading
import time
from pathlib import Path

from ._constants import DEFAULT_CFG, FRAME_H, FRAME_W, MODEL_DIR, TARGET_FPS
from .capture import CaptureMixin
from .debug import DebugMixin
from .model import ModelMixin
from .osc import OscMixin
from .tracking import TrackingMixin

logger = logging.getLogger(__name__)


class PlayerTracker(TrackingMixin, ModelMixin, CaptureMixin, OscMixin, DebugMixin):
    """Detects and follows players in VRChat using screen capture, YOLO + boxmot, and OSC.

    Uses boxmot DeepOCSORT for tracking -- IDs stay stable across occlusion
    and frame exits via OSNet ReID embeddings. This makes the follow lock
    actually stick to one person instead of grabbing whoever walks into the
    centre of the screen.
    """

    def __init__(self, config, osc=None):
        self.config = config
        self.osc = osc
        self.model = None
        self._bot_tracker = None
        self._torch_device = "cpu"
        self._active = False
        self._thread = None
        self._camera = None
        self._use_half = False
        self._preload_ready = threading.Event()
        self._vision_debug = False
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

        # config
        self._cfg = dict(DEFAULT_CFG)
        self._load_config()

        # fps metrics
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()
        now = time.perf_counter()
        self._next_cache_cleanup = now + float(self._cfg.get("cache_cleanup_interval", 300.0))
        self._next_tracker_reset = now + float(self._cfg.get("tracker_reset_interval", 1800.0))

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        if not value and self._active:
            self.stopfollow()
        self._active = value

    # ── Config I/O ────────────────────────────────────────────────────────

    def _load_config(self):
        config_path = Path(MODEL_DIR) / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    self._cfg.update(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load tracker config: {e}")

    def _save_config(self):
        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump(self._cfg, f, indent=2)

    # ── Public API (called by Gemini tools) ───────────────────────────────

    def startfollow(self, mode="auto"):
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
        """Set desired follow distance as target bounding-box area fraction (0.01..0.5)."""
        value = max(0.01, min(0.5, float(value)))
        self._cfg["target_area"] = value
        self._save_config()
        return {"result": "ok", "message": f"follow distance set to {value:.3f}"}

    def cleartarget(self):
        """Force-drop the current locked target so the tracker re-acquires."""
        old = self._locked_id
        self._locked_id = None
        self._lock_lost_time = None
        # also wipe the boxmot tracker so old reid embeddings dont
        # immediately re-attach
        self._bot_tracker = None
        return {"result": "ok", "message": f"cleared locked target (was id={old})"}

    # ── Main loop ─────────────────────────────────────────────────────────

    def _run_loop(self):
        import cv2

        # init screen capture FIRST (before CUDA loads)
        capture_fn = self._init_screen_capture()
        if capture_fn is None:
            self._active = False
            return

        # now load model (triggers torch/CUDA import)
        if not self._preload_ready.is_set():
            logger.info("Waiting for model preload...")
            self._preload_ready.wait(timeout=60)
        import torch
        self._ensure_model()
        self._ensure_boxmot_tracker()

        logger.info(f"Player tracker started -- target {TARGET_FPS} FPS")
        frame_interval = 1.0 / TARGET_FPS

        try:
            while self._active:
                t0 = time.perf_counter()

                frame = capture_fn()
                if frame is None:
                    time.sleep(0.001)
                    continue

                resized = cv2.resize(frame, (FRAME_W, FRAME_H))

                # yolo detect
                with torch.no_grad():
                    results = self.model.predict(
                        resized,
                        conf=self._cfg["confidence_threshold"],
                        iou=self._cfg["iou_threshold"],
                        classes=[0],
                        max_det=self._cfg["max_detections"],
                        verbose=False,
                        half=self._use_half,
                    )

                # feed detections to boxmot for reid-aware tracking
                dets = self._parse_yolo_to_dets(results)

                if self._bot_tracker in (None, False):
                    self._ensure_boxmot_tracker()

                if self._bot_tracker and self._bot_tracker is not False:
                    try:
                        tracks = self._bot_tracker.update(dets, resized)
                    except Exception as e:
                        logger.warning(f"boxmot update failed: {e}, recreating tracker")
                        self._bot_tracker = None
                        tracks = None
                else:
                    # no boxmot available -- best-effort, dets without stable ids
                    import numpy as np
                    if dets.shape[0] > 0:
                        fake_ids = np.arange(dets.shape[0]).reshape(-1, 1).astype(np.float32)
                        det_ind = np.arange(dets.shape[0]).reshape(-1, 1).astype(np.float32)
                        tracks = np.concatenate(
                            [dets[:, :4], fake_ids, dets[:, 4:6], det_ind],
                            axis=1,
                        )
                    else:
                        tracks = None

                detections = self._parse_tracks(tracks)

                self._update_tracking(detections)
                self._send_osc()

                if self._vision_debug:
                    self._push_debug_frame(resized, detections)

                self._maybe_refresh_tracker_state(torch)

                # fps
                self._frame_count += 1
                elapsed = time.perf_counter() - self._fps_timer
                if elapsed >= 2.0:
                    self._fps = self._frame_count / elapsed
                    logger.info(
                        f"Tracker: {self._fps:.1f} FPS | "
                        f"target_id={self._locked_id} | "
                        f"area={self._current_target_area:.4f}"
                    )
                    self._frame_count = 0
                    self._fps_timer = time.perf_counter()

                # frame pacing
                dt = time.perf_counter() - t0
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)

        except Exception as e:
            logger.error(f"Tracker loop error: {e}", exc_info=True)
        finally:
            self._zero_osc()
            self._cleanup_inference_cache(torch if "torch" in locals() else None)
            self._close_screen_capture()
            self._bot_tracker = None
            self._active = False
            logger.info(f"Player tracker stopped (last avg {self._fps:.1f} FPS)")
