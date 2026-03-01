"""
Player tracking and following system.
YOLOv8n + ByteTrack + dxcam screen capture → VRChat OSC movement.
"""

import json
import logging
import time
import threading
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = "models/yolov8"
MODEL_NAME = "yolov8n.pt"
FRAME_W = 640
FRAME_H = 360
TARGET_FPS = 30

DEFAULT_CFG = {
    "confidence_threshold": 0.40,
    "iou_threshold": 0.45,
    "target_area": 0.08,
    "deadzone": 0.04,
    "smoothing_alpha": 0.45,
    "turn_gain": 2.5,
    "center_distance_weight": 1.0,
    "area_weight": 0.5,
    "lock_timeout": 2.0,
    "reacquire_threshold": 0.3,
    "max_detections": 10,
    "forward_scale_min": 0.5,
    "forward_scale_max": 1.0,
    "strafe_threshold": 0.25,
    "strafe_scale": 0.6,
}


class PlayerTracker:
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

        # Tracking state
        self._locked_id = None
        self._lock_lost_time = None
        self._current_target_area = 0.0
        self._smoothed_look_h = 0.0
        self._smoothed_look_v = 0.0
        self._smoothed_forward = 0.0

        # Config
        self._cfg = dict(DEFAULT_CFG)
        self._load_config()

        # FPS metrics
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()

    # ── Properties ────────────────────────────────────────────────────────

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

    # ── Model Loading ─────────────────────────────────────────────────────

    def _ensure_model(self):
        if self.model is not None:
            return

        import torch
        from ultralytics import YOLO
        import numpy as np

        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / MODEL_NAME

        if not model_path.exists():
            logger.info(f"Downloading {MODEL_NAME} to {model_dir}...")
            temp_model = YOLO(MODEL_NAME)
            default_path = Path(MODEL_NAME)
            if default_path.exists():
                shutil.move(str(default_path), str(model_path))
            self.model = temp_model
        else:
            self.model = YOLO(str(model_path))

        # CUDA setup with detailed diagnostics
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA available: {gpu_name} ({vram:.1f} GB)")
            self.model.to(device)
            self._use_half = True  # FP16 handled by half= param in track()
        else:
            device = "cpu"
            self._use_half = False
            logger.warning(
                "CUDA not available — running on CPU (expect <10 FPS). "
                "Install PyTorch CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )
            self.model.to(device)

        logger.info(f"{MODEL_NAME} loaded on {device} (FP16={self._use_half})")

        # Warmup inference (JIT compile kernels, allocate buffers)
        logger.info("Running warmup inference...")
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.track(
                dummy, persist=False, conf=0.5, classes=[0],
                max_det=5, verbose=False, half=self._use_half,
            )
        logger.info("Warmup done")

    # ── Public API (called by Gemini tools) ───────────────────────────────

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
        """Stop following."""
        if not self._active:
            return {"result": "ok", "message": "not following"}

        self._active = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        return {"result": "ok", "message": "stopped following"}

    def setfollowdistance(self, value):
        """Set desired follow distance as target bounding-box area fraction (0.01–0.5)."""
        value = max(0.01, min(0.5, float(value)))
        self._cfg["target_area"] = value
        self._save_config()
        return {"result": "ok", "message": f"follow distance set to {value:.3f}"}

    # ── Internal State ────────────────────────────────────────────────────

    def _reset_state(self):
        self._locked_id = None
        self._lock_lost_time = None
        self._current_target_area = 0.0
        self._smoothed_look_h = 0.0
        self._smoothed_look_v = 0.0
        self._smoothed_forward = 0.0
        self._first_frame = True
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()

    # ── Screen Capture Init ─────────────────────────────────────────────────

    def _init_screen_capture(self):
        """Try dxcam, fall back to mss. Returns a callable that grabs BGR frames."""
        import numpy as np

        # Try dxcam (fastest, GPU-accelerated)
        try:
            import dxcam

            if self._camera is not None:
                try:
                    del self._camera
                except Exception:
                    pass
                self._camera = None

            # Try default (primary monitor), then explicit indices
            for args in [
                {},
                {"output_idx": 0},
                {"device_idx": 0, "output_idx": 0},
            ]:
                try:
                    self._camera = dxcam.create(output_color="BGR", **args)
                    logger.info(f"dxcam initialized ({args or 'default'})")

                    def _grab_dxcam():
                        return self._camera.grab()

                    return _grab_dxcam
                except Exception:
                    self._camera = None
                    continue

            logger.warning("dxcam: all init attempts failed, falling back to mss")
        except ImportError:
            logger.warning("dxcam not installed, falling back to mss")

        # Fallback: mss (works everywhere, slightly slower)
        try:
            import mss

            sct = mss.mss()
            monitor_cfg = getattr(self.config, "vision_monitor", 1)
            if monitor_cfg >= len(sct.monitors):
                monitor_cfg = 1
            monitor = sct.monitors[monitor_cfg]
            logger.info(
                f"mss initialized (monitor {monitor_cfg}: "
                f"{monitor['width']}x{monitor['height']})"
            )

            def _grab_mss():
                return np.array(sct.grab(monitor))[:, :, :3]

            return _grab_mss
        except Exception as e:
            logger.error(f"Screen capture init failed entirely: {e}")
            return None

    # ── Main Tracking Loop (runs in thread) ───────────────────────────────

    def _run_loop(self):
        import cv2
        import torch
        import numpy as np

        self._ensure_model()

        # Init screen capture — try dxcam first, fall back to mss
        capture_fn = self._init_screen_capture()
        if capture_fn is None:
            self._active = False
            return

        logger.info(f"Player tracker started — target {TARGET_FPS} FPS")
        frame_interval = 1.0 / TARGET_FPS

        try:
            while self._active:
                t0 = time.perf_counter()

                # ── Capture ──
                frame = capture_fn()
                if frame is None:
                    time.sleep(0.001)
                    continue

                # Resize to 640×360
                resized = cv2.resize(frame, (FRAME_W, FRAME_H))

                # ── Detect + Track ──
                with torch.no_grad():
                    results = self.model.track(
                        resized,
                        persist=not self._first_frame,
                        tracker="bytetrack.yaml",
                        conf=self._cfg["confidence_threshold"],
                        iou=self._cfg["iou_threshold"],
                        classes=[0],
                        max_det=self._cfg["max_detections"],
                        verbose=False,
                        half=self._use_half,
                    )
                self._first_frame = False

                # ── Process ──
                detections = self._parse_results(results)
                self._update_tracking(detections)
                self._send_osc()

                # ── FPS ──
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

                # ── Frame pacing ──
                dt = time.perf_counter() - t0
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)

        except Exception as e:
            logger.error(f"Tracker loop error: {e}", exc_info=True)
        finally:
            self._zero_osc()
            if self._camera is not None:
                try:
                    del self._camera
                except Exception:
                    pass
                self._camera = None
            self._active = False
            logger.info(f"Player tracker stopped (last avg {self._fps:.1f} FPS)")

    # ── Detection Parsing ─────────────────────────────────────────────────

    def _parse_results(self, results):
        """Extract person detections with tracking IDs from YOLO+ByteTrack."""
        detections = []
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            if int(boxes.cls[i]) != 0:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            track_id = int(boxes.id[i]) if boxes.id is not None else None

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            area = (w * h) / (FRAME_W * FRAME_H)

            # Normalised distance from frame centre (0 = dead centre, ~1.41 = corner)
            norm_dx = (cx - FRAME_W / 2) / (FRAME_W / 2)
            norm_dy = (cy - FRAME_H / 2) / (FRAME_H / 2)
            center_dist = (norm_dx**2 + norm_dy**2) ** 0.5

            detections.append(
                {
                    "id": track_id,
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "center_dist": center_dist,
                    "conf": conf,
                }
            )

        return detections

    # ── Target Selection & Scoring ────────────────────────────────────────

    def _score(self, det):
        """Score a detection — lower is better (prefer centred + large)."""
        return (
            self._cfg["center_distance_weight"] * det["center_dist"]
            - self._cfg["area_weight"] * det["area"]
        )

    def _update_tracking(self, detections):
        """Select / maintain target and compute smoothed movement values."""
        cfg = self._cfg
        alpha = cfg["smoothing_alpha"]
        now = time.time()

        # ── No detections ──
        if not detections:
            if self._locked_id is not None:
                if self._lock_lost_time is None:
                    self._lock_lost_time = now
                elif now - self._lock_lost_time > cfg["lock_timeout"]:
                    self._locked_id = None
                    self._lock_lost_time = None

            self._current_target_area = 0.0
            # Decay smoothed values toward zero
            self._smoothed_look_h *= 1 - alpha
            self._smoothed_look_v *= 1 - alpha
            self._smoothed_forward *= 1 - alpha
            return

        # ── Resolve target ──
        trackable = [d for d in detections if d["id"] is not None]
        target = None

        # Try to re-find locked target
        if self._locked_id is not None:
            for d in trackable:
                if d["id"] == self._locked_id:
                    target = d
                    self._lock_lost_time = None
                    break

            if target is None:
                # Locked ID not in this frame
                if self._lock_lost_time is None:
                    self._lock_lost_time = now
                elif now - self._lock_lost_time > cfg["lock_timeout"]:
                    self._locked_id = None
                    self._lock_lost_time = None

        # Score trackable detections and potentially (re)acquire
        if trackable:
            scored = sorted(trackable, key=self._score)
            best = scored[0]

            if target is None:
                # No current lock — acquire best
                target = best
                self._locked_id = best["id"]
                self._lock_lost_time = None
                logger.debug(f"Locked target {self._locked_id}")
            elif best["id"] != self._locked_id:
                # Check if a new target is *significantly* better
                if self._score(target) - self._score(best) > cfg["reacquire_threshold"]:
                    target = best
                    self._locked_id = best["id"]
                    self._lock_lost_time = None
                    logger.debug(f"Switched to better target {self._locked_id}")

        # Fallback: use closest-to-centre un-tracked detection
        if target is None and detections:
            target = min(detections, key=lambda d: d["center_dist"])

        if target is None:
            self._current_target_area = 0.0
            self._smoothed_look_h *= 1 - alpha
            self._smoothed_look_v *= 1 - alpha
            self._smoothed_forward *= 1 - alpha
            return

        # ── Compute raw movement values ──
        self._current_target_area = target["area"]
        deadzone = cfg["deadzone"]
        target_area = cfg["target_area"]

        # Normalised screen-space offsets (−1 … +1)
        dx = (target["cx"] - FRAME_W / 2) / (FRAME_W / 2)
        dy = (target["cy"] - FRAME_H / 2) / (FRAME_H / 2)
        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))

        # Apply deadzone
        if abs(dx) < deadzone:
            dx = 0.0
        if abs(dy) < deadzone:
            dy = 0.0

        # Apply turn gain — makes turning more aggressive so target stays on-screen
        gain = cfg["turn_gain"]
        raw_look_h = max(-1.0, min(1.0, dx * gain))
        raw_look_v = -dy * 0.4

        # Forward control based on bounding-box area vs target area
        if target["area"] < target_area:
            deficit = (target_area - target["area"]) / target_area
            raw_forward = cfg["forward_scale_min"] + deficit * (
                cfg["forward_scale_max"] - cfg["forward_scale_min"]
            )
            raw_forward = min(raw_forward, cfg["forward_scale_max"])
        else:
            raw_forward = 0.0

        # ── EMA smoothing ──
        self._smoothed_look_h = (
            self._smoothed_look_h * (1 - alpha) + raw_look_h * alpha
        )
        self._smoothed_look_v = (
            self._smoothed_look_v * (1 - alpha) + raw_look_v * alpha
        )
        self._smoothed_forward = (
            self._smoothed_forward * (1 - alpha) + raw_forward * alpha
        )

    # ── OSC Output ────────────────────────────────────────────────────────

    def _send_osc(self):
        """Send smoothed axis values to VRChat via OSC. All axes are float −1…+1."""
        if not self.osc:
            return

        client = self.osc.client
        cfg = self._cfg
        dz = cfg["deadzone"]

        # Clamp final outputs to [-1, 1]
        look_h = max(-1.0, min(1.0, self._smoothed_look_h))
        forward = max(-1.0, min(1.0, self._smoothed_forward))

        # ── Turn axis (proportional, smooth in Desktop) ──
        if abs(look_h) < dz:
            look_h = 0.0
        client.send_message("/input/LookHorizontal", float(look_h))

        # ── Forward / backward axis ──
        client.send_message("/input/Vertical", float(forward))

        # ── Strafe axis (only when heavily off-centre) ──
        if abs(look_h) > cfg["strafe_threshold"]:
            strafe = max(-1.0, min(1.0, look_h * cfg["strafe_scale"]))
            client.send_message("/input/Horizontal", float(strafe))
        else:
            client.send_message("/input/Horizontal", 0.0)

    def _zero_osc(self):
        """Reset all axes to zero — called on shutdown."""
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/LookHorizontal", 0.0)
        client.send_message("/input/Vertical", 0.0)
        client.send_message("/input/Horizontal", 0.0)
