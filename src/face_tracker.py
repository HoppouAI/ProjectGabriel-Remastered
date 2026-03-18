import json
import logging
import random
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = "models/yolov8"
FACE_MODEL_NAME = "yolov8n-face.pt"
FACE_MODEL_URL = "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov8n-face.pt"

FRAME_W = 640
FRAME_H = 360
TARGET_FPS = 15  # Lower FPS than person tracker since face tracking is less critical

DEFAULT_CFG = {
    "confidence_threshold": 0.35,
    "smoothing_alpha": 0.35,
    "max_turn_rate": 0.12,
    "deadzone": 0.15,
    "turn_gain": 2.5,
    "min_output": 0.45,
    "lock_timeout": 3.0,
    "idle_switch_min": 5.0,
    "idle_switch_max": 10.0,
}


class FaceTracker:
    """Detects faces on screen and smoothly looks at the closest one via OSC.

    Two modes:
      - Speaking mode: lock onto the closest/largest face and track it
      - Idle mode: every 5-10s randomly pick a visible face to glance at
    """

    def __init__(self, config, osc=None):
        self.config = config
        self.osc = osc
        self.model = None
        self._active = False
        self._thread = None
        self._camera = None
        self._use_half = False
        self._preload_ready = threading.Event()

        # External state reference - set by main.py / gemini_live
        self._speaking_ref = None  # callable that returns True when AI is speaking
        self._player_tracker_ref = None  # reference to PlayerTracker to check if following
        self._wanderer_ref = None  # reference to Wanderer to check if wandering

        # Tracking state
        self._locked_id = None
        self._lock_lost_time = None
        self._smoothed_look_h = 0.0
        self._smoothed_look_v = 0.0

        # Idle behaviour
        self._idle_next_switch = 0.0
        self._idle_target_id = None

        # Config
        self._cfg = dict(DEFAULT_CFG)
        self._load_config()

        # FPS metrics
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()

    @property
    def active(self):
        return self._active

    def set_speaking_callback(self, callback):
        """Set a callable that returns True when the AI is currently speaking."""
        self._speaking_ref = callback

    def set_player_tracker(self, tracker):
        """Set reference to PlayerTracker so face tracker yields when following."""
        self._player_tracker_ref = tracker

    def set_wanderer(self, wanderer):
        """Set reference to Wanderer so face tracker yields while wandering."""
        self._wanderer_ref = wanderer

    def _is_speaking(self):
        if self._speaking_ref is not None:
            return self._speaking_ref()
        return False

    def _player_tracker_active(self):
        if self._player_tracker_ref is not None:
            return self._player_tracker_ref.active
        return False

    def _wanderer_active(self):
        if self._wanderer_ref is not None:
            return self._wanderer_ref._active and not self._wanderer_ref._paused
        return False

    def _load_config(self):
        config_path = Path(MODEL_DIR) / "face_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    self._cfg.update(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load face tracker config: {e}")

    def _save_config(self):
        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "face_config.json", "w") as f:
            json.dump(self._cfg, f, indent=2)

    def preload(self):
        """Pre-load the face YOLO model in a background thread."""
        def _do_preload():
            try:
                logger.info("Face tracker: background preload starting...")
                self._ensure_model()
                logger.info("Face tracker: model ready")
            except Exception as e:
                logger.error(f"Face tracker preload failed: {e}")
            finally:
                self._preload_ready.set()

        t = threading.Thread(target=_do_preload, daemon=True, name="face-preload")
        t.start()

    def _ensure_model(self):
        if self.model is not None:
            return

        import torch
        from ultralytics import YOLO
        import numpy as np

        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / FACE_MODEL_NAME

        if not model_path.exists():
            logger.info(f"Downloading {FACE_MODEL_NAME} to {model_dir}...")
            try:
                import requests
                resp = requests.get(FACE_MODEL_URL, allow_redirects=True, timeout=60)
                resp.raise_for_status()
                with open(model_path, "wb") as f:
                    f.write(resp.content)
            except ImportError:
                import urllib.request
                urllib.request.urlretrieve(FACE_MODEL_URL, str(model_path))
            logger.info(f"Downloaded {FACE_MODEL_NAME} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
            # Re-save with current ultralytics to fix old format references
            _tmp = YOLO(str(model_path))
            _tmp.save(str(model_path))
            del _tmp

        self.model = YOLO(str(model_path))

        if torch.cuda.is_available():
            device = "cuda"
            self.model.to(device)
            self._use_half = True
        else:
            device = "cpu"
            self._use_half = False
            self.model.to(device)

        logger.info(f"{FACE_MODEL_NAME} loaded on {device} (FP16={self._use_half})")

        # Warmup
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.predict(
                dummy, conf=0.5, max_det=5, verbose=False, half=self._use_half,
            )
        logger.info("Face tracker warmup done")

    def start(self):
        """Start the face tracking loop."""
        if self._active and self._thread and self._thread.is_alive():
            return

        self._active = True
        self._reset_state()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="face-tracker"
        )
        self._thread.start()

    def stop(self):
        """Stop the face tracking loop."""
        if not self._active:
            return

        self._active = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _reset_state(self):
        self._locked_id = None
        self._lock_lost_time = None
        self._smoothed_look_h = 0.0
        self._smoothed_look_v = 0.0
        self._idle_next_switch = 0.0
        self._idle_target_id = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()

    def _init_screen_capture(self):
        """Returns a callable that grabs BGR frames. Uses mss to avoid conflicting with
        the person tracker's bettercam instance."""
        import numpy as np

        try:
            import mss
            sct = mss.mss()
            monitor_cfg = getattr(self.config, "vision_monitor", 1)
            if monitor_cfg >= len(sct.monitors):
                monitor_cfg = 1
            monitor = sct.monitors[monitor_cfg]
            logger.info(
                f"Face tracker: mss initialized (monitor {monitor_cfg}: "
                f"{monitor['width']}x{monitor['height']})"
            )

            def _grab_mss():
                return np.array(sct.grab(monitor))[:, :, :3]
            return _grab_mss
        except Exception as e:
            logger.error(f"Face tracker screen capture init failed: {e}")
            return None

    def _run_loop(self):
        import cv2
        import numpy as np

        capture_fn = self._init_screen_capture()
        if capture_fn is None:
            logger.error("Face tracker: no screen capture available, stopping")
            self._active = False
            return

        if not self._preload_ready.is_set():
            logger.info("Face tracker: waiting for model preload...")
            self._preload_ready.wait(timeout=60)

        import torch
        self._ensure_model()

        logger.info(f"Face tracker started at {TARGET_FPS} FPS")
        frame_interval = 1.0 / TARGET_FPS

        try:
            while self._active:
                t0 = time.perf_counter()

                # Pause detection when player tracker is following or wanderer is active
                if self._player_tracker_active() or self._wanderer_active():
                    self._zero_osc()
                    time.sleep(0.5)
                    continue

                frame = capture_fn()
                if frame is None:
                    time.sleep(0.001)
                    continue

                resized = cv2.resize(frame, (FRAME_W, FRAME_H))

                # Face detection (no tracking IDs - use predict instead of track)
                with torch.no_grad():
                    results = self.model.predict(
                        resized,
                        conf=self._cfg["confidence_threshold"],
                        max_det=10,
                        verbose=False,
                        half=self._use_half,
                    )

                detections = self._parse_faces(results)
                self._update_tracking(detections)
                self._send_osc()

                # FPS
                self._frame_count += 1
                elapsed = time.perf_counter() - self._fps_timer
                if elapsed >= 30.0:
                    self._fps = self._frame_count / elapsed
                    speaking = self._is_speaking()
                    mode = "speaking" if speaking else "idle"
                    logger.info(
                        f"Face tracker: {self._fps:.1f} FPS | faces={len(detections)} | "
                        f"mode={mode} | look_h={self._smoothed_look_h:.3f}"
                    )
                    self._frame_count = 0
                    self._fps_timer = time.perf_counter()

                dt = time.perf_counter() - t0
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)

        except Exception as e:
            logger.error(f"Face tracker loop error: {e}", exc_info=True)
        finally:
            self._zero_osc()
            if self._camera is not None:
                try:
                    del self._camera
                except Exception:
                    pass
                self._camera = None
            self._active = False
            logger.info(f"Face tracker stopped (last avg {self._fps:.1f} FPS)")

    def _parse_faces(self, results):
        """Extract face detections from YOLO results."""
        detections = []
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            area = (w * h) / (FRAME_W * FRAME_H)

            # Normalised offset from frame centre
            norm_dx = (cx - FRAME_W / 2) / (FRAME_W / 2)
            norm_dy = (cy - FRAME_H / 2) / (FRAME_H / 2)
            center_dist = (norm_dx**2 + norm_dy**2) ** 0.5

            detections.append({
                "idx": i,
                "cx": cx,
                "cy": cy,
                "norm_dx": norm_dx,
                "norm_dy": norm_dy,
                "area": area,
                "center_dist": center_dist,
                "conf": conf,
            })

        return detections

    def _score(self, det):
        """Score a face detection - lower is better. Prefer centred + large (close) faces."""
        return det["center_dist"] - det["area"] * 3.0

    def _update_tracking(self, detections):
        """Select target face and compute smoothed look values."""
        cfg = self._cfg
        alpha = cfg["smoothing_alpha"]
        now = time.time()
        speaking = self._is_speaking()

        if not detections:
            # No faces - decay toward centre
            self._smoothed_look_h *= 1 - alpha
            self._smoothed_look_v *= 1 - alpha
            self._locked_id = None
            self._idle_target_id = None
            return

        if speaking:
            # Speaking mode: lock onto the best (closest/most centred) face
            scored = sorted(detections, key=self._score)
            target = scored[0]
        else:
            # Idle mode: pick a random face every 5-10 seconds
            if now >= self._idle_next_switch or self._idle_target_id is None:
                target = random.choice(detections)
                self._idle_target_id = target["idx"]
                interval = random.uniform(cfg["idle_switch_min"], cfg["idle_switch_max"])
                self._idle_next_switch = now + interval
            else:
                # Try to find the previously selected face by index proximity
                # Since we don't have persistent IDs, pick the closest detection to where the
                # previous target was (by spatial proximity to current look direction)
                target = min(detections, key=lambda d: abs(d["norm_dx"] - self._smoothed_look_h))

        # Compute raw look values
        gain = cfg["turn_gain"]
        raw_look_h = max(-1.0, min(1.0, target["norm_dx"] * gain))
        raw_look_v = -target["norm_dy"] * 0.4  # Gentle vertical correction

        # EMA smoothing
        new_look_h = self._smoothed_look_h * (1 - alpha) + raw_look_h * alpha
        new_look_v = self._smoothed_look_v * (1 - alpha) + raw_look_v * alpha

        # Rate limiter
        max_rate = cfg["max_turn_rate"]
        delta_h = new_look_h - self._smoothed_look_h
        if abs(delta_h) > max_rate:
            new_look_h = self._smoothed_look_h + max_rate * (1 if delta_h > 0 else -1)

        self._smoothed_look_h = new_look_h
        self._smoothed_look_v = new_look_v

    def _send_osc(self):
        """Send smoothed look values to VRChat. Yields when player tracker or wanderer is active."""
        if not self.osc:
            return

        # Don't fight the player tracker or wanderer for LookHorizontal
        if self._player_tracker_active() or self._wanderer_active():
            return

        client = self.osc.client
        cfg = self._cfg
        dz = cfg["deadzone"]
        min_out = cfg["min_output"]

        look_h = max(-1.0, min(1.0, self._smoothed_look_h))
        look_v = max(-1.0, min(1.0, self._smoothed_look_v))

        # Apply deadzone
        if abs(look_h) < dz:
            look_h = 0.0
        elif abs(look_h) < min_out:
            # Boost to minimum so VRChat actually registers the turn
            look_h = min_out if look_h > 0 else -min_out

        if abs(look_v) < dz:
            look_v = 0.0

        client.send_message("/input/LookHorizontal", float(look_h))
        client.send_message("/input/LookVertical", float(look_v))

    def _zero_osc(self):
        """Reset look axes to zero."""
        if not self.osc:
            return
        self.osc.client.send_message("/input/LookHorizontal", 0.0)
        self.osc.client.send_message("/input/LookVertical", 0.0)
