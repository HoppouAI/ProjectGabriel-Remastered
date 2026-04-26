"""
Player tracking and following system.
YOLOv8n + ByteTrack + screen capture → VRChat OSC movement.

Capture priority:
  1. bettercam (DXGI Desktop Duplication — fastest, GPU-accelerated)
     ⚠ Must be initialised BEFORE torch/CUDA loads, otherwise DXGI
       fails with DXGI_ERROR_UNSUPPORTED on hybrid-GPU systems because
       CUDA forces the process onto the discrete GPU.
  2. mss (GDI BitBlt — works everywhere, ~10 FPS)
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
    "target_area": 0.04,
    "sprint_area": 0.015,
    "deadzone": 0.07,
    "smoothing_alpha": 0.40,
    "turn_gain": 1.8,
    "max_turn_rate": 0.12,
    "center_distance_weight": 1.0,
    "area_weight": 0.5,
    "lock_timeout": 5.0,
    "reacquire_threshold": 1.0,
    "max_detections": 10,
    "forward_scale_min": 0.5,
    "forward_scale_max": 0.7,
    "strafe_threshold": 0.25,
    "strafe_scale": 0.6,
    "too_close_area": 0.072,
    "backup_scale": 0.5,
    "cache_cleanup_interval": 300.0,
    "tracker_reset_interval": 1800.0,
}


# ── Early bettercam init (BEFORE torch/CUDA) ─────────────────────────────
# CUDA linking makes DXGI DuplicateOutput fail on some systems.
# By creating the camera object now, the DXGI context is established
# while the process is still on the iGPU.
_early_camera = None
_early_camera_backend = None

def _try_early_bettercam():
    """Attempt to create a bettercam camera before any CUDA import."""
    global _early_camera, _early_camera_backend
    try:
        import bettercam
        for args in [
            {},
            {"output_idx": 0},
            {"device_idx": 0, "output_idx": 0},
        ]:
            try:
                cam = bettercam.create(output_color="BGR", **args)
                time.sleep(0.15)  # let DXGI settle (known bettercam quirk)
                _early_camera = cam
                _early_camera_backend = "bettercam"
                logger.info(f"Early bettercam init OK ({args or 'default'})")
                return
            except Exception as e:
                logger.debug(f"Early bettercam attempt ({args}): {e}")
                continue
        logger.info("Early bettercam init failed on all attempts")
    except ImportError:
        logger.debug("bettercam not installed — skipping early init")

_try_early_bettercam()


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
        self._preload_ready = threading.Event()
        self._vision_debug = False  # set True when vision debug server is running
        self._next_cache_cleanup = 0.0
        self._next_tracker_reset = 0.0

        # Tracking state
        self._locked_id = None
        self._lock_lost_time = None
        self._current_target_area = 0.0
        self._smoothed_look_h = 0.0
        self._smoothed_look_v = 0.0
        self._smoothed_forward = 0.0
        self._sprinting = False

        # Config
        self._cfg = dict(DEFAULT_CFG)
        self._load_config()

        # FPS metrics
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()
        now = time.perf_counter()
        self._next_cache_cleanup = now + float(self._cfg.get("cache_cleanup_interval", 300.0))
        self._next_tracker_reset = now + float(self._cfg.get("tracker_reset_interval", 1800.0))

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

    # ── Background Preload ────────────────────────────────────────────────

    def preload(self):
        """Pre-load the YOLO model + warmup in a background thread.
        Non-blocking — call at startup so startFollow is instant later."""
        def _do_preload():
            try:
                logger.info("Background preload: loading YOLO model...")
                self._ensure_model()
                logger.info("Background preload: model ready")
            except Exception as e:
                logger.error(f"Background preload failed: {e}")
            finally:
                self._preload_ready.set()

        t = threading.Thread(target=_do_preload, daemon=True, name="yolo-preload")
        t.start()

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
        self._sprinting = False
        self._first_frame = True
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()

    # ── Screen Capture Init ─────────────────────────────────────────────────

    def _init_screen_capture(self):
        """
        Returns a callable that grabs BGR frames.
        Priority:
          1. bettercam (early-init'd before CUDA to avoid DXGI conflict)
          2. bettercam (late init — works if CUDA didn't poison DXGI)
          3. mss (GDI BitBlt — works everywhere, ~10 FPS)
        """
        import numpy as np
        global _early_camera, _early_camera_backend

        # ── 1) Use early-initialised bettercam if available ──
        if _early_camera is not None:
            self._camera = _early_camera
            _early_camera = None  # transfer ownership
            logger.info("Using early-initialised bettercam (pre-CUDA)")

            cam_ref = self._camera
            def _grab_bettercam_early():
                return cam_ref.grab()
            return _grab_bettercam_early

        # ── 2) Try bettercam late (may work on single-GPU desktops) ──
        try:
            import bettercam

            for args in [
                {},
                {"output_idx": 0},
                {"device_idx": 0, "output_idx": 0},
            ]:
                try:
                    self._camera = bettercam.create(output_color="BGR", **args)
                    time.sleep(0.15)
                    logger.info(f"bettercam late-init OK ({args or 'default'})")

                    cam_ref = self._camera
                    def _grab_bettercam_late():
                        return cam_ref.grab()
                    return _grab_bettercam_late
                except Exception:
                    self._camera = None
                    continue

            logger.warning("bettercam: all late-init attempts failed")
        except ImportError:
            pass

        # ── 3) Fallback: mss (GDI — works everywhere, ~10 FPS) ──
        try:
            import mss

            sct = mss.mss()
            monitor_cfg = getattr(self.config, "vision_monitor", 1)
            if monitor_cfg >= len(sct.monitors):
                monitor_cfg = 1
            monitor = sct.monitors[monitor_cfg]
            self._camera = sct
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

    def _close_screen_capture(self):
        camera = self._camera
        if camera is None:
            return
        for method_name in ("stop", "release", "close"):
            method = getattr(camera, method_name, None)
            if not method:
                continue
            try:
                method()
            except Exception:
                pass
        self._camera = None

    def _cleanup_inference_cache(self, torch_module=None):
        try:
            predictor = getattr(self.model, "predictor", None) if self.model else None
            if predictor is not None and hasattr(predictor, "results"):
                predictor.results = None
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Tracker cache cleanup skipped: {e}")
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def _maybe_refresh_tracker_state(self, torch_module):
        now = time.perf_counter()
        cleanup_interval = float(self._cfg.get("cache_cleanup_interval", 300.0))
        if cleanup_interval > 0 and now >= self._next_cache_cleanup:
            self._cleanup_inference_cache(torch_module)
            self._next_cache_cleanup = now + cleanup_interval

        reset_interval = float(self._cfg.get("tracker_reset_interval", 1800.0))
        if reset_interval > 0 and now >= self._next_tracker_reset:
            self._first_frame = True
            self._locked_id = None
            self._lock_lost_time = None
            self._next_tracker_reset = now + reset_interval
            logger.info("Tracker state refreshed to keep long sessions stable")

    # ── Main Tracking Loop (runs in thread) ───────────────────────────────

    def _run_loop(self):
        import cv2
        import numpy as np

        # ── Init screen capture FIRST (before CUDA loads) ──
        # This ensures the early-bettercam DXGI context is used before
        # torch/CUDA poisoning can occur.
        capture_fn = self._init_screen_capture()
        if capture_fn is None:
            self._active = False
            return

        # ── Now load model (triggers torch/CUDA import) ──
        # If preload() was called at startup, this returns immediately.
        # Otherwise it blocks here to load + warmup.
        if not self._preload_ready.is_set():
            logger.info("Waiting for model preload...")
            self._preload_ready.wait(timeout=60)
        import torch
        self._ensure_model()

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

                # ── Vision debug frame (if server is running) ──
                if self._vision_debug:
                    self._push_debug_frame(resized, results, detections)

                self._maybe_refresh_tracker_state(torch)

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
            self._cleanup_inference_cache(torch if "torch" in locals() else None)
            self._close_screen_capture()
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

        # Forward/backward control based on bounding-box area vs target area
        sprint_area = cfg["sprint_area"]
        too_close_area = cfg.get("too_close_area", target_area * 1.8)
        backup_scale = cfg.get("backup_scale", 0.5)
        if target["area"] < target_area:
            deficit = (target_area - target["area"]) / target_area
            raw_forward = cfg["forward_scale_min"] + deficit * (
                cfg["forward_scale_max"] - cfg["forward_scale_min"]
            )
            raw_forward = min(raw_forward, cfg["forward_scale_max"])
        elif target["area"] > too_close_area:
            excess = (target["area"] - too_close_area) / too_close_area
            raw_forward = -min(excess * backup_scale, backup_scale)
        else:
            raw_forward = 0.0

        # Sprint when target is very far away (smooth transition)
        self._sprinting = target["area"] < sprint_area and raw_forward > 0.3

        # ── EMA smoothing ──
        new_look_h = self._smoothed_look_h * (1 - alpha) + raw_look_h * alpha
        new_look_v = self._smoothed_look_v * (1 - alpha) + raw_look_v * alpha
        new_forward = self._smoothed_forward * (1 - alpha) + raw_forward * alpha

        # ── Rate limiter: cap how fast turn can change per frame ──
        max_rate = cfg["max_turn_rate"]
        delta_h = new_look_h - self._smoothed_look_h
        if abs(delta_h) > max_rate:
            new_look_h = self._smoothed_look_h + max_rate * (1 if delta_h > 0 else -1)

        self._smoothed_look_h = new_look_h
        self._smoothed_look_v = new_look_v
        self._smoothed_forward = new_forward

    # ── Vision Debug ─────────────────────────────────────────────────────

    def _push_debug_frame(self, frame, results, detections):
        """Draw bounding boxes on frame and push to vision debug server."""
        import cv2
        try:
            from vision_server import update_frame
        except ImportError:
            self._vision_debug = False
            return

        annotated = frame.copy()
        boxes = results[0].boxes if results and results[0].boxes is not None else None
        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                conf = float(boxes.conf[i])
                track_id = int(boxes.id[i]) if boxes.id is not None else None

                # Green box for locked target, white for others
                is_locked = track_id is not None and track_id == self._locked_id
                color = (0, 255, 0) if is_locked else (200, 200, 200)
                thickness = 2 if is_locked else 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                label = f"ID:{track_id}" if track_id else "?"
                label += f" {conf:.0%}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Crosshair at center
        ch = 10
        cx, cy = FRAME_W // 2, FRAME_H // 2
        cv2.line(annotated, (cx - ch, cy), (cx + ch, cy), (0, 0, 255), 1)
        cv2.line(annotated, (cx, cy - ch), (cx, cy + ch), (0, 0, 255), 1)

        # Encode to JPEG
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])

        update_frame(jpeg.tobytes(), {
            "fps": self._fps,
            "target_id": self._locked_id,
            "target_area": self._current_target_area,
            "osc_look_h": self._smoothed_look_h,
            "osc_look_v": self._smoothed_look_v,
            "osc_forward": self._smoothed_forward,
            "osc_strafe": 0.0,
            "sprinting": self._sprinting,
            "detections": len(detections),
            "frame_w": FRAME_W,
            "frame_h": FRAME_H,
        })

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
        look_v = max(-1.0, min(1.0, self._smoothed_look_v))
        forward = max(-1.0, min(1.0, self._smoothed_forward))

        # ── Turn axis (proportional, smooth in Desktop) ──
        if abs(look_h) < dz:
            look_h = 0.0
        client.send_message("/input/LookHorizontal", float(look_h))

        # ── Vertical look axis ──
        if abs(look_v) < dz:
            look_v = 0.0
        client.send_message("/input/LookVertical", float(look_v))

        # ── Forward / backward axis ──
        client.send_message("/input/Vertical", float(forward))

        # ── Sprint (button: 1 = press, 0 = release) ──
        client.send_message("/input/Run", 1 if self._sprinting else 0)

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
        client.send_message("/input/LookVertical", 0.0)
        client.send_message("/input/Vertical", 0.0)
        client.send_message("/input/Horizontal", 0.0)
        client.send_message("/input/Run", 0)
