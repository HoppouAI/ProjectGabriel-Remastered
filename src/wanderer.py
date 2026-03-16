"""
Autonomous wandering system using DPT-Large depth estimation.

Takes screenshots, estimates depth, and uses reactive obstacle avoidance
to navigate VRChat maps without running into walls. The AI can toggle
wandering on/off via function calls.

Screen capture: mss (avoids bettercam conflict with player tracker).
Depth model: Intel/dpt-large from HuggingFace (transformers).
Navigation: Reactive controller - analyze depth zones and steer away from obstacles.
"""

import logging
import random
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

FRAME_W = 384
FRAME_H = 288
TARGET_FPS = 10  # Smaller frames allow higher FPS for faster reactions

DEFAULT_CFG = {
    "wall_threshold": 0.45,      # Depth value above this = obstacle (0-1 normalized, higher = closer)
    "obstacle_ratio": 0.35,      # If this fraction of a zone is blocked, consider it an obstacle
    "forward_speed": 0.6,        # Base forward movement speed
    "turn_speed": 0.5,           # Turn speed when avoiding obstacles
    "smoothing_alpha": 0.5,        # EMA smoothing for movement (higher = faster reaction)
    "random_turn_chance": 0.08,  # Chance per frame to do a random turn
    "random_look_chance": 0.05,  # Chance per frame to look up/down
    "jump_chance": 0.02,         # Chance per frame to jump
    "min_straight_time": 2.0,    # Min seconds to walk straight before random turn
    "zone_left_range": [0.0, 0.33],    # Left third of screen
    "zone_center_range": [0.25, 0.75], # Center of screen (wider for safety)
    "zone_right_range": [0.67, 1.0],   # Right third of screen
}


class Wanderer:
    """Autonomous VRChat wanderer using depth estimation for obstacle avoidance."""

    def __init__(self, config, osc=None):
        self.config = config
        self.osc = osc
        self._active = False
        self._thread = None
        self._model = None
        self._transform = None
        self._device = None
        self._preload_ready = threading.Event()
        self._face_tracker_ref = None
        self._emotion_system_ref = None

        # Navigation state
        self._smoothed_turn = 0.0
        self._smoothed_forward = 0.0
        self._smoothed_look_v = 0.0
        self._last_straight_time = 0.0
        self._current_action = "idle"
        self._stuck_count = 0        # Consecutive frames where stuck (velocity near zero)
        self._stuck_turn_dir = 1.0   # Sustained turn direction when stuck
        self._moving_stuck_frames = 0  # Frames where we're sending forward but VelocityZ is ~0
        self._committed_turn_dir = 0.0  # Anti-oscillation: committed turn direction
        self._committed_turn_until = 0.0  # Time until which the turn commitment holds

        # Config
        self._cfg = dict(DEFAULT_CFG)

    @property
    def active(self):
        return self._active

    # ── Model Loading ─────────────────────────────────────────────────────

    def preload(self):
        """Pre-load DPT model in background thread."""
        def _do_preload():
            try:
                logger.info("Wanderer: loading DPT-Large depth model...")
                self._load_model()
                logger.info("Wanderer: model ready")
            except Exception as e:
                logger.error(f"Wanderer: preload failed: {e}")
            finally:
                self._preload_ready.set()

        t = threading.Thread(target=_do_preload, daemon=True, name="wanderer-preload")
        t.start()

    def _load_model(self):
        """Load Intel DPT-Large model from HuggingFace."""
        if self._model is not None:
            return

        import torch
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        self._transform = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self._model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        if torch.cuda.is_available():
            self._device = "cuda"
            self._model.to("cuda")
            logger.info(f"DPT-Large on CUDA ({torch.cuda.get_device_name(0)})")
        else:
            self._device = "cpu"
            logger.info("DPT-Large on CPU (will be slow)")

        self._model.eval()

    # ── Screen Capture (mss) ──────────────────────────────────────────────

    def _init_screen_capture(self):
        """Initialize mss screen capture, returns a grab callable."""
        import mss as mss_lib

        sct = mss_lib.mss()
        monitor_cfg = getattr(self.config, "vision_monitor", 1)
        if monitor_cfg >= len(sct.monitors):
            monitor_cfg = 1
        monitor = sct.monitors[monitor_cfg]
        logger.info(f"Wanderer: mss initialized (monitor {monitor_cfg}: {monitor['width']}x{monitor['height']})")

        def _grab():
            return np.array(sct.grab(monitor))[:, :, :3]  # BGR, drop alpha

        return _grab

    # ── Depth Estimation ──────────────────────────────────────────────────

    def _estimate_depth(self, frame):
        """Run DPT-Large on a frame and return normalized depth map (0=far, 1=close)."""
        import torch
        from PIL import Image
        import cv2

        # Resize for model input
        rgb = cv2.cvtColor(cv2.resize(frame, (FRAME_W, FRAME_H)), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        inputs = self._transform(images=img, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            depth = outputs.predicted_depth

        # Interpolate to frame size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(FRAME_H, FRAME_W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # DPT-Large outputs disparity-like values: higher raw = closer to camera
        # Normalize to 0-1 where 1.0 = closest (obstacle), 0.0 = farthest (clear)
        # No inversion needed - native output direction is correct
        depth_np = depth.cpu().numpy()
        d_min, d_max = depth_np.min(), depth_np.max()
        if not hasattr(self, "_raw_logged"):
            logger.info(f"Wanderer raw depth range: min={d_min:.2f} max={d_max:.2f}")
            self._raw_logged = True
        if d_max - d_min > 1e-6:
            depth_norm = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_np)

        return depth_norm

    # ── Zone Analysis ─────────────────────────────────────────────────────

    def _analyze_zones(self, depth_map):
        """Split depth map into left/center/right zones and check for obstacles.

        Returns dict with obstacle ratios per zone (0.0 = clear, 1.0 = fully blocked).
        Analyzes from 20% to 85% height to catch furniture and low obstacles
        while avoiding the sky (top) and the immediate floor at feet (bottom).
        """
        cfg = self._cfg
        h, w = depth_map.shape
        y_start = int(h * 0.20)
        y_end = int(h * 0.85)

        crop = depth_map[y_start:y_end, :]
        threshold = cfg["wall_threshold"]

        zones = {}
        for name, (x0_frac, x1_frac) in [
            ("left", cfg["zone_left_range"]),
            ("center", cfg["zone_center_range"]),
            ("right", cfg["zone_right_range"]),
        ]:
            x0 = int(x0_frac * w)
            x1 = int(x1_frac * w)
            zone = crop[:, x0:x1]
            blocked = np.mean(zone > threshold)
            zones[name] = blocked

        return zones

    # ── Navigation Decision ───────────────────────────────────────────────

    def _decide_movement(self, zones):
        """Given zone obstacle ratios, decide turn and forward values."""
        cfg = self._cfg
        alpha = cfg["smoothing_alpha"]
        obs_threshold = cfg["obstacle_ratio"]

        center_blocked = zones["center"] > obs_threshold
        left_blocked = zones["left"] > obs_threshold
        right_blocked = zones["right"] > obs_threshold

        now = time.monotonic()
        target_turn = 0.0
        target_forward = cfg["forward_speed"]
        target_look_v = 0.0

        if center_blocked:
            if left_blocked and right_blocked:
                # Cornered - sustained hard turn in one direction + back up
                self._stuck_count += 1
                if self._stuck_count == 1:
                    # Pick direction only if not already committed (avoids flip on depth flicker)
                    if now >= self._committed_turn_until:
                        self._stuck_turn_dir = random.choice([-1.0, 1.0])
                    else:
                        self._stuck_turn_dir = self._committed_turn_dir
                target_turn = self._stuck_turn_dir * 1.0
                target_forward = -0.3
                self._current_action = "stuck"
                self._smoothed_turn = target_turn
                self._committed_turn_dir = self._stuck_turn_dir
                self._committed_turn_until = max(self._committed_turn_until, now + 1.0)
            elif left_blocked:
                target_turn = 0.8  # Turn right
                target_forward *= 0.3
                self._current_action = "turning_right"
                self._stuck_count = 0
                if now >= self._committed_turn_until:
                    self._committed_turn_dir = 1.0
                    self._committed_turn_until = now + 0.5
            elif right_blocked:
                target_turn = -0.8  # Turn left
                target_forward *= 0.3
                self._current_action = "turning_left"
                self._stuck_count = 0
                if now >= self._committed_turn_until:
                    self._committed_turn_dir = -1.0
                    self._committed_turn_until = now + 0.5
            else:
                # Center blocked but sides open - turn toward more open side
                if now >= self._committed_turn_until:
                    if zones["left"] < zones["right"]:
                        self._committed_turn_dir = -1.0
                    else:
                        self._committed_turn_dir = 1.0
                    self._committed_turn_until = now + 0.5
                target_turn = self._committed_turn_dir * 0.7
                self._current_action = "turning_left" if self._committed_turn_dir < 0 else "turning_right"
                target_forward *= 0.4
                self._stuck_count = max(0, self._stuck_count - 1)
            self._last_straight_time = now
        elif left_blocked or right_blocked:
            # Side obstacle(s) but center is clear
            # Require higher ratio for side-only reactions to avoid false triggers on doorframes
            side_threshold = obs_threshold + 0.25  # e.g. 0.60 instead of 0.35
            left_strongly_blocked = zones["left"] > side_threshold
            right_strongly_blocked = zones["right"] > side_threshold
            self._stuck_count = max(0, self._stuck_count - 1)
            if left_strongly_blocked and right_strongly_blocked:
                # Hallway: both sides strongly blocked, center clear - just walk straight
                self._current_action = "hallway"
                target_forward = cfg["forward_speed"]
            elif left_strongly_blocked:
                target_turn = 0.6  # Steer right
                target_forward *= 0.7
                self._current_action = "avoid_left"
                if now >= self._committed_turn_until:
                    self._committed_turn_dir = 1.0
                    self._committed_turn_until = now + 0.5
            elif right_strongly_blocked:
                target_turn = -0.6  # Steer left
                target_forward *= 0.7
                self._current_action = "avoid_right"
                if now >= self._committed_turn_until:
                    self._committed_turn_dir = -1.0
                    self._committed_turn_until = now + 0.5
            else:
                # Sides mildly blocked (doorframes etc) - just walk through
                self._current_action = "walking"
            self._last_straight_time = now
        else:
            # Path is clear - walk forward with occasional random behavior
            self._current_action = "walking"
            self._stuck_count = 0

            # Random turn for exploration
            straight_duration = now - self._last_straight_time
            if straight_duration > cfg["min_straight_time"] and random.random() < cfg["random_turn_chance"]:
                target_turn = random.choice([-1, 1]) * random.uniform(0.5, 0.8)
                self._last_straight_time = now
                self._current_action = "random_turn"

            # Random look up/down
            if random.random() < cfg["random_look_chance"]:
                target_look_v = random.uniform(-0.2, 0.2)

        # Anti-oscillation: if we've committed to a turn direction, maintain it
        # This prevents left-right-left flickering in doorways and narrow passages
        if now < self._committed_turn_until and target_turn != 0.0:
            # If the new turn would flip direction, keep the committed one
            if (target_turn > 0) != (self._committed_turn_dir > 0):
                target_turn = self._committed_turn_dir * abs(target_turn)

        # Velocity-based stuck detection: if we're trying to move forward but
        # VelocityZ from VRChat says we're not actually moving, we're stuck
        if self.osc and target_forward > 0.2:
            vel_z = abs(self.osc.velocity_z)
            if vel_z < 0.05:
                self._moving_stuck_frames += 1
            else:
                self._moving_stuck_frames = 0

            # If stuck for 5+ frames (~1 second), override to hard turn
            if self._moving_stuck_frames >= 5:
                if self._moving_stuck_frames == 5:
                    self._stuck_turn_dir = random.choice([-1.0, 1.0])
                target_turn = self._stuck_turn_dir * 1.0
                target_forward = -0.3
                self._current_action = "velocity_stuck"
                self._smoothed_turn = target_turn  # Skip EMA for instant turn
                if self._moving_stuck_frames >= 15:
                    # Really stuck for 3+ seconds, try jumping
                    self._do_jump()
                    self._moving_stuck_frames = 0
        else:
            self._moving_stuck_frames = 0

        # EMA smoothing (skipped for turn when actively avoiding obstacles)
        avoiding = self._current_action in ("stuck", "velocity_stuck", "turning_left", "turning_right", "avoid_left", "avoid_right")
        if avoiding:
            self._smoothed_turn = target_turn  # Instant turn for obstacle avoidance
        else:
            self._smoothed_turn = alpha * target_turn + (1 - alpha) * self._smoothed_turn
        self._smoothed_forward = alpha * target_forward + (1 - alpha) * self._smoothed_forward
        # Vertical look: faster decay back to center (0.7 alpha vs 0.4 for movement)
        self._smoothed_look_v = 0.7 * target_look_v + 0.3 * self._smoothed_look_v

        return self._smoothed_turn, self._smoothed_forward, self._smoothed_look_v

    # ── OSC Output ────────────────────────────────────────────────────────

    def _send_osc(self, turn, forward, look_v):
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/LookHorizontal", float(max(-1, min(1, turn))))
        # Deadzone for vertical look - VRChat treats this as a rate, tiny values accumulate
        if abs(look_v) < 0.05:
            look_v = 0.0
        client.send_message("/input/LookVertical", float(max(-1, min(1, look_v))))
        client.send_message("/input/Vertical", float(max(-1, min(1, forward))))
        client.send_message("/input/Run", 0)
        client.send_message("/input/Horizontal", 0.0)

    def _zero_osc(self):
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/LookHorizontal", 0.0)
        client.send_message("/input/LookVertical", 0.0)
        client.send_message("/input/Vertical", 0.0)
        client.send_message("/input/Horizontal", 0.0)
        client.send_message("/input/Run", 0)

    def _do_jump(self):
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/Jump", 1)
        time.sleep(0.05)
        client.send_message("/input/Jump", 0)

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        """Start wandering."""
        if self._active and self._thread and self._thread.is_alive():
            return {"result": "ok", "message": "already wandering"}

        # Pause face tracker to avoid conflicting OSC commands
        if self._face_tracker_ref and self._face_tracker_ref.active:
            self._face_tracker_ref.stop()

        # Suppress idle animation while wandering
        if self._emotion_system_ref:
            self._emotion_system_ref.set_wandering(True)

        self._active = True
        self._smoothed_turn = 0.0
        self._smoothed_forward = 0.0
        self._smoothed_look_v = 0.0
        self._last_straight_time = time.monotonic()
        self._current_action = "starting"

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="wanderer")
        self._thread.start()
        return {"result": "ok", "message": "started wandering"}

    def stop(self):
        """Stop wandering."""
        if not self._active:
            return {"result": "ok", "message": "not wandering"}

        self._active = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Re-enable idle animation
        if self._emotion_system_ref:
            self._emotion_system_ref.set_wandering(False)

        # Resume face tracker
        if self._face_tracker_ref and not self._face_tracker_ref.active:
            self._face_tracker_ref.start()

        return {"result": "ok", "message": "stopped wandering"}

    # ── Main Loop ─────────────────────────────────────────────────────────

    def _run_loop(self):
        import cv2

        capture_fn = self._init_screen_capture()
        if capture_fn is None:
            logger.error("Wanderer: screen capture failed to init")
            self._active = False
            return

        if not self._preload_ready.is_set():
            logger.info("Wanderer: waiting for model preload...")
            self._preload_ready.wait(timeout=120)
        self._load_model()

        if self._model is None:
            logger.error("Wanderer: model failed to load")
            self._active = False
            return

        logger.info(f"Wanderer started - target {TARGET_FPS} FPS")
        frame_interval = 1.0 / TARGET_FPS
        _log_counter = 0
        _first_frame = True

        try:
            while self._active:
                t0 = time.perf_counter()

                frame = capture_fn()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # Resize for depth estimation
                frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))

                # Depth estimation
                t_depth = time.perf_counter()
                depth_map = self._estimate_depth(frame_resized)
                depth_ms = (time.perf_counter() - t_depth) * 1000

                if _first_frame:
                    d_mean = float(depth_map.mean())
                    d_min_v = float(depth_map.min())
                    d_max_v = float(depth_map.max())
                    logger.info(
                        f"Wanderer first frame: depth min={d_min_v:.4f} mean={d_mean:.4f} max={d_max_v:.4f} | "
                        f"shape={depth_map.shape} | {depth_ms:.0f}ms"
                    )
                    _first_frame = False
                depth_ms = (time.perf_counter() - t_depth) * 1000

                # Analyze zones
                zones = self._analyze_zones(depth_map)

                # Decide movement
                turn, forward, look_v = self._decide_movement(zones)

                # Send OSC
                self._send_osc(turn, forward, look_v)

                # Random jump
                if random.random() < self._cfg["jump_chance"]:
                    self._do_jump()

                # Log every ~2 seconds (or every frame for first 5 frames)
                _log_counter += 1
                if _log_counter <= 5 or _log_counter % (TARGET_FPS * 2) == 0:
                    d_mean = float(depth_map.mean())
                    d_min_v = float(depth_map.min())
                    d_max_v = float(depth_map.max())
                    vel_z = self.osc.velocity_z if self.osc else 0.0
                    logger.info(
                        f"Wanderer: {self._current_action} | "
                        f"zones L={zones['left']:.2f} C={zones['center']:.2f} R={zones['right']:.2f} | "
                        f"turn={turn:.3f} fwd={forward:.3f} lookV={look_v:.3f} | "
                        f"velZ={vel_z:.3f} | "
                        f"depth mean={d_mean:.2f} | "
                        f"{depth_ms:.0f}ms"
                    )

                # Frame pacing
                elapsed = time.perf_counter() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Wanderer loop error: {e}", exc_info=True)
        finally:
            self._zero_osc()
            self._active = False
            logger.info("Wanderer stopped")
