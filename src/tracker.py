import os
import json
import logging
import asyncio
import time
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLOTracker:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.active = False
        self._load_yolo_config()
        self._last_target = None
        self._last_direction = None
        self._no_player_time = 0

    def _load_yolo_config(self):
        config_path = Path(self.config.yolo_model_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.yolo_config = json.load(f)
        else:
            self.yolo_config = {
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "input_size": 640,
                "max_distance": 0.6,
                "min_distance": 0.3,
                "deadzone": 0.15,
                "reference_height": 200,
                "reference_distance": 1.0,
                "update_interval": 0.1,
            }

    def ensure_model(self):
        if self.model is not None:
            return

        from ultralytics import YOLO
        import torch

        model_dir = Path(self.config.yolo_model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / self.config.yolo_model_name

        if not model_path.exists():
            logger.info("Downloading YOLO26n model...")
            temp_model = YOLO(self.config.yolo_model_name)
            default_path = Path(self.config.yolo_model_name)
            if default_path.exists():
                shutil.move(str(default_path), str(model_path))
            self.model = temp_model
        else:
            self.model = YOLO(str(model_path))

        # Use CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        logger.info(f"YOLO26n model loaded on {device}")

    def detect_players(self, frame):
        """Detect all people in frame and return sorted by estimated distance."""
        if self.model is None:
            return []
        
        import cv2
        
        # Resize frame to 640x640 for YOLO input
        frame_h, frame_w = frame.shape[:2]
        resized = cv2.resize(frame, (640, 640))
        
        results = self.model(
            resized,
            conf=self.yolo_config.get("confidence_threshold", 0.5),
            iou=self.yolo_config.get("iou_threshold", 0.45),
            verbose=False,
        )
        
        if not results or len(results[0].boxes) == 0:
            return []
        
        # Scale factors to convert back to original frame coordinates
        scale_x = frame_w / 640
        scale_y = frame_h / 640
        
        players = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            # Class 0 is person in COCO dataset
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Scale back to original frame size
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                box_height = y2 - y1
                distance = self._estimate_distance(box_height)
                conf = float(box.conf[0])
                players.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1 + x2) / 2,
                    "distance": distance,
                    "confidence": conf,
                })
        
        # Sort by distance (closest first)
        players.sort(key=lambda p: p["distance"])
        return players

    def _estimate_distance(self, box_height):
        """Estimate distance based on bounding box height."""
        ref_height = self.yolo_config.get("reference_height", 200)
        ref_distance = self.yolo_config.get("reference_distance", 1.0)
        if box_height <= 0:
            return float("inf")
        return ref_distance * (ref_height / box_height)

    async def tracking_loop(self, osc):
        """Main tracking loop that follows detected players."""
        import mss
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        try:
            import pygetwindow as gw
        except ImportError:
            gw = None
            logger.warning("pygetwindow not installed, using full monitor capture")

        self.ensure_model()
        sct = mss.mss()
        
        # Try to get VRChat window, fall back to monitor
        monitor = None
        if gw:
            windows = gw.getWindowsWithTitle("VRChat")
            if windows:
                win = windows[0]
                try:
                    win.activate()
                except Exception:
                    pass  # Window might already be active
                monitor = {
                    "left": win.left,
                    "top": win.top,
                    "width": win.width,
                    "height": win.height,
                }
                logger.info(f"Tracking VRChat window: {win.width}x{win.height} at ({win.left}, {win.top})")
            else:
                logger.warning("VRChat window not found, available windows: " + 
                             str([w.title for w in gw.getAllWindows() if w.title][:10]))
        
        if not monitor:
            monitor_idx = getattr(self.config, 'vision_monitor', 1)
            if monitor_idx >= len(sct.monitors):
                monitor_idx = 1
            monitor = sct.monitors[monitor_idx]
            logger.info(f"Tracking monitor {monitor_idx}: {monitor['width']}x{monitor['height']}")

        self._last_target = None
        self._last_direction = None
        self._no_player_time = 0

        logger.info("Person tracking started")
        
        # Use a thread pool for CPU-bound YOLO inference to avoid blocking audio
        with ThreadPoolExecutor(max_workers=1) as executor:
            while self.active:
                try:
                    frame = np.array(sct.grab(monitor))[:, :, :3]
                    # Run detection in thread pool to not block audio
                    players = await asyncio.get_event_loop().run_in_executor(
                        executor, self.detect_players, frame
                    )
                    await self._track_and_move_with_players(players, monitor["width"], monitor["height"], osc)
                    await asyncio.sleep(self.yolo_config.get("update_interval", 0.1))
                except Exception as e:
                    logger.error(f"Tracking error: {e}")
                    await asyncio.sleep(0.5)

        osc.stop_movement()
        logger.info("Person tracking stopped")

    async def _track_and_move_with_players(self, players, width, height, osc):
        """Track player and send movement commands (using pre-detected players)."""
        max_dist = self.yolo_config.get("max_distance", 0.6)
        min_dist = self.yolo_config.get("min_distance", 0.3)
        deadzone = self.yolo_config.get("deadzone", 0.15)
        
        if players:
            self._no_player_time = 0
            
            # Use closest player
            target = players[0]
            
            distance = target["distance"]
            player_center_x = target["cx"]
            screen_center_x = width / 2
            
            # Forward/backward movement based on distance
            if distance > max_dist:
                osc._move_forward()
                osc._stop_backward()
            elif distance < min_dist:
                osc._move_backward()
                osc._stop_forward()
            else:
                osc._stop_forward()
                osc._stop_backward()
            
            # Rotation based on horizontal deviation
            dead_zone_pixels = width * deadzone
            deviation = player_center_x - screen_center_x
            
            if deviation > dead_zone_pixels and self._last_direction != "right":
                await osc.rotate_right()
                self._last_direction = "right"
            elif deviation < -dead_zone_pixels and self._last_direction != "left":
                await osc.rotate_left()
                self._last_direction = "left"
            else:
                self._last_direction = None
            
            self._last_target = target
        else:
            # No player detected
            osc._stop_forward()
            osc._stop_backward()
            self._no_player_time += 1
            
            # After a while, search by rotating
            if self._no_player_time > 50:
                await osc.rotate_right()
            
            self._last_direction = None
            self._last_target = None
