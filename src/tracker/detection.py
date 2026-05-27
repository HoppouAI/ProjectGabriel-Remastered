"""Detection parsing, target scoring, and the EMA-smoothed targeting state machine."""

import logging
import time

from .config import FRAME_H, FRAME_W

logger = logging.getLogger("src.tracker")


class DetectionMixin:
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

            # normalised distance from frame centre (0 = dead centre, ~1.41 = corner)
            norm_dx = (cx - FRAME_W / 2) / (FRAME_W / 2)
            norm_dy = (cy - FRAME_H / 2) / (FRAME_H / 2)
            center_dist = (norm_dx**2 + norm_dy**2) ** 0.5

            detections.append({
                "id": track_id,
                "cx": cx,
                "cy": cy,
                "area": area,
                "center_dist": center_dist,
                "conf": conf,
            })

        return detections

    def _score(self, det):
        """Lower is better, prefer centred + large."""
        return (
            self._cfg["center_distance_weight"] * det["center_dist"]
            - self._cfg["area_weight"] * det["area"]
        )

    def _update_tracking(self, detections):
        """Pick/maintain target, then update smoothed look_h/look_v/forward."""
        cfg = self._cfg
        alpha = cfg["smoothing_alpha"]
        now = time.time()

        if not detections:
            if self._locked_id is not None:
                if self._lock_lost_time is None:
                    self._lock_lost_time = now
                elif now - self._lock_lost_time > cfg["lock_timeout"]:
                    self._locked_id = None
                    self._lock_lost_time = None

            self._current_target_area = 0.0
            self._smoothed_look_h *= 1 - alpha
            self._smoothed_look_v *= 1 - alpha
            self._smoothed_forward *= 1 - alpha
            return

        trackable = [d for d in detections if d["id"] is not None]
        target = None

        if self._locked_id is not None:
            for d in trackable:
                if d["id"] == self._locked_id:
                    target = d
                    self._lock_lost_time = None
                    break

            if target is None:
                if self._lock_lost_time is None:
                    self._lock_lost_time = now
                elif now - self._lock_lost_time > cfg["lock_timeout"]:
                    self._locked_id = None
                    self._lock_lost_time = None

        if trackable:
            scored = sorted(trackable, key=self._score)
            best = scored[0]

            if target is None:
                target = best
                self._locked_id = best["id"]
                self._lock_lost_time = None
                logger.debug(f"Locked target {self._locked_id}")
            elif best["id"] != self._locked_id:
                # only switch if the new candidate is *significantly* better
                if self._score(target) - self._score(best) > cfg["reacquire_threshold"]:
                    target = best
                    self._locked_id = best["id"]
                    self._lock_lost_time = None
                    logger.debug(f"Switched to better target {self._locked_id}")

        # fallback: pick closest-to-centre un-tracked detection
        if target is None and detections:
            target = min(detections, key=lambda d: d["center_dist"])

        if target is None:
            self._current_target_area = 0.0
            self._smoothed_look_h *= 1 - alpha
            self._smoothed_look_v *= 1 - alpha
            self._smoothed_forward *= 1 - alpha
            return

        self._current_target_area = target["area"]
        deadzone = cfg["deadzone"]
        target_area = cfg["target_area"]

        # normalized screen-space offsets, -1..+1
        dx = (target["cx"] - FRAME_W / 2) / (FRAME_W / 2)
        dy = (target["cy"] - FRAME_H / 2) / (FRAME_H / 2)
        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))

        if abs(dx) < deadzone:
            dx = 0.0
        if abs(dy) < deadzone:
            dy = 0.0

        gain = cfg["turn_gain"]
        raw_look_h = max(-1.0, min(1.0, dx * gain))
        raw_look_v = -dy * 0.4

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

        # sprint when target is very far away
        self._sprinting = target["area"] < sprint_area and raw_forward > 0.3

        new_look_h = self._smoothed_look_h * (1 - alpha) + raw_look_h * alpha
        new_look_v = self._smoothed_look_v * (1 - alpha) + raw_look_v * alpha
        new_forward = self._smoothed_forward * (1 - alpha) + raw_forward * alpha

        # rate limiter: cap how fast turn can change per frame
        max_rate = cfg["max_turn_rate"]
        delta_h = new_look_h - self._smoothed_look_h
        if abs(delta_h) > max_rate:
            new_look_h = self._smoothed_look_h + max_rate * (1 if delta_h > 0 else -1)

        self._smoothed_look_h = new_look_h
        self._smoothed_look_v = new_look_v
        self._smoothed_forward = new_forward
