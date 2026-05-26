"""Detection parsing + target selection + movement smoothing.

Now uses boxmot DeepOCSORT for ReID-aware tracking. The id assigned to a
person stays stable across occlusion / leaving frame because boxmot stores
an appearance embedding for each track and matches re-entries by cosine
similarity. This is what makes the lock actually stick to one person.
"""

import logging
import time

from ._constants import FRAME_H, FRAME_W

logger = logging.getLogger(__name__)


class TrackingMixin:
    def _reset_state(self):
        self._locked_id = None
        self._lock_lost_time = None
        self._current_target_area = 0.0
        self._smoothed_look_h = 0.0
        self._smoothed_look_v = 0.0
        self._smoothed_forward = 0.0
        self._sprinting = False
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()
        # boxmot tracker state -- recreated on next frame
        self._bot_tracker = None

    def _parse_yolo_to_dets(self, results):
        """Convert ultralytics Results into the (N, 6) numpy array boxmot wants.

        Columns: [x1, y1, x2, y2, conf, cls]
        """
        import numpy as np

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)

        # only keep person class (0)
        mask = (cls.flatten() == 0)
        if not mask.any():
            return np.empty((0, 6), dtype=np.float32)

        return np.concatenate(
            [xyxy[mask], conf[mask], cls[mask]],
            axis=1,
        ).astype(np.float32)

    def _parse_tracks(self, tracks):
        """Convert boxmot (M, 8) track output into our detection dict format.

        boxmot columns: [x1, y1, x2, y2, id, conf, cls, det_ind]
        """
        detections = []
        if tracks is None or len(tracks) == 0:
            return detections

        for row in tracks:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            conf = float(row[5])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            area = (w * h) / (FRAME_W * FRAME_H)

            norm_dx = (cx - FRAME_W / 2) / (FRAME_W / 2)
            norm_dy = (cy - FRAME_H / 2) / (FRAME_H / 2)
            center_dist = (norm_dx**2 + norm_dy**2) ** 0.5

            detections.append({
                "id": track_id,
                "cx": cx,
                "cy": cy,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": area,
                "center_dist": center_dist,
                "conf": conf,
            })

        return detections

    def _score(self, det):
        """Score a detection -- lower is better (prefer centred + large)."""
        return (
            self._cfg["center_distance_weight"] * det["center_dist"]
            - self._cfg["area_weight"] * det["area"]
        )

    def _update_tracking(self, detections):
        """Select / maintain target and compute smoothed movement values.

        Behavior with boxmot DeepOCSORT:
        - Once locked, we only follow that id. boxmot keeps the id stable
          across occlusion via the ReID embedding.
        - If the locked id isnt in frame, we wait `lock_timeout` seconds
          for boxmot to re-attach the id (eg they walked back into view).
        - After lock_timeout, if `strict_lock` is true we stay idle and
          wait forever. If false, we acquire whoever scores best.
        - Even while locked, we only steal to a new id if it scores
          dramatically better (reacquire_threshold, default 5.0).
        """
        cfg = self._cfg
        alpha = cfg["smoothing_alpha"]
        now = time.time()
        strict = bool(cfg.get("strict_lock", True))

        # no detections
        if not detections:
            if self._locked_id is not None:
                if self._lock_lost_time is None:
                    self._lock_lost_time = now
                elif now - self._lock_lost_time > cfg["lock_timeout"]:
                    if not strict:
                        # legacy behavior -- give up the lock so the next
                        # detection can take over
                        self._locked_id = None
                        self._lock_lost_time = None
                    # strict mode: keep _locked_id set forever, just stay idle

            self._current_target_area = 0.0
            self._smoothed_look_h *= 1 - alpha
            self._smoothed_look_v *= 1 - alpha
            self._smoothed_forward *= 1 - alpha
            return

        trackable = [d for d in detections if d["id"] is not None]
        target = None

        # try to re-find the locked target
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
                    if not strict:
                        self._locked_id = None
                        self._lock_lost_time = None
                    # else: stay locked, ignore other people, idle

        # decide whether to (re)acquire a different target
        if trackable:
            scored = sorted(trackable, key=self._score)
            best = scored[0]

            if target is None and self._locked_id is None:
                # totally unlocked -- acquire the best candidate
                target = best
                self._locked_id = best["id"]
                self._lock_lost_time = None
                logger.info(f"Acquired target id={self._locked_id}")
            elif target is not None and best["id"] != self._locked_id:
                # locked and seeing someone else -- only steal if WAY better
                if self._score(target) - self._score(best) > cfg["reacquire_threshold"]:
                    target = best
                    self._locked_id = best["id"]
                    self._lock_lost_time = None
                    logger.info(f"Switched to better target id={self._locked_id}")

        # if we still have no target but have detections and strict_lock is
        # off, fall back to the closest-to-centre detection (even untracked)
        if target is None and not strict and detections:
            target = min(detections, key=lambda d: d["center_dist"])

        if target is None:
            # strict lock with the original out of view -- idle
            self._current_target_area = 0.0
            self._smoothed_look_h *= 1 - alpha
            self._smoothed_look_v *= 1 - alpha
            self._smoothed_forward *= 1 - alpha
            return

        # compute raw movement values
        self._current_target_area = target["area"]
        deadzone = cfg["deadzone"]
        target_area = cfg["target_area"]

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

        self._sprinting = target["area"] < sprint_area and raw_forward > 0.3

        # ema smoothing
        new_look_h = self._smoothed_look_h * (1 - alpha) + raw_look_h * alpha
        new_look_v = self._smoothed_look_v * (1 - alpha) + raw_look_v * alpha
        new_forward = self._smoothed_forward * (1 - alpha) + raw_forward * alpha

        # rate limiter on turn axis
        max_rate = cfg["max_turn_rate"]
        delta_h = new_look_h - self._smoothed_look_h
        if abs(delta_h) > max_rate:
            new_look_h = self._smoothed_look_h + max_rate * (1 if delta_h > 0 else -1)

        self._smoothed_look_h = new_look_h
        self._smoothed_look_v = new_look_v
        self._smoothed_forward = new_forward
