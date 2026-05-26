"""Vision debug -- pushes annotated frames to the vision server WebUI."""

import logging

from ._constants import FRAME_H, FRAME_W

logger = logging.getLogger(__name__)


class DebugMixin:
    def _push_debug_frame(self, frame, detections):
        """Draw boxmot tracks on frame and push to vision debug server."""
        import cv2
        try:
            from vision_server import update_frame
        except ImportError:
            self._vision_debug = False
            return

        annotated = frame.copy()
        for det in detections:
            x1, y1 = int(det["x1"]), int(det["y1"])
            x2, y2 = int(det["x2"]), int(det["y2"])
            track_id = det["id"]
            conf = det["conf"]

            is_locked = track_id is not None and track_id == self._locked_id
            color = (0, 255, 0) if is_locked else (200, 200, 200)
            thickness = 2 if is_locked else 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{track_id}" if track_id is not None else "?"
            label += f" {conf:.0%}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # crosshair at center
        ch = 10
        cx, cy = FRAME_W // 2, FRAME_H // 2
        cv2.line(annotated, (cx - ch, cy), (cx + ch, cy), (0, 0, 255), 1)
        cv2.line(annotated, (cx, cy - ch), (cx, cy + ch), (0, 0, 255), 1)

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
