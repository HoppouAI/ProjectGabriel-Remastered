"""Constants shared across the tracker package."""

MODEL_DIR = "models/yolov8"
MODEL_NAME = "yolov8n.pt"
FRAME_W = 640
FRAME_H = 360
TARGET_FPS = 30

DEFAULT_CFG = {
    "confidence_threshold": 0.40,
    "iou_threshold": 0.45,
    "target_area": 0.04,
    "sprint_area": 0.025,
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
    # full forward axis while actively sprinting so VRChat doesnt cap us at walk
    "sprint_forward_axis": 1.0,
    # hysteresis: must drop below sprint_area to start, must climb back above
    # this fraction of target_area to stop. avoids on/off flicker at the edge.
    "sprint_release_area_ratio": 0.6,
    "strafe_threshold": 0.25,
    "strafe_scale": 0.6,
    "too_close_area": 0.072,
    "backup_scale": 0.5,
    "cache_cleanup_interval": 300.0,
    "tracker_reset_interval": 1800.0,
}
