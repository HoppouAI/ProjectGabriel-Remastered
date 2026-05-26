"""Constants and default config for the player tracker."""

MODEL_DIR = "models/yolov8"
MODEL_NAME = "yolov8n.pt"
REID_MODEL_NAME = "osnet_x0_25_msmt17.pt"
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
    # how long to keep the lock alive after the ReID model stops seeing the
    # target. boxmot will usually re-attach the same id when they come back
    # so we can afford a fairly long wait without grabbing a random person.
    "lock_timeout": 30.0,
    # when locked, a candidate has to score this much better than the
    # current target to be allowed to steal the lock. set high so noisy
    # frames dont cause id swaps.
    "reacquire_threshold": 5.0,
    # when true, the tracker will NOT auto-acquire a new target after the
    # lock has timed out. the only way to get a target back is to see the
    # original person again (boxmot reid keeps their id) or call
    # startFollow again. set to false to get the old auto-grab behavior.
    "strict_lock": True,
    "max_detections": 10,
    "forward_scale_min": 0.5,
    "forward_scale_max": 0.7,
    "strafe_threshold": 0.25,
    "strafe_scale": 0.6,
    "too_close_area": 0.072,
    "backup_scale": 0.5,
    "cache_cleanup_interval": 300.0,
    "tracker_reset_interval": 1800.0,
    # reid model variant. osnet_x0_25 is the tiny one (~3 mb, <5 ms per
    # detection), the bigger ones are more accurate but slower.
    "reid_model": "osnet_x0_25_msmt17.pt",
    "reid_half": True,
}
