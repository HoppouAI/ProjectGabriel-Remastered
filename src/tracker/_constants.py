"""Constants and default config for the player tracker."""

MODEL_DIR = "models/yolov8"
MODEL_NAME = "yolov8n.pt"
REID_MODEL_NAME = "osnet_x1_0_msmt17.pt"
FRAME_W = 640
FRAME_H = 360
TARGET_FPS = 30


DEFAULT_CFG = {
    "confidence_threshold": 0.30,
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
    # target. once this expires we re-acquire the best-scoring track (unless
    # strict_lock is on).
    "lock_timeout": 3.0,
    # when locked, a candidate has to score this much better than the
    # current target to be allowed to steal the lock. set high so noisy
    # frames cant cause id swaps while the target is still in view.
    "reacquire_threshold": 5.0,
    # when true, the tracker will NOT auto-acquire a new target after the
    # lock has timed out. it sits idle and waits forever until boxmot reid
    # re-attaches the original person. useful in crowded worlds where you
    # want to follow exactly one person. default false because vrchat angles
    # (back of head, sitting, far away) often confuse the reid embedding
    # and you usually just want to keep following whoever's nearest.
    "strict_lock": False,
    "max_detections": 10,
    "forward_scale_min": 0.5,
    "forward_scale_max": 0.7,
    "strafe_threshold": 0.25,
    "strafe_scale": 0.6,
    "too_close_area": 0.072,
    "backup_scale": 0.5,
    "cache_cleanup_interval": 300.0,
    "tracker_reset_interval": 1800.0,
    # reid model variant. osnet_x1_0 is the full size one (~10 mb, ~10ms
    # per detection on cuda) and matches way better across angles/avatars
    # than the tiny x0_25 variant. drop back to osnet_x0_25_msmt17.pt if
    # you need more speed.
    "reid_model": "osnet_x1_0_msmt17.pt",
    "reid_half": True,
    # boxmot deepocsort overrides. defaults are tuned for MOT challenge
    # which has very different framerate/density assumptions than vrchat.
    # min_hits=1 means new tracks are returned immediately (vs waiting 3
    # frames). max_age=150 keeps an embedding around for ~5s at 30fps so a
    # quick occlusion / turnaround doesnt lose the id permanently.
    "bot_min_hits": 1,
    "bot_max_age": 150,
    "bot_det_thresh": 0.30,
    "bot_iou_threshold": 0.30,
}
