"""Vision tab (YOLO tracker debug stream + live config)."""
from __future__ import annotations

import json
import time as _t
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..models import YoloConfigInput
from ..shared import shared_state

router = APIRouter()

YOLO_CONFIG_PATH = Path("models/yolov8/config.json")

# whitelist of keys the UI is allowed to write. anything else is ignored
_YOLO_CFG_KEYS = {
    "confidence_threshold", "iou_threshold", "target_area", "sprint_area",
    "deadzone", "smoothing_alpha", "turn_gain", "max_turn_rate",
    "center_distance_weight", "area_weight", "lock_timeout", "reacquire_threshold",
    "max_detections", "forward_scale_min", "forward_scale_max",
    "strafe_threshold", "strafe_scale", "too_close_area", "backup_scale",
}


def _vision_state():
    try:
        from vision_server import _state as vs_state
        return vs_state
    except Exception:
        return None


@router.get("/api/vision/state")
async def vision_state():
    vs = _vision_state()
    tracker = shared_state.get("tracker")
    enabled = bool(tracker)
    if not vs:
        return JSONResponse({
            "enabled": enabled,
            "has_frame": False,
            "fps": 0.0,
            "message": "vision module not loaded",
        })
    with vs["lock"]:
        return JSONResponse({
            "enabled": enabled,
            "has_frame": vs["jpeg_bytes"] is not None,
            "fps": vs["fps"],
            "target_id": vs["target_id"],
            "target_area": vs["target_area"],
            "osc_look_h": vs["osc_look_h"],
            "osc_forward": vs["osc_forward"],
            "osc_strafe": vs["osc_strafe"],
            "sprinting": vs["sprinting"],
            "detections": vs["detections"],
            "frame_w": vs["frame_w"],
            "frame_h": vs["frame_h"],
        })


@router.get("/api/vision/stream")
async def vision_stream():
    vs = _vision_state()
    if not vs:
        raise HTTPException(status_code=404, detail="vision module unavailable")

    def generate():
        while True:
            with vs["lock"]:
                frame = vs["jpeg_bytes"]
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            _t.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/api/vision/yolo-config")
async def vision_yolo_config_get():
    tracker = shared_state.get("tracker")
    if tracker and hasattr(tracker, "get_config"):
        live = tracker.get_config()
    elif YOLO_CONFIG_PATH.exists():
        try:
            live = json.loads(YOLO_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to read config: {e}")
    else:
        live = {}
    return {"config": live, "tracker_running": bool(tracker)}


@router.post("/api/vision/yolo-config")
async def vision_yolo_config_set(body: YoloConfigInput):
    if YOLO_CONFIG_PATH.exists():
        try:
            current = json.loads(YOLO_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    else:
        current = {}

    incoming = body.config or {}
    rejected = []
    for k, v in incoming.items():
        if k not in _YOLO_CFG_KEYS:
            rejected.append(k)
            continue
        try:
            current[k] = float(v) if not isinstance(v, bool) else v
        except (TypeError, ValueError):
            rejected.append(k)

    YOLO_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    YOLO_CONFIG_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")

    reloaded = False
    tracker = shared_state.get("tracker")
    if tracker and hasattr(tracker, "reload_config"):
        try:
            reloaded = bool(tracker.reload_config())
        except Exception:
            reloaded = False

    return {"success": True, "reloaded": reloaded, "rejected": rejected, "config": current}
