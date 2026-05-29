"""
Vision frame buffer -- tracker pushes annotated JPEG frames + stats here,
WebUI Vision tab pulls them from /api/vision/* in control_server.
"""

import threading

_state = {
    "jpeg_bytes": None,  # latest annotated frame as JPEG bytes
    "lock": threading.Lock(),
    "fps": 0.0,
    "target_id": None,
    "target_area": 0.0,
    "osc_look_h": 0.0,
    "osc_forward": 0.0,
    "osc_strafe": 0.0,
    "sprinting": False,
    "detections": 0,
    "frame_w": 0,
    "frame_h": 0,
}


def update_frame(jpeg_bytes, stats):
    """Called by tracker to push an annotated frame + stats."""
    with _state["lock"]:
        _state["jpeg_bytes"] = jpeg_bytes
        _state.update(stats)
