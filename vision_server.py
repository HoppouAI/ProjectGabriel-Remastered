"""
Vision Debug WebUI — view what the tracker sees with bounding boxes + stats.
Run on a separate port, non-blocking. Enable via config.yml → yolo.vision_debug.
"""

import io
import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Shared state — written by tracker, read by web server
_state = {
    "jpeg_bytes": None,  # latest annotated frame as JPEG
    "lock": threading.Lock(),
    "tracker_ref": None,  # reference to PlayerTracker for stats
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

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Gabriel Vision Debug</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; color: #eee; font-family: monospace; display: flex; flex-direction: column; align-items: center; height: 100vh; }
  h1 { padding: 8px 0; font-size: 16px; color: #0f0; }
  .container { display: flex; gap: 12px; padding: 8px; max-width: 100%; }
  .stream { border: 2px solid #333; background: #000; max-width: 70vw; max-height: 80vh; }
  .stats { background: #1a1a1a; border: 1px solid #333; padding: 12px; min-width: 240px; border-radius: 4px; }
  .stats h2 { color: #0f0; font-size: 14px; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }
  .stat { display: flex; justify-content: space-between; padding: 3px 0; font-size: 13px; }
  .stat .label { color: #888; }
  .stat .value { color: #0f0; font-weight: bold; }
  .stat .value.warn { color: #f80; }
  .stat .value.bad { color: #f00; }
  .bar { height: 8px; background: #333; margin: 2px 0; border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; background: #0f0; transition: width 0.2s; }
  .bar-fill.neg { background: #f00; }
</style>
</head>
<body>
<h1>Gabriel Vision Debug</h1>
<div class="container">
  <img class="stream" src="/vision/stream" alt="YOLO stream" />
  <div class="stats" id="stats">
    <h2>Tracker Stats</h2>
    <div id="stat-lines">Loading...</div>
  </div>
</div>
<script>
async function poll() {
  try {
    const r = await fetch('/vision/data');
    const d = await r.json();
    const fpsClass = d.fps >= 15 ? 'value' : d.fps >= 8 ? 'value warn' : 'value bad';
    let html = '';
    html += stat('FPS', d.fps.toFixed(1), fpsClass);
    html += stat('Target ID', d.target_id ?? 'None');
    html += stat('Target Area', (d.target_area * 100).toFixed(1) + '%');
    html += stat('Detections', d.detections);
    html += stat('Sprinting', d.sprinting ? 'YES' : 'no');
    html += '<h2 style="color:#0f0;font-size:13px;margin:8px 0 4px;border-bottom:1px solid #333;padding-bottom:4px;">OSC Outputs</h2>';
    html += statBar('LookH', d.osc_look_h);
    html += statBar('Forward', d.osc_forward);
    html += statBar('Strafe', d.osc_strafe);
    html += stat('Frame', d.frame_w + 'x' + d.frame_h);
    document.getElementById('stat-lines').innerHTML = html;
  } catch(e) {}
  setTimeout(poll, 200);
}
function stat(label, value, cls) {
  cls = cls || 'value';
  return '<div class="stat"><span class="label">' + label + '</span><span class="' + cls + '">' + value + '</span></div>';
}
function statBar(label, val) {
  const pct = Math.abs(val) * 50;
  const left = val < 0 ? (50 - pct) : 50;
  const cls = val < 0 ? 'bar-fill neg' : 'bar-fill';
  return '<div class="stat"><span class="label">' + label + '</span><span class="value">' + val.toFixed(3) + '</span></div>'
       + '<div class="bar"><div class="' + cls + '" style="margin-left:' + left + '%;width:' + pct + '%"></div></div>';
}
poll();
</script>
</body>
</html>"""


def update_frame(jpeg_bytes, stats):
    """Called by tracker to push an annotated frame + stats."""
    with _state["lock"]:
        _state["jpeg_bytes"] = jpeg_bytes
        _state.update(stats)


def run_vision_server(port=8767, tracker=None):
    """Start the vision debug server in a background thread."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        import uvicorn
    except ImportError:
        logger.warning("Vision debug server requires fastapi+uvicorn — skipping")
        return

    _state["tracker_ref"] = tracker
    vapp = FastAPI(title="Gabriel Vision Debug")

    @vapp.get("/vision", response_class=HTMLResponse)
    async def vision_page():
        return HTML_PAGE

    @vapp.get("/vision/data")
    async def vision_data():
        with _state["lock"]:
            return JSONResponse({
                "fps": _state["fps"],
                "target_id": _state["target_id"],
                "target_area": _state["target_area"],
                "osc_look_h": _state["osc_look_h"],
                "osc_forward": _state["osc_forward"],
                "osc_strafe": _state["osc_strafe"],
                "sprinting": _state["sprinting"],
                "detections": _state["detections"],
                "frame_w": _state["frame_w"],
                "frame_h": _state["frame_h"],
            })

    @vapp.get("/vision/stream")
    async def vision_stream():
        def generate():
            while True:
                with _state["lock"]:
                    frame = _state["jpeg_bytes"]
                if frame is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                time.sleep(0.05)  # ~20 FPS max for the stream

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    def _run():
        uvicorn.run(vapp, host="0.0.0.0", port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True, name="vision-debug-server")
    t.start()
    logger.info(f"Vision debug server started on http://localhost:{port}/vision")
