"""Standalone screen capture helper.

A tiny, dependency-light way to grab a single screenshot as JPEG bytes
without dragging in the full Gemini Live session stack. Used by the
plugin API (ctx.capture_vision_frame) so any plugin can pull a frame of
whatever the AI sees, on its own schedule, as a background thing.

mss + Pillow only, both core deps. Blocking, so callers in async code
should run it through asyncio.to_thread.
"""

import io
import logging

logger = logging.getLogger(__name__)


def capture_screen_jpeg(monitor_idx: int = 1, max_size: int = 1024, quality: int = 80) -> bytes | None:
    """Grab one screenshot of `monitor_idx`, downscale so the longest edge
    is at most `max_size`, and JPEG-encode at `quality` (1-100).

    Returns the JPEG bytes (mime type image/jpeg), or None if the capture
    deps are missing or the grab fails. Blocking: call it through
    asyncio.to_thread from async code.

    `monitor_idx` follows mss numbering: 0 is the virtual bounding box
    over all monitors, 1 is the primary display, 2+ are extra displays.
    Out-of-range falls back to 0. Pass max_size=0 to skip downscaling.
    """
    try:
        import mss
        from PIL import Image
    except ImportError as e:
        logger.warning(f"screen capture unavailable (mss/PIL missing): {e}")
        return None
    try:
        with mss.mss() as sct:
            if monitor_idx < 0 or monitor_idx >= len(sct.monitors):
                monitor_idx = 0
            monitor = sct.monitors[monitor_idx]
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            if max_size and (img.width > max_size or img.height > max_size):
                img.thumbnail([max_size, max_size])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(quality))
            return buf.getvalue()
    except Exception as e:
        logger.error(f"screen capture failed: {e}")
        return None
