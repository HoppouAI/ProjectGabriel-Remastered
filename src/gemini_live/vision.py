"""Screen capture / vision streaming loop for the Gemini Live session.

Grabs the configured monitor with mss, downscales + JPEG-encodes via PIL,
and drops the frames into the same realtime out queue the audio loop uses.
Honors the various vision_* config knobs (interval, idle slowdown, pause
on output, max size, quality) and auto-tunes for 3.1 models.
"""

import asyncio
import io
import logging

import mss
from PIL import Image

logger = logging.getLogger(__name__)


class VisionLoopMixin:
    def _capture_screen_frame(self):
        try:
            with mss.mss() as sct:
                monitor_idx = self.config.vision_monitor
                if monitor_idx >= len(sct.monitors):
                    monitor_idx = 0
                monitor = sct.monitors[monitor_idx]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                max_size = self.config.vision_max_size
                # Use smaller resolution for 3.1 models to save tokens
                if self.config.is_31_model and max_size > 768:
                    max_size = 768
                if img.width > max_size or img.height > max_size:
                    img.thumbnail([max_size, max_size])
                buffer = io.BytesIO()
                quality = self.config.vision_quality
                # Use lower JPEG quality for 3.1 models (smaller payload)
                if self.config.is_31_model and quality > 60:
                    quality = 60
                img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                return buffer.read()
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None

    async def _capture_screen_loop(self):
        with mss.mss() as sct:
            monitor_idx = self.config.vision_monitor
            if monitor_idx >= len(sct.monitors):
                logger.warning(f"Monitor {monitor_idx} not found, using monitor 0")
                monitor_idx = 0
            monitor = sct.monitors[monitor_idx]
            logger.debug(f"Capturing monitor {monitor_idx}: {monitor['width']}x{monitor['height']}")
        interval = self.config.vision_interval
        # Auto-increase interval for 3.1 models if user hasn't set a higher value
        if self.config.is_31_model and interval < 2.0:
            interval = 2.0
            logger.info(f"Vision interval increased to {interval}s for 3.1 model (token optimization)")
        pause_on_output = self.config.vision_pause_on_output
        pause_on_idle = self.config.vision_pause_on_idle
        idle_interval = self.config.vision_idle_interval
        if pause_on_output:
            logger.debug("Vision pause enabled (skips frames during speech/music, not live music)")
        if pause_on_idle:
            logger.debug(f"Vision slows to {idle_interval}s interval when idle (normal: {interval}s)")
        try:
            while True:
                current_interval = interval
                # Slow down when AI is idle (nobody talking, no active tasks)
                if pause_on_idle and self._is_idle:
                    current_interval = idle_interval
                # Skip frame when AI is speaking or music is playing (unless live music is active)
                if pause_on_output:
                    music_gen = getattr(self.tool_handler, 'music_gen', None)
                    music_gen_active = music_gen.is_active if music_gen else False
                    if not music_gen_active:
                        if self._speaking or self.audio.is_music_playing():
                            await asyncio.sleep(current_interval)
                            continue
                frame = await asyncio.to_thread(self._capture_screen_frame)
                if frame:
                    try:
                        self._out_queue.put_nowait(("video", frame))
                    except asyncio.QueueFull:
                        try:
                            self._out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            self._out_queue.put_nowait(("video", frame))
                        except asyncio.QueueFull:
                            pass
                await asyncio.sleep(current_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Screen capture loop error: {e}")
            raise
