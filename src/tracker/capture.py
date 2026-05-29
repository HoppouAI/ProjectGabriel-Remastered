"""Screen capture backends. bettercam preferred (pre-CUDA init), mss fallback."""

import logging
import time

from . import early_capture

logger = logging.getLogger("src.tracker")


class CaptureMixin:
    def _init_screen_capture(self):
        """Return a callable that grabs BGR frames. Tries:
        1. early-initd bettercam (set up before CUDA loaded)
        2. bettercam late-init (single-GPU machines can survive this)
        3. mss (GDI BitBlt, works everywhere, ~10 FPS)
        """
        import numpy as np

        cam = early_capture.take_early_camera()
        if cam is not None:
            self._camera = cam
            logger.info("Using early-initialised bettercam (pre-CUDA)")

            def _grab_bettercam_early():
                return cam.grab()
            return _grab_bettercam_early

        try:
            import bettercam

            for args in [
                {},
                {"output_idx": 0},
                {"device_idx": 0, "output_idx": 0},
            ]:
                try:
                    self._camera = bettercam.create(output_color="BGR", **args)
                    time.sleep(0.15)
                    logger.info(f"bettercam late-init OK ({args or 'default'})")

                    cam_ref = self._camera
                    def _grab_bettercam_late():
                        return cam_ref.grab()
                    return _grab_bettercam_late
                except Exception:
                    self._camera = None
                    continue

            logger.warning("bettercam: all late-init attempts failed")
        except ImportError:
            pass

        try:
            import mss

            sct = mss.mss()
            monitor_cfg = getattr(self.config, "vision_monitor", 1)
            if monitor_cfg >= len(sct.monitors):
                monitor_cfg = 1
            monitor = sct.monitors[monitor_cfg]
            self._camera = sct
            logger.info(
                f"mss initialized (monitor {monitor_cfg}: "
                f"{monitor['width']}x{monitor['height']})"
            )

            def _grab_mss():
                return np.array(sct.grab(monitor))[:, :, :3]

            return _grab_mss
        except Exception as e:
            logger.error(f"Screen capture init failed entirely: {e}")
            return None

    def _close_screen_capture(self):
        camera = self._camera
        if camera is None:
            return
        for method_name in ("stop", "release", "close"):
            method = getattr(camera, method_name, None)
            if not method:
                continue
            try:
                method()
            except Exception:
                pass
        self._camera = None
