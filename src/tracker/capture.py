"""Screen capture init/teardown. Three backends, fastest first."""

import logging
import time

from . import _early

logger = logging.getLogger(__name__)


class CaptureMixin:
    def _init_screen_capture(self):
        """
        Returns a callable that grabs BGR frames.
        Priority:
          1. bettercam (early-init'd before CUDA to avoid DXGI conflict)
          2. bettercam (late init -- works if CUDA didn't poison DXGI)
          3. mss (GDI BitBlt -- works everywhere, ~10 FPS)
        """
        import numpy as np

        # 1) use early-initialised bettercam if available
        early = _early.consume_early_camera()
        if early is not None:
            self._camera = early
            logger.info("Using early-initialised bettercam (pre-CUDA)")

            cam_ref = self._camera
            def _grab_bettercam_early():
                return cam_ref.grab()
            return _grab_bettercam_early

        # 2) try bettercam late (may work on single-GPU desktops)
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

        # 3) fallback: mss (GDI -- works everywhere, ~10 FPS)
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
