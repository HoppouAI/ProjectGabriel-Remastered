"""Pre-CUDA bettercam initialization.

Must be imported BEFORE torch/CUDA loads. CUDA forces hybrid-GPU machines
onto the discrete GPU, which makes DXGI DuplicateOutput fail. By grabbing
a bettercam handle while we're still on the iGPU, the DXGI context is
already established and the late-init path can reuse it.

Calling this at module import time means just `import src.tracker` (which
imports this module) is enough to trigger the early grab.
"""

import logging
import time

logger = logging.getLogger("src.tracker")

_early_camera = None
_early_camera_backend = None


def _try_early_bettercam():
    """Try to create a bettercam camera before any CUDA import."""
    global _early_camera, _early_camera_backend
    try:
        import bettercam
        for args in [
            {},
            {"output_idx": 0},
            {"device_idx": 0, "output_idx": 0},
        ]:
            try:
                cam = bettercam.create(output_color="BGR", **args)
                time.sleep(0.15)  # let DXGI settle (bettercam quirk)
                _early_camera = cam
                _early_camera_backend = "bettercam"
                logger.debug(f"Early bettercam init OK ({args or 'default'})")
                return
            except Exception as e:
                logger.debug(f"Early bettercam attempt ({args}): {e}")
                continue
        logger.debug("Early bettercam init failed on all attempts")
    except ImportError:
        logger.debug("bettercam not installed, skipping early init")


def take_early_camera():
    """Pop the early camera handle (caller takes ownership)."""
    global _early_camera
    cam = _early_camera
    _early_camera = None
    return cam


_try_early_bettercam()
