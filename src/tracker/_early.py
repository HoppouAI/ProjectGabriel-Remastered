"""Early bettercam init -- must run BEFORE torch/CUDA is imported anywhere.

CUDA linking makes DXGI DuplicateOutput fail on some hybrid GPU systems
(DXGI_ERROR_UNSUPPORTED) because CUDA forces the process onto the discrete
GPU and the iGPU is the one DXGI wants to talk to. by creating the camera
object before any torch import we keep the DXGI context on the iGPU.

importing this module triggers _try_early_bettercam() automatically.
src/tracker/__init__.py must import this module FIRST so the side effect
happens before player.py (which transitively pulls torch through ultralytics).
"""

import logging
import time

logger = logging.getLogger(__name__)


_early_camera = None
_early_camera_backend = None


def _try_early_bettercam():
    """Attempt to create a bettercam camera before any CUDA import."""
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
                time.sleep(0.15)  # let dxgi settle (known bettercam quirk)
                _early_camera = cam
                _early_camera_backend = "bettercam"
                logger.debug(f"Early bettercam init OK ({args or 'default'})")
                return
            except Exception as e:
                logger.debug(f"Early bettercam attempt ({args}): {e}")
                continue
        logger.debug("Early bettercam init failed on all attempts")
    except ImportError:
        logger.debug("bettercam not installed -- skipping early init")


def consume_early_camera():
    """Return the early camera (if any) and clear the module global so the
    caller owns it. Returns None if nothing was pre-initialised."""
    global _early_camera
    cam = _early_camera
    _early_camera = None
    return cam


_try_early_bettercam()
