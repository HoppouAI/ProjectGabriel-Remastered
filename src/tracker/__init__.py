"""Player tracking package.

Detects + follows players in VRChat using YOLOv8 for detection and boxmot
DeepOCSORT for ReID-aware tracking. ReID keeps the lock stuck to ONE
person even when they leave the frame or get occluded -- when they come
back, boxmot recognises them by appearance and restores the same id.

IMPORTANT: importing this package triggers _early.py which calls
_try_early_bettercam() before any CUDA / torch import can happen. dont
re-order the imports below.
"""

from . import _early  # noqa: F401 -- side-effect: early bettercam init
from .player import PlayerTracker

__all__ = ["PlayerTracker"]
