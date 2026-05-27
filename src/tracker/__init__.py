"""Player tracking package.

IMPORTANT: early_capture must be imported first so the bettercam DXGI
context is established before anything else can pull in torch/CUDA.
"""

from . import early_capture  # noqa: F401  (must be first, see module docstring)
from .core import PlayerTracker

__all__ = ["PlayerTracker"]
