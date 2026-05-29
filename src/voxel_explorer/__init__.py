"""Voxel trail explorer package.

Drives the avatar via OSC inputs to fill in the `VoxelNavManager` graph
and to follow A* paths produced from it.

Public surface stays flat:
    from src.voxel_explorer import VoxelExplorer, ExplorerState

Implementation is split into:
    state.py     -- ExplorerState dataclass
    motion.py    -- MotionMixin (OSC send, CheckImpeded, align)
    follow.py    -- FollowMixin (drive-to-waypoint queue)
    targeting.py -- TargetingMixin (pick / give up on targets)
    explorer.py  -- VoxelExplorer (composes the mixins, owns the tick loop)
"""

from .explorer import VoxelExplorer
from .state import ExplorerState

__all__ = ["VoxelExplorer", "ExplorerState"]
