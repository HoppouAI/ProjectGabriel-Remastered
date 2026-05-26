"""3D voxel navigation, ported from the reference NodeManager + Pathfinding.

Public surface is preserved as flat re-exports so callers keep using
`from src.voxel_nav import VoxelNavManager, world_to_serial, ...`.

Implementation lives in:
    coords.py       -- grid constants, NodeType, Serial, Node, world<->voxel math
    graph.py        -- threadsafe Graph
    pathfinding.py  -- A*, VoxelPathResult, neighbor expansion
    manager.py      -- VoxelNavManager (learning + persistence + discovery)
"""

from .coords import (
    CELL_SIZE,
    HALF_CELL,
    SCALE,
    Node,
    NodeType,
    Serial,
    serial_to_center,
    serial_to_position,
    world_to_serial,
)
from .graph import Graph
from .manager import VoxelNavManager
from .pathfinding import VoxelPathResult, find_path_astar

__all__ = [
    "CELL_SIZE",
    "HALF_CELL",
    "SCALE",
    "Graph",
    "Node",
    "NodeType",
    "Serial",
    "VoxelNavManager",
    "VoxelPathResult",
    "find_path_astar",
    "serial_to_center",
    "serial_to_position",
    "world_to_serial",
]
