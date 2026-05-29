"""Voxel coord math, types, constants.

The grid is uniform 0.25 m cubes. A `Serial` is the integer voxel coord
tuple, computed by floor(world * 4). `world_to_serial` and friends are
the only conversions the rest of the system should need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


CELL_SIZE = 0.25
HALF_CELL = 0.125
SCALE = 4.0  # 1.0 / CELL_SIZE

# diagonal cost constants used by the pathfinder. kept here so anyone
# poking at the cost model only has one file to edit.
_SQRT2 = 1.4142135
_SQRT3 = 1.7320508
_VERTICAL_PENALTY = 0.4142


class NodeType(IntEnum):
    REACHABLE = 0
    UNREACHABLE = 1
    IFFY = 2


Serial = tuple[int, int, int]


def world_to_serial(x: float, y: float, z: float) -> Serial:
    return (
        int(math.floor(x * SCALE)),
        int(math.floor(y * SCALE)),
        int(math.floor(z * SCALE)),
    )


def serial_to_center(s: Serial) -> tuple[float, float, float]:
    return (s[0] * CELL_SIZE + HALF_CELL,
            s[1] * CELL_SIZE + HALF_CELL,
            s[2] * CELL_SIZE + HALF_CELL)


def serial_to_position(s: Serial) -> tuple[float, float, float]:
    return (s[0] * CELL_SIZE, s[1] * CELL_SIZE, s[2] * CELL_SIZE)


@dataclass
class Node:
    serial: Serial
    node_type: NodeType = NodeType.REACHABLE
    is_turn: bool = False
    label: str = ""
