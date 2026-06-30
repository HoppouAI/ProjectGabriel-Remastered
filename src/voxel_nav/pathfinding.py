"""26-connected A* over the voxel `Graph`."""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

from .coords import (
    NodeType,
    Serial,
    serial_to_center,
)
from .graph import Graph


logger = logging.getLogger(__name__)


def _neighbors(graph: Graph, serial: Serial) -> Iterable[tuple[Serial, float]]:
    """Returns (neighbor_serial, edge_cost) for every REACHABLE 26-connected
    neighbor. Uses Graph.get_pathable_neighbors which acquires the lock once
    for all 26 lookups instead of thrashing it per candidate."""
    return graph.get_pathable_neighbors(serial)


def _heuristic(a: Serial, b: Serial) -> float:
    # straight line distance in cell units (matches Vector3.Distance on
    # node positions since each axis is a uniform scale).
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def _filter_path(path: list[Serial]) -> list[Serial]:
    """Collapse straight runs, keep only turn points and the final goal.

    Direction comparison is done on XZ only (Y changes don't count as a
    "turn" because vertical moves are common on stairs)."""
    if not path or len(path) < 2:
        return path[:]
    out: list[Serial] = []
    for i in range(1, len(path)):
        is_turn = (i == len(path) - 1)
        if i + 1 < len(path):
            a = (path[i + 1][0] - path[i][0], path[i + 1][2] - path[i][2])
            b = (path[i][0] - path[i - 1][0], path[i][2] - path[i - 1][2])
            if a != b:
                is_turn = True
        if is_turn:
            out.append(path[i])
    return out


def _line_of_sight(graph: Graph, a: Serial, b: Serial) -> bool:
    """True if every voxel the straight segment a->b passes through is a
    REACHABLE node. Conservative on purpose: unmapped or wall cells block
    LOS so a shortcut never cuts across geometry we havent confirmed
    walkable. 3D traversal so stairs/ramps stay valid too."""
    if a == b:
        return True
    ax, ay, az = a
    bx, by, bz = b
    sx = bx - ax
    sy = by - ay
    sz = bz - az
    stepx = 1 if sx > 0 else (-1 if sx < 0 else 0)
    stepy = 1 if sy > 0 else (-1 if sy < 0 else 0)
    stepz = 1 if sz > 0 else (-1 if sz < 0 else 0)
    inf = math.inf
    # amanatides-woo voxel walk from center(a) to center(b). t runs 0..1.
    tmax_x = ((ax + (1 if stepx > 0 else 0)) - (ax + 0.5)) / sx if sx != 0 else inf
    tmax_y = ((ay + (1 if stepy > 0 else 0)) - (ay + 0.5)) / sy if sy != 0 else inf
    tmax_z = ((az + (1 if stepz > 0 else 0)) - (az + 0.5)) / sz if sz != 0 else inf
    tdx = abs(1.0 / sx) if sx != 0 else inf
    tdy = abs(1.0 / sy) if sy != 0 else inf
    tdz = abs(1.0 / sz) if sz != 0 else inf
    cx, cy, cz = ax, ay, az
    guard = abs(sx) + abs(sy) + abs(sz) + 4
    while True:
        n = graph.get((cx, cy, cz))
        if n is None or n.node_type != NodeType.REACHABLE:
            return False
        if (cx, cy, cz) == (bx, by, bz):
            return True
        if tmax_x <= tmax_y and tmax_x <= tmax_z:
            cx += stepx
            tmax_x += tdx
        elif tmax_y <= tmax_z:
            cy += stepy
            tmax_y += tdy
        else:
            cz += stepz
            tmax_z += tdz
        guard -= 1
        if guard < 0:
            return False


def _string_pull(graph: Graph, full: list[Serial]) -> list[Serial]:
    """Greedy line-of-sight smoothing. Keeps a waypoint only where LOS from
    the current anchor to the next cell breaks, collapsing the grid
    staircase into long straight runs. Mirrors the 2D planners _smooth_los
    but on the 3D voxel graph."""
    if len(full) <= 2:
        return full[:]
    out: list[Serial] = [full[0]]
    anchor = 0
    for i in range(2, len(full)):
        if not _line_of_sight(graph, full[anchor], full[i]):
            out.append(full[i - 1])
            anchor = i - 1
    out.append(full[-1])
    return out


@dataclass
class VoxelPathResult:
    found: bool
    serials: list[Serial] = field(default_factory=list)        # filtered (turn points)
    full_serials: list[Serial] = field(default_factory=list)   # every cell
    smoothed: list[Serial] = field(default_factory=list)       # LOS string-pulled
    cost: float = 0.0
    nodes_expanded: int = 0

    @property
    def world_waypoints(self) -> list[tuple[float, float, float]]:
        return [serial_to_center(s) for s in self.serials]


def find_path_astar(graph: Graph, start: Serial, goal: Serial,
                    max_nodes: int = 50_000) -> VoxelPathResult:
    """A* over the voxel graph. 26-connected, prevents corner clipping.

    Both `start` and `goal` must already exist in the graph as Reachable
    nodes. Use `Graph.find_closest()` first if you need to snap a free
    world position onto the trail.
    """
    if start not in graph or goal not in graph:
        return VoxelPathResult(found=False)
    if start == goal:
        return VoxelPathResult(found=True, serials=[goal], full_serials=[goal],
                               smoothed=[goal])

    open_set: list[tuple[float, int, Serial]] = []
    came_from: dict[Serial, Serial] = {}
    g_score: dict[Serial, float] = {start: 0.0}
    counter = 0
    heapq.heappush(open_set, (_heuristic(start, goal), counter, start))
    expanded = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            full: list[Serial] = [current]
            while current in came_from:
                current = came_from[current]
                full.append(current)
            full.reverse()
            filtered = _filter_path(full)
            smoothed = _string_pull(graph, full)
            return VoxelPathResult(
                found=True, serials=filtered, full_serials=full,
                smoothed=smoothed, cost=g_score[goal], nodes_expanded=expanded,
            )

        expanded += 1
        if expanded > max_nodes:
            logger.warning("voxel_nav: A* hit max_nodes=%d, aborting", max_nodes)
            break

        for neighbor, cost in _neighbors(graph, current):
            tentative = g_score[current] + cost
            if tentative < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                f = tentative + _heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, counter, neighbor))

    return VoxelPathResult(found=False, nodes_expanded=expanded)
