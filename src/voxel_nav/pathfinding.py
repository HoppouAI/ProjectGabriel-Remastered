"""26-connected A* over the voxel `Graph`. Port of the reference Pathfinding.cs."""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

from .coords import (
    NodeType,
    Serial,
    _SQRT2,
    _SQRT3,
    _VERTICAL_PENALTY,
    serial_to_center,
)
from .graph import Graph


logger = logging.getLogger(__name__)


def _is_pathable(graph: Graph, s: Serial) -> bool:
    n = graph.get(s)
    return n is not None and n.node_type == NodeType.REACHABLE


def _corner_unreachable(graph: Graph, a: Serial, b: Serial) -> bool:
    """A diagonal step from a->b is blocked if BOTH orthogonal neighbors
    that share the corner are not reachable. Prevents wall clipping."""
    dx = b[0] - a[0]
    dz = b[2] - a[2]
    if _is_pathable(graph, (a[0] + dx, a[1], a[2])):
        return False
    if _is_pathable(graph, (a[0], a[1], a[2] + dz)):
        return False
    return True


def _neighbors(graph: Graph, serial: Serial) -> Iterable[tuple[Serial, float]]:
    sx, sy, sz = serial
    for dy in (-1, 0, 1):
        ny = sy + dy
        same_layer = (dy == 0)
        # orthogonal XZ moves
        ortho_cost = 1.0 if same_layer else _SQRT2
        for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cand = (sx + dx, ny, sz + dz)
            if _is_pathable(graph, cand):
                yield cand, ortho_cost + abs(dy) * _VERTICAL_PENALTY
        # diagonal XZ moves
        diag_cost = _SQRT2 if same_layer else _SQRT3
        for dx, dz in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            cand = (sx + dx, ny, sz + dz)
            if not _is_pathable(graph, cand):
                continue
            if _corner_unreachable(graph, serial, cand):
                continue
            yield cand, diag_cost + abs(dy) * _VERTICAL_PENALTY


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


@dataclass
class VoxelPathResult:
    found: bool
    serials: list[Serial] = field(default_factory=list)        # filtered (turn points)
    full_serials: list[Serial] = field(default_factory=list)   # every cell
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
        return VoxelPathResult(found=True, serials=[goal], full_serials=[goal])

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
            return VoxelPathResult(
                found=True, serials=filtered, full_serials=full,
                cost=g_score[goal], nodes_expanded=expanded,
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
