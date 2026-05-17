"""A* pathfinding on top of OccupancyGrid.

Plans a route from one world XZ point to another using the 8-connected
grid. Treats CELL_OBSTACLE (and optionally CELL_UNKNOWN) as impassable.

Output is a list of waypoint world coordinates suitable for feeding to
a wanderer or navigator that drives OSC inputs to walk the path.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass

from src.spatial_map import CELL_OBSTACLE, CELL_UNKNOWN, OccupancyGrid

logger = logging.getLogger(__name__)

# 8-connected neighbours: (dx, dz, cost)
_NEIGHBOURS = [
    (1, 0, 1.0),  (-1, 0, 1.0),  (0, 1, 1.0),  (0, -1, 1.0),
    (1, 1, 1.41421356), (1, -1, 1.41421356),
    (-1, 1, 1.41421356), (-1, -1, 1.41421356),
]


@dataclass
class PathResult:
    found: bool
    waypoints: list[tuple[float, float]]   # list of (world_x, world_z)
    cells_expanded: int
    distance_meters: float


def _heuristic(cx: int, cz: int, gx: int, gz: int) -> float:
    # octile distance, admissible for 8-connected grids
    dx = abs(cx - gx)
    dz = abs(cz - gz)
    return (dx + dz) + (1.41421356 - 2) * min(dx, dz)


def _is_passable(
    grid: OccupancyGrid,
    cx: int,
    cz: int,
    *,
    allow_unknown: bool,
    obstacle_confidence: int,
) -> bool:
    wx, wz = grid.cell_to_world_center(cx, cz)
    if grid.is_blocked(wx, wz, min_confidence=obstacle_confidence):
        return False
    state = grid.state(wx, wz)
    if state == CELL_OBSTACLE:
        return False
    if state == CELL_UNKNOWN and not allow_unknown:
        return False
    return True


def find_path(
    grid: OccupancyGrid,
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    max_cells: int = 20000,
    allow_unknown: bool = True,
    obstacle_confidence: int = 2,
    inflate_obstacles: int = 1,
) -> PathResult:
    """Plan a path from start (wx, wz) to goal (wx, wz) on the occupancy
    grid. Returns a PathResult with world-space waypoints.

    `allow_unknown` lets the planner traverse never-seen cells. Off for
    strict safety, on by default so the AI can explore.
    `inflate_obstacles` adds a Chebyshev radius around blocked cells so the
    robot doesnt clip walls.
    """
    sx, sz = start
    gx, gz = goal
    start_cell = grid.world_to_cell(sx, sz)
    goal_cell = grid.world_to_cell(gx, gz)

    if start_cell == goal_cell:
        return PathResult(True, [start, goal], 0, math.hypot(gx - sx, gz - sz))

    # precompute inflated blocked set for the local search area only -- cheap
    # since we just consult is_passable per neighbour, but we apply inflation
    # by also checking adjacent cells.
    def passable(cx, cz):
        if not _is_passable(grid, cx, cz, allow_unknown=allow_unknown,
                            obstacle_confidence=obstacle_confidence):
            return False
        if inflate_obstacles > 0:
            for dx in range(-inflate_obstacles, inflate_obstacles + 1):
                for dz in range(-inflate_obstacles, inflate_obstacles + 1):
                    if dx == 0 and dz == 0:
                        continue
                    wx, wz = grid.cell_to_world_center(cx + dx, cz + dz)
                    if grid.is_blocked(wx, wz, min_confidence=obstacle_confidence):
                        return False
        return True

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    heapq.heappush(open_heap, (
        _heuristic(*start_cell, *goal_cell), counter, start_cell,
    ))
    expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            return _build_result(grid, came_from, current, start_cell)
        expanded += 1
        if expanded > max_cells:
            logger.info("pathfinder: aborting, expanded %d cells", expanded)
            return PathResult(False, [], expanded, 0.0)

        cx, cz = current
        for dx, dz, step_cost in _NEIGHBOURS:
            nb = (cx + dx, cz + dz)
            if not passable(*nb):
                continue
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                counter += 1
                f = tentative + _heuristic(*nb, *goal_cell)
                heapq.heappush(open_heap, (f, counter, nb))

    return PathResult(False, [], expanded, 0.0)


def _build_result(
    grid: OccupancyGrid,
    came_from: dict[tuple[int, int], tuple[int, int]],
    end_cell: tuple[int, int],
    start_cell: tuple[int, int],
) -> PathResult:
    path_cells: list[tuple[int, int]] = [end_cell]
    cur = end_cell
    while cur in came_from:
        cur = came_from[cur]
        path_cells.append(cur)
    path_cells.reverse()
    # convert to world coords using cell centres
    waypoints = [grid.cell_to_world_center(cx, cz) for cx, cz in path_cells]
    # simplify with line-of-sight smoothing so the avatar doesnt wiggle
    waypoints = _smooth_los(grid, waypoints)
    dist = 0.0
    for i in range(1, len(waypoints)):
        x0, z0 = waypoints[i - 1]
        x1, z1 = waypoints[i]
        dist += math.hypot(x1 - x0, z1 - z0)
    return PathResult(True, waypoints, len(came_from), dist)


def _smooth_los(grid: OccupancyGrid, waypoints: list[tuple[float, float]]
                ) -> list[tuple[float, float]]:
    """String pull: keep only waypoints that change line-of-sight visibility."""
    if len(waypoints) <= 2:
        return waypoints
    result = [waypoints[0]]
    anchor_idx = 0
    for i in range(2, len(waypoints)):
        if not _line_clear(grid, waypoints[anchor_idx], waypoints[i]):
            result.append(waypoints[i - 1])
            anchor_idx = i - 1
    result.append(waypoints[-1])
    return result


def _line_clear(grid: OccupancyGrid, a: tuple[float, float],
                b: tuple[float, float]) -> bool:
    """Bresenham-ish line check on the grid, stepping at half-cell intervals."""
    x0, z0 = a
    x1, z1 = b
    dist = math.hypot(x1 - x0, z1 - z0)
    if dist == 0:
        return True
    step = grid.cell_size * 0.5
    steps = max(1, int(dist / step))
    for i in range(1, steps):
        t = i / steps
        wx = x0 + (x1 - x0) * t
        wz = z0 + (z1 - z0) * t
        if grid.is_blocked(wx, wz, min_confidence=2):
            return False
    return True
