"""Tests for the A* pathfinder + waypoint store."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from src.spatial_map import OccupancyGrid
from src.pathfinder import find_path
from src.waypoints import WaypointStore


def _block_line(grid: OccupancyGrid, x0, x1, z, votes=5):
    """Paint a horizontal wall of obstacle votes across an X range at fixed Z."""
    step = grid.cell_size * 0.5
    x = x0
    while x <= x1:
        grid.mark_obstacle(x, z, weight=votes)
        x += step


def test_straight_path_through_open_grid():
    grid = OccupancyGrid(cell_size_meters=0.5)
    # totally unknown world, planner should still find a straight line because
    # allow_unknown defaults to True
    res = find_path(grid, (0.0, 0.0), (5.0, 0.0))
    assert res.found
    assert len(res.waypoints) >= 2
    assert res.distance_meters >= 4.5


def test_path_around_wall():
    grid = OccupancyGrid(cell_size_meters=0.5)
    # wall from x=-2 to x=2 at z=3
    _block_line(grid, -2.0, 2.0, 3.0, votes=10)
    res = find_path(grid, (0.0, 0.0), (0.0, 6.0), inflate_obstacles=0)
    assert res.found
    # path must not pass through the wall row, expect it to bend around
    crossed_wall_in_range = any(
        abs(z - 3.0) < 0.25 and -2.5 <= x <= 2.5
        for (x, z) in res.waypoints
    )
    assert not crossed_wall_in_range


def test_no_path_when_fully_blocked():
    grid = OccupancyGrid(cell_size_meters=0.5)
    # full wall across the whole search corridor + extra margin
    _block_line(grid, -50.0, 50.0, 3.0, votes=10)
    res = find_path(
        grid, (0.0, 0.0), (0.0, 6.0),
        max_cells=2000, inflate_obstacles=0, allow_unknown=False,
    )
    assert not res.found


def test_path_respects_unknown_when_disabled():
    grid = OccupancyGrid(cell_size_meters=0.5)
    # leave grid empty - all unknown
    res = find_path(grid, (0.0, 0.0), (3.0, 0.0), allow_unknown=False)
    assert not res.found


def test_waypoint_store_round_trip(tmp_path):
    root = tmp_path / "wp"
    store = WaypointStore("wrld_TEST-123", root=root)
    store.add("spawn", 0.0, 0.0, note="entrance")
    store.add("bar", 5.5, -2.0, y=1.0, yaw=90.0)
    assert len(store.list()) == 2

    # reopen and confirm persistence
    again = WaypointStore("wrld_TEST-123", root=root)
    assert {w.name for w in again.list()} == {"spawn", "bar"}
    bar = again.get("BAR")
    assert bar is not None
    assert bar.x == pytest.approx(5.5)
    assert bar.yaw == pytest.approx(90.0)


def test_waypoint_remove_and_nearest(tmp_path):
    store = WaypointStore("alpha", root=tmp_path / "wp")
    store.add("a", 0.0, 0.0)
    store.add("b", 10.0, 0.0)
    store.add("c", 0.0, 10.0)
    near = store.nearest(0.5, 0.1)
    assert near is not None and near.name == "a"
    assert store.remove("a") is True
    assert store.get("a") is None
    assert store.remove("ghost") is False
