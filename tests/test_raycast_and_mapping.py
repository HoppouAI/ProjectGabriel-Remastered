"""Tests for the raycast OSC state, pose strip encoding, and spatial mapping
math. These cover everything testable WITHOUT the Unity-side shader/avatar
being in place yet, so we can develop the python side in isolation.
"""

from __future__ import annotations

import math

import pytest

from src.raycast import (
    RaycastState,
    forward_blocked,
    drop_ahead,
    pick_clear_direction,
)
from src.pose_decoder import (
    GRID_W,
    GRID_H,
    WorldPose,
    decode_strip,
    encode_pose,
)
from src.spatial_map import (
    CELL_FREE,
    CELL_OBSTACLE,
    CELL_UNKNOWN,
    OccupancyGrid,
    RayConfig,
    SpatialMapper,
    _ray_hit_world_xz,
)


# --- raycast OSC state --------------------------------------------------

class TestRaycastState:
    def test_update_splits_name_and_field(self):
        s = RaycastState()
        assert s.update("Fwd_Hit", 1) is True
        assert s.update("Fwd_Distance", 0.75) is True
        assert s.update("Fwd_Ratio", 0.5) is True
        r = s.get("Fwd")
        assert r.hit is True
        assert r.distance == pytest.approx(0.75)
        assert r.ratio == pytest.approx(0.5)

    def test_handles_underscore_in_ray_name(self):
        s = RaycastState()
        assert s.update("Drop_Left_Distance", 1.2) is True
        r = s.get("Drop_Left")
        assert r is not None
        assert r.distance == pytest.approx(1.2)

    def test_rejects_unknown_field(self):
        s = RaycastState()
        assert s.update("Fwd_NotAField", 1.0) is False
        assert s.get("Fwd") is None

    def test_rejects_empty_or_malformed(self):
        s = RaycastState()
        assert s.update("", 1) is False
        assert s.update("Fwd", 1) is False
        assert s.update("_Hit", 1) is False
        assert s.update("Fwd_", 1) is False

    def test_clamps_ratio_and_distance(self):
        s = RaycastState()
        s.update("Fwd_Distance", -5.0)
        s.update("Fwd_Ratio", 2.5)
        r = s.get("Fwd")
        assert r.distance == 0.0
        assert r.ratio == 1.0

    def test_declare_makes_get_return_default(self):
        s = RaycastState()
        s.declare("Fwd")
        r = s.get("Fwd")
        assert r is not None
        assert r.hit is False
        assert r.distance == 0.0

    def test_get_returns_copy(self):
        s = RaycastState()
        s.update("Fwd_Distance", 1.0)
        a = s.get("Fwd")
        a.distance = 999.0
        b = s.get("Fwd")
        assert b.distance == pytest.approx(1.0)


class TestNavigationHelpers:
    def test_forward_blocked_true_when_close_hit(self):
        s = RaycastState()
        s.update("Fwd_Hit", 1)
        s.update("Fwd_Distance", 0.5)
        assert forward_blocked(s, threshold_meters=1.0) is True

    def test_forward_blocked_false_when_far(self):
        s = RaycastState()
        s.update("Fwd_Hit", 1)
        s.update("Fwd_Distance", 5.0)
        assert forward_blocked(s, threshold_meters=1.0) is False

    def test_forward_blocked_false_when_stale(self):
        s = RaycastState()
        # never updated -> stale
        assert forward_blocked(s) is False

    def test_drop_ahead_true_when_no_hit(self):
        s = RaycastState()
        s.update("DropFwd_Hit", 0)
        s.update("DropFwd_Distance", 0.0)
        assert drop_ahead(s) is True

    def test_drop_ahead_false_when_close_floor(self):
        s = RaycastState()
        s.update("DropFwd_Hit", 1)
        s.update("DropFwd_Distance", 0.4)
        assert drop_ahead(s, safe_distance_meters=1.5) is False

    def test_pick_clear_direction_picks_widest(self):
        s = RaycastState()
        s.update("Fwd_Hit", 1)
        s.update("Fwd_Distance", 1.0)
        s.update("Left_Hit", 1)
        s.update("Left_Distance", 3.0)
        s.update("Right_Hit", 1)
        s.update("Right_Distance", 2.0)
        best = pick_clear_direction(
            s,
            [("Fwd", 0), ("Left", -90), ("Right", 90)],
            min_clearance_meters=1.5,
        )
        assert best == "Left"


# --- pose pixel encoding ------------------------------------------------

class TestPoseEncoding:
    def test_round_trip_origin(self):
        original = WorldPose(x=0.0, y=0.0, z=0.0, yaw=0.0, timestamp=0.0)
        data = encode_pose(original)
        assert len(data) == GRID_W * GRID_H * 3
        decoded = decode_strip(data, timestamp=123.0)
        assert decoded is not None
        assert decoded.x == pytest.approx(0.0, abs=0.01)
        assert decoded.y == pytest.approx(0.0, abs=0.01)
        assert decoded.z == pytest.approx(0.0, abs=0.01)
        assert decoded.yaw == pytest.approx(0.0, abs=1.0)
        assert decoded.timestamp == 123.0

    @pytest.mark.parametrize("pose", [
        WorldPose(1.5, 2.5, -3.75, 45.0, 0.0),
        WorldPose(-100.25, 5.0, 250.5, 270.0, 0.0),
        WorldPose(0.01, 0.01, 0.01, 359.9, 0.0),
        WorldPose(-4000.0, -100.0, 4000.0, 180.0, 0.0),
    ])
    def test_round_trip_various(self, pose: WorldPose):
        decoded = decode_strip(encode_pose(pose))
        assert decoded is not None
        assert decoded.x == pytest.approx(pose.x, abs=0.02)
        assert decoded.y == pytest.approx(pose.y, abs=0.02)
        assert decoded.z == pytest.approx(pose.z, abs=0.02)
        # yaw goes through sin/cos with ~0.01 packing error per component,
        # so it can drift up to about 1 degree after atan2
        assert decoded.yaw == pytest.approx(pose.yaw % 360.0, abs=1.0)

    def test_rejects_bad_markers(self):
        data = bytearray(encode_pose(WorldPose(1, 2, 3, 90, 0)))
        # wipe the row 0 RED marker (cell 32)
        idx = (0 * GRID_W + 32) * 3
        data[idx] = 0
        data[idx + 1] = 0
        data[idx + 2] = 0
        assert decode_strip(bytes(data)) is None

    def test_rejects_swapped_markers(self):
        data = bytearray(encode_pose(WorldPose(1, 2, 3, 90, 0)))
        # swap row 0 cell 32 from RED to GREEN
        idx = (0 * GRID_W + 32) * 3
        data[idx] = 0
        data[idx + 1] = 255
        data[idx + 2] = 0
        assert decode_strip(bytes(data)) is None

    def test_rejects_short_input(self):
        assert decode_strip(b"\xFF\x00\x00") is None

    def test_negative_yaw_normalizes(self):
        pose = WorldPose(0, 0, 0, -90.0, 0)
        decoded = decode_strip(encode_pose(pose))
        assert decoded is not None
        assert decoded.yaw == pytest.approx(270.0, abs=1.0)


# --- ray projection -----------------------------------------------------

class TestRayProjection:
    def test_forward_hit_at_origin_yaw_zero(self):
        # avatar at origin, yaw 0 (facing +Z), forward ray hits at 5m
        from src.raycast import RayReading
        pose = WorldPose(0, 0, 0, 0, 0)
        cfg = RayConfig("Fwd", yaw_offset_deg=0, max_distance=10.0)
        reading = RayReading("Fwd", hit=True, distance=5.0, ratio=0.5, last_updated=1.0)
        result = _ray_hit_world_xz(pose, cfg, reading)
        assert result is not None
        wx, wz, dist = result
        assert wx == pytest.approx(0.0, abs=1e-6)
        assert wz == pytest.approx(5.0, abs=1e-6)
        assert dist == pytest.approx(5.0)

    def test_right_ray_when_facing_north(self):
        # yaw 0 = facing +Z, ray offset +90 = right = +X
        from src.raycast import RayReading
        pose = WorldPose(10, 0, 20, 0, 0)
        cfg = RayConfig("Right", yaw_offset_deg=90, max_distance=10.0)
        reading = RayReading("Right", hit=True, distance=3.0, ratio=0.3, last_updated=1.0)
        wx, wz, _ = _ray_hit_world_xz(pose, cfg, reading)
        assert wx == pytest.approx(13.0, abs=1e-6)
        assert wz == pytest.approx(20.0, abs=1e-6)

    def test_avatar_rotated_90_forward_goes_east(self):
        # yaw 90 = facing +X, forward ray should hit at +X
        from src.raycast import RayReading
        pose = WorldPose(0, 0, 0, 90, 0)
        cfg = RayConfig("Fwd", yaw_offset_deg=0, max_distance=10.0)
        reading = RayReading("Fwd", hit=True, distance=4.0, ratio=0.4, last_updated=1.0)
        wx, wz, _ = _ray_hit_world_xz(pose, cfg, reading)
        assert wx == pytest.approx(4.0, abs=1e-6)
        assert wz == pytest.approx(0.0, abs=1e-6)

    def test_no_hit_uses_max_distance(self):
        from src.raycast import RayReading
        pose = WorldPose(0, 0, 0, 0, 0)
        cfg = RayConfig("Fwd", yaw_offset_deg=0, max_distance=15.0)
        reading = RayReading("Fwd", hit=False, distance=0.0, ratio=1.0, last_updated=1.0)
        wx, wz, dist = _ray_hit_world_xz(pose, cfg, reading)
        assert dist == pytest.approx(15.0)
        assert wz == pytest.approx(15.0, abs=1e-6)

    def test_pitch_shortens_horizontal_range(self):
        # 45 deg downward pitch -> horizontal = dist * cos(45)
        from src.raycast import RayReading
        pose = WorldPose(0, 0, 0, 0, 0)
        cfg = RayConfig("DropFwd", yaw_offset_deg=0, pitch_deg=-45, max_distance=10.0)
        reading = RayReading("DropFwd", hit=True, distance=10.0, ratio=1.0, last_updated=1.0)
        wx, wz, _ = _ray_hit_world_xz(pose, cfg, reading)
        assert wz == pytest.approx(10.0 * math.cos(math.radians(-45)), abs=1e-6)


# --- occupancy grid -----------------------------------------------------

class TestOccupancyGrid:
    def test_mark_and_query(self):
        grid = OccupancyGrid(cell_size_meters=0.5)
        grid.mark_obstacle(1.2, 3.4, weight=3)
        assert grid.state(1.2, 3.4) == CELL_OBSTACLE
        assert grid.is_blocked(1.2, 3.4, min_confidence=2) is True
        assert grid.is_blocked(1.2, 3.4, min_confidence=5) is False
        assert grid.state(50.0, 50.0) == CELL_UNKNOWN

    def test_free_overrides_single_obstacle(self):
        grid = OccupancyGrid()
        grid.mark_obstacle(0, 0)
        grid.mark_free(0, 0)
        grid.mark_free(0, 0)
        assert grid.state(0, 0) == CELL_FREE
        assert grid.is_blocked(0, 0) is False

    def test_cells_share_per_cell_bucket(self):
        # both points fall in the same 0.5m cell so cell_count is 1
        grid = OccupancyGrid(cell_size_meters=0.5)
        grid.mark_obstacle(0.1, 0.1)
        grid.mark_obstacle(0.4, 0.3)
        assert grid.cell_count() == 1

    def test_save_load_round_trip(self, tmp_path):
        grid = OccupancyGrid(cell_size_meters=0.5)
        grid.mark_obstacle(1.0, 2.0)
        grid.mark_free(3.0, 4.0)
        path = tmp_path / "world.json"
        grid.save(path)
        loaded = OccupancyGrid.load(path)
        assert loaded.cell_size == pytest.approx(0.5)
        assert loaded.state(1.0, 2.0) == CELL_OBSTACLE
        assert loaded.state(3.0, 4.0) == CELL_FREE


# --- mapper integration -------------------------------------------------

class _FakePoseReader:
    def __init__(self, pose: WorldPose | None):
        self._pose = pose
    def get(self) -> WorldPose | None:
        return self._pose


class TestSpatialMapperTick:
    def test_tick_marks_obstacle_at_ray_hit(self):
        rays = RaycastState()
        rays.update("Fwd_Hit", 1)
        rays.update("Fwd_Distance", 3.0)
        pose = WorldPose(x=0, y=0, z=0, yaw=0, timestamp=0)
        reader = _FakePoseReader(pose)
        mapper = SpatialMapper(
            raycast_state=rays,
            pose_reader=reader,  # type: ignore[arg-type]
            grid=OccupancyGrid(cell_size_meters=0.5),
        )
        mapper.configure_rays([RayConfig("Fwd", yaw_offset_deg=0, max_distance=10)])
        processed = mapper.tick_once()
        assert processed == 1
        # expected hit at (0, 3.0)
        assert mapper.grid.state(0.0, 3.0) == CELL_OBSTACLE
        # cells along the ray should be marked free
        assert mapper.grid.state(0.0, 1.0) == CELL_FREE

    def test_tick_does_nothing_without_pose(self):
        rays = RaycastState()
        rays.update("Fwd_Hit", 1)
        rays.update("Fwd_Distance", 2.0)
        mapper = SpatialMapper(
            raycast_state=rays,
            pose_reader=_FakePoseReader(None),  # type: ignore[arg-type]
        )
        mapper.configure_rays([RayConfig("Fwd")])
        assert mapper.tick_once() == 0
