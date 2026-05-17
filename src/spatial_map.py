"""World mapping using raycast hits + decoded player pose.

Maintains a coarse 2D top-down occupancy grid in world coordinates. Each
tick we:

    1. Read the player's world pose (from `PoseExfilReader`)
    2. For each fresh raycast reading, compute the hit point in world space
    3. Mark the hit cell as 'obstacle', mark cells along the ray as 'free'

Over time this builds a map of the current world. Persisted per-world so
re-joining a place keeps prior knowledge.

This is intentionally simple -- a sparse dict-based grid, not a numpy
array, because most VRChat worlds are mostly empty and we want low memory
plus easy serialization. If you ever want pathfinding on dense urban
worlds, swap the storage backend later.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.raycast import RayReading, RaycastState
from src.pose_decoder import PoseExfilReader, WorldPose

logger = logging.getLogger(__name__)


# cell states
CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_OBSTACLE = 2


@dataclass
class RayConfig:
    """Configuration for a single raycast on the avatar, used to project the
    hit into world space from the player pose.

    `yaw_offset_deg` is the ray's heading relative to the player's forward
    facing (0 = straight ahead, 90 = right, -90 = left, 180 = behind).
    `pitch_deg` is the up/down angle (0 = horizontal, -90 = straight down).
    `max_distance` is the raycast's configured max range, used when no hit
    so we can still mark the swept cells as free."""
    name: str
    yaw_offset_deg: float = 0.0
    pitch_deg: float = 0.0
    max_distance: float = 10.0


class OccupancyGrid:
    """Sparse 2D occupancy grid keyed by integer cell coords.

    World XZ plane (Unity Y is up). Cell size in meters configurable.
    Obstacle counts let us be Bayesian-lite: a cell with many obstacle hits
    is more confidently blocked than one with a single stray hit."""

    def __init__(self, cell_size_meters: float = 0.25):
        self._cell_size = max(0.05, cell_size_meters)
        self._lock = threading.RLock()
        # cell -> (obstacle_count, free_count)
        self._cells: dict[tuple[int, int], tuple[int, int]] = {}

    @property
    def cell_size(self) -> float:
        return self._cell_size

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        return (int(math.floor(x / self._cell_size)),
                int(math.floor(z / self._cell_size)))

    def cell_to_world_center(self, cx: int, cz: int) -> tuple[float, float]:
        return ((cx + 0.5) * self._cell_size, (cz + 0.5) * self._cell_size)

    def mark_obstacle(self, x: float, z: float, weight: int = 1) -> None:
        key = self.world_to_cell(x, z)
        with self._lock:
            obs, free = self._cells.get(key, (0, 0))
            self._cells[key] = (obs + weight, free)

    def mark_free(self, x: float, z: float, weight: int = 1) -> None:
        key = self.world_to_cell(x, z)
        with self._lock:
            obs, free = self._cells.get(key, (0, 0))
            self._cells[key] = (obs, free + weight)

    def state(self, x: float, z: float) -> int:
        key = self.world_to_cell(x, z)
        with self._lock:
            entry = self._cells.get(key)
        if entry is None:
            return CELL_UNKNOWN
        obs, free = entry
        if obs == 0 and free == 0:
            return CELL_UNKNOWN
        if obs > free:
            return CELL_OBSTACLE
        return CELL_FREE

    def is_blocked(self, x: float, z: float, min_confidence: int = 2) -> bool:
        """A cell is blocked when its obstacle count beats its free count by
        at least `min_confidence` votes. Keeps single noisy hits from walling
        off the path."""
        key = self.world_to_cell(x, z)
        with self._lock:
            entry = self._cells.get(key)
        if entry is None:
            return False
        obs, free = entry
        return (obs - free) >= min_confidence

    def cell_count(self) -> int:
        with self._lock:
            return len(self._cells)

    def snapshot(self) -> dict[tuple[int, int], tuple[int, int]]:
        with self._lock:
            return dict(self._cells)

    # --- persistence (per-world) -------------------------------------------

    def save(self, path: Path) -> None:
        with self._lock:
            payload = {
                "cell_size_meters": self._cell_size,
                "cells": [
                    {"cx": cx, "cz": cz, "obs": obs, "free": free}
                    for (cx, cz), (obs, free) in self._cells.items()
                ],
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OccupancyGrid":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("spatial_map: failed to load %s: %s", path, e)
            return cls()
        grid = cls(cell_size_meters=float(data.get("cell_size_meters", 0.25)))
        with grid._lock:
            for entry in data.get("cells", []):
                key = (int(entry["cx"]), int(entry["cz"]))
                grid._cells[key] = (int(entry["obs"]), int(entry["free"]))
        return grid


# --- ray projection ------------------------------------------------------

def _ray_hit_world_xz(pose: WorldPose, ray_cfg: RayConfig, reading: RayReading
                      ) -> tuple[float, float, float] | None:
    """Project a raycast reading into world XZ + the effective distance used.

    Returns (world_x, world_z, distance_used) where distance_used is the ray's
    actual hit distance if it hit, otherwise its configured max range (so the
    caller can sweep the FREE cells along that path).
    """
    if reading.hit:
        dist = reading.distance
        if dist <= 0.0:
            return None
    else:
        dist = ray_cfg.max_distance

    # combine player yaw with ray's body-frame yaw offset to get world yaw
    # Unity yaw is measured clockwise from +Z when looking down at the XZ plane
    world_yaw_deg = pose.yaw + ray_cfg.yaw_offset_deg
    world_yaw_rad = math.radians(world_yaw_deg)
    pitch_rad = math.radians(ray_cfg.pitch_deg)
    horizontal = dist * math.cos(pitch_rad)

    # Unity forward at yaw 0 is +Z, +90 rotates toward +X
    dx = horizontal * math.sin(world_yaw_rad)
    dz = horizontal * math.cos(world_yaw_rad)

    return (pose.x + dx, pose.z + dz, dist)


class SpatialMapper:
    """Fuses pose + raycasts into the occupancy grid. Background thread, runs
    at the rate of the slowest input (usually the pose decoder)."""

    def __init__(
        self,
        *,
        raycast_state: RaycastState,
        pose_reader: PoseExfilReader,
        grid: OccupancyGrid | None = None,
        ray_configs: list[RayConfig] | None = None,
        tick_hz: float = 10.0,
        save_path: Path | None = None,
        save_every_seconds: float = 30.0,
    ):
        self._raycasts = raycast_state
        self._pose_reader = pose_reader
        self.grid = grid or OccupancyGrid()
        self._ray_configs = {c.name: c for c in (ray_configs or [])}
        self._tick_interval = 1.0 / max(0.5, tick_hz)
        self._save_path = save_path
        self._save_every = save_every_seconds
        self._last_save = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def configure_rays(self, ray_configs: list[RayConfig]) -> None:
        self._ray_configs = {c.name: c for c in ray_configs}
        # also declare them on the raycast state so consumers don't see None
        self._raycasts.declare_many(self._ray_configs.keys())

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="spatial-mapper"
        )
        self._thread.start()
        logger.info(
            "spatial_map: started, %d rays, %.1f Hz, cell=%.2fm",
            len(self._ray_configs), 1.0 / self._tick_interval,
            self.grid.cell_size,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._save_path is not None:
            try:
                self.grid.save(self._save_path)
            except Exception as e:
                logger.warning("spatial_map: final save failed: %s", e)

    def tick_once(self) -> int:
        """Run a single mapping tick. Returns number of rays processed. Useful
        for tests and for hand-driving from the main loop."""
        pose = self._pose_reader.get()
        if pose is None:
            return 0
        processed = 0
        for name, cfg in self._ray_configs.items():
            reading = self._raycasts.get(name)
            if reading is None or not reading.is_fresh(max_age_seconds=1.5):
                continue
            projected = _ray_hit_world_xz(pose, cfg, reading)
            if projected is None:
                continue
            wx, wz, dist = projected
            # sweep along the ray marking free cells, then mark the endpoint
            self._sweep_free(pose.x, pose.z, wx, wz, dist)
            if reading.hit:
                self.grid.mark_obstacle(wx, wz)
            processed += 1
        return processed

    def _sweep_free(self, x0: float, z0: float, x1: float, z1: float,
                    distance: float) -> None:
        """Mark cells along the segment from (x0,z0) to (x1,z1) as free.
        Stops just short of the endpoint to avoid stealing votes from the
        obstacle cell."""
        cell_size = self.grid.cell_size
        # step in half-cell increments for decent coverage
        step = cell_size * 0.5
        if distance <= step:
            return
        steps = int(distance / step)
        if steps <= 1:
            return
        dx = (x1 - x0) / steps
        dz = (z1 - z0) / steps
        # leave the last step alone so the obstacle vote isn't undercut
        for i in range(1, steps - 1):
            self.grid.mark_free(x0 + dx * i, z0 + dz * i)

    def _run(self) -> None:
        while not self._stop.is_set():
            tick_start = time.monotonic()
            try:
                self.tick_once()
            except Exception as e:
                logger.warning("spatial_map: tick error: %s", e)

            # periodic save
            if (self._save_path is not None
                    and tick_start - self._last_save >= self._save_every):
                try:
                    self.grid.save(self._save_path)
                    self._last_save = tick_start
                except Exception as e:
                    logger.warning("spatial_map: save failed: %s", e)

            elapsed = time.monotonic() - tick_start
            sleep_for = self._tick_interval - elapsed
            if sleep_for > 0:
                self._stop.wait(timeout=sleep_for)
