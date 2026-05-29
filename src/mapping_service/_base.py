"""Shared base class + small helpers for the MappingService package."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.pose_decoder import PoseExfilReader, WorldPose
from src.voxel_explorer import VoxelExplorer
from src.voxel_nav import VoxelNavManager
from src.waypoints import WaypointStore

logger = logging.getLogger(__name__)


@dataclass
class _RegionGuess:
    monitor_index: int
    abs_x: int
    abs_y: int
    cell: int


class MappingServiceBase:
    """Holds the __init__ + the cheap world-id resolution helpers.

    All the heavy mixins (lifecycle / state / nav / etc) sit on top of
    this and just touch the attributes set up here.
    """

    DEFAULT_WORLD = "default"

    def __init__(self, osc, *, instance_monitor=None,
                 data_dir: str | Path = "data/voxel_nav"):
        self._osc = osc
        self._instance_monitor = instance_monitor
        self._lock = threading.RLock()

        self._nav = VoxelNavManager(data_dir=data_dir, learning_mode=True)
        self._reader: Optional[PoseExfilReader] = None
        self._explorer: Optional[VoxelExplorer] = None
        self._waypoints: Optional[WaypointStore] = None

        self._world_id: str = self.DEFAULT_WORLD
        self._world_name: str = ""
        self._running = False
        self._explore_enabled = False
        # True when the explorer was spun up only for a path-follow (goto)
        # and should be torn down once the follow completes, rather than
        # falling through into frontier discovery.
        self._explorer_follow_only = False
        # if set (degrees), once the current follow finishes we rotate the
        # avatar to match this yaw and then sit still. used so waypoint
        # gotos land you facing the same way you were when you saved.
        self._pending_align_yaw: Optional[float] = None
        # manual mapping: user drives the avatar themselves, we just label
        # cells they walk through as Reachable (via nav.observe) and use the
        # forward raycast to flag the cell directly in front as a wall when
        # we get a near-zero reading. great for fast first-pass mapping.
        self._manual_mapping = False
        self._manual_wall_throttle: dict[tuple[int, int, int], float] = {}
        self._manual_debug_last: float = 0.0
        # tunables for the raycast wall trip (also exposed via API)
        self.manual_wall_distance = 0.35   # m -- at or below this counts as wall
        self.manual_wall_ratio = 0.07      # ratio fallback for short rays
        self.manual_ray_name = "Fwd"        # which named ray to read
        # while manual mapping is on, hard-lock the avatar's yaw to the
        # nearest cardinal (0/90/180/270) and strafe-correct so they walk
        # down the center of a single voxel row. user just pushes forward.
        self.manual_grid_snap = True
        self._last_error: str = ""
        self._region: Optional[_RegionGuess] = None
        self._last_pose: Optional[WorldPose] = None
        self._last_pose_t: float = 0.0
        self._tick_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # tunables exposed via the UI / api
        self._tick_hz: float = 20.0          # pose sample rate
        self._force_run: bool = False        # always sprint while exploring
        # default move speed for follow + explore. proxies through to the
        # explorer's speed_mode any time we (re)create one.
        self._speed_mode: str = "fast"

        # ensure waypoint store at least exists for the default world so the
        # UI can list/add even when mapping hasnt been started yet.
        self._ensure_waypoints(self._world_id)

    # ------------------------------------------------------------------
    # world id
    # ------------------------------------------------------------------
    def _resolve_world_id(self) -> str:
        if self._instance_monitor is not None:
            try:
                wid = getattr(self._instance_monitor, "world_id", "")
                if wid:
                    return wid
                # legacy: current_location is "world:instance", strip instance
                loc = getattr(self._instance_monitor, "current_location", "")
                if loc and ":" in loc:
                    return loc.split(":", 1)[0]
                if loc:
                    return loc
            except Exception:
                pass
        return self.DEFAULT_WORLD

    def _resolve_world_name(self) -> str:
        if self._instance_monitor is not None:
            try:
                return getattr(self._instance_monitor, "world_name", "") or ""
            except Exception:
                pass
        return ""

    def _ensure_waypoints(self, world_id: str) -> None:
        if self._waypoints is None or self._waypoints.world_id != world_id:
            self._waypoints = WaypointStore(world_id)
