"""Raycast-guided seek fallback.

When the A* follow dead-ends (replan limit hit, or no mapped path exists
because chunks of the world arent voxelized yet) we'd normally cancel and
the avatar just spins in place re-asking for a route. Instead, seek mode
heads straight at the goal's world position and leans on the in-engine
VRCRaycast sensors to dodge whatever's in the way, mapping as it walks.
Once observe() has filled in enough cells for a real A* path to reappear,
it hands control back to the queue follower.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from src.voxel_nav import find_path_astar, serial_to_center, world_to_serial


logger = logging.getLogger(__name__)


class SeekMixin:
    """Reactive goto fallback. Relies on attributes from VoxelExplorer:
        self.nav, self.state, self.osc, self.speed_mode, self.force_run,
        self._final_yaw_deg, self._aligning, self._align_start_t,
        self._follow_active, self._path_queue, self._follow_goal,
        self._follow_replans, self._abandoned, plus the seek state vars and
        SEEK_* constants set in __init__. Calls _turn_toward,
        _apply_raycast_assist, _ra_forward_clearance, _send_osc.
    """

    def start_seek_to(self, gx: float, gy: float, gz: float, *,
                      label: str = "",
                      final_yaw_deg: Optional[float] = None) -> bool:
        """Public entry: head straight at a world position on raycasts even
        when there's no A* path at all yet (initial plan failed, goal sits in
        unmapped space). Maps as it walks and flips to A* follow the moment a
        real path appears. Returns False if we cant seek (no fresh raycast
        data to steer with), so the caller can surface the original error."""
        with self._lock:
            # bail before we touch any state if we cant actually steer, so a
            # failed seek never leaves the explorer started and idle (which
            # would slide it into frontier discovery mode).
            if self._ra_forward_clearance() is None:
                return False
            if not self._active:
                self.start()
            goal_cell = world_to_serial(gx, gy, gz)
            now = time.time()
            self._follow_label = label or "seek"
            self._final_yaw_deg = final_yaw_deg
            self._aligning = False
            self._path_queue = []
            self._follow_active = False
            self._follow_goal = goal_cell
            self._follow_replans = 0
            self._seek_active = True
            self._seek_goal_world = (gx, gy, gz)
            self._seek_goal_cell = goal_cell
            self._seek_start_t = now
            self._seek_best_dist = math.inf
            self._seek_best_t = now
            self._seek_next_astar_t = now + self.SEEK_ASTAR_RETRY_S
            self._seek_engagements = 1
            s = self.state
            s.target = None
            s.target_source = None
            s.e_count = 0.0
            s.last_distance = math.inf
            s.last_cell = None
            s.last_progress_t = now
            s.action = "seek"
            self._ec_multiplier = 1.0
            logger.info("voxel_explorer: starting direct raycast seek to %s (%s)",
                        goal_cell, self._follow_label)
            return True

    def _begin_seek(self, why: str) -> bool:
        """Flip from A* follow into raycast seek toward the goal. Returns
        False if we cant seek (no goal, cap reached, or no fresh ray data),
        so the caller falls back to plain cancel."""
        goal = self._follow_goal
        if goal is None:
            return False
        if self._seek_engagements >= self.SEEK_MAX_ENGAGEMENTS:
            logger.warning("voxel_explorer: seek cap reached, giving up (%s)", why)
            return False
        # seek is blind navigation, it only works if the raycasts are feeding us
        if self._ra_forward_clearance() is None:
            logger.info("voxel_explorer: no raycast data, cant seek (%s)", why)
            return False
        gx, gy, gz = serial_to_center(goal)
        now = time.time()
        self._seek_active = True
        self._seek_goal_world = (gx, gy, gz)
        self._seek_goal_cell = goal
        self._seek_start_t = now
        self._seek_best_dist = math.inf
        self._seek_best_t = now
        self._seek_next_astar_t = now + self.SEEK_ASTAR_RETRY_S
        self._seek_engagements += 1
        # seek owns motion now, drop the queue machinery
        self._follow_active = False
        self.state.target = None
        self.state.target_source = None
        self.state.action = "seek"
        logger.info("voxel_explorer: seeking to %s via raycasts (%s), attempt %d",
                    goal, why, self._seek_engagements)
        return True

    def _seek_tick(self, pose_x: float, pose_y: float, pose_z: float,
                   fx: float, fz: float) -> None:
        s = self.state
        gx, gy, gz = self._seek_goal_world
        dx = gx - pose_x
        dz = gz - pose_z
        dist = math.hypot(dx, dz)
        now = time.time()

        if (dist <= self.SEEK_ARRIVE_RADIUS
                and abs(gy - pose_y) <= self.SEEK_ARRIVE_Y):
            logger.info("voxel_explorer: seek reached goal %s", self._seek_goal_cell)
            self._finish_seek(success=True)
            return

        # progress watchdog: only count real closing distance toward the goal
        if dist < self._seek_best_dist - self.SEEK_PROGRESS_EPS:
            self._seek_best_dist = dist
            self._seek_best_t = now
        elif now - self._seek_best_t > self.SEEK_NO_PROGRESS_S:
            logger.warning("voxel_explorer: seek stuck (%.0fs no progress) "
                           "%.1fm from goal, giving up",
                           self.SEEK_NO_PROGRESS_S, dist)
            self._finish_seek(success=False)
            return
        if now - self._seek_start_t > self.SEEK_MAX_S:
            logger.warning("voxel_explorer: seek timed out %.1fm from goal, "
                           "giving up", dist)
            self._finish_seek(success=False)
            return

        # mapping has been filling in as we walk. if a real path to the goal
        # exists now, hand back to the efficient queue follower.
        if now >= self._seek_next_astar_t:
            self._seek_next_astar_t = now + self.SEEK_ASTAR_RETRY_S
            if self._seek_try_reacquire():
                return

        if dist < 1e-6:
            ndx, ndz = fx, fz
        else:
            ndx, ndz = dx / dist, dz / dist
        turn, dot, _facing = self._turn_toward(fx, fz, ndx, ndz)
        forward = 0.0 if dot < 0.2 else max(0.0, min(dot, 1.0))

        turn, forward, clearance = self._apply_raycast_assist(turn, forward, dist)
        if clearance is None:
            logger.warning("voxel_explorer: seek lost raycast data, giving up")
            self._finish_seek(success=False)
            return

        mode = (self.speed_mode or "fast").lower()
        if mode in ("slow", "walk"):
            forward *= 0.5
            run = False
        elif mode == "normal":
            run = False
        elif mode in ("sprint", "run"):
            run = True
        else:
            run = dist >= 2.0 or bool(getattr(self, "force_run", False))
        if clearance < self.RA_RUN_MIN_CLEAR:
            run = False
        self._send_osc(forward, turn, run=run)
        s.action = f"seek {dist:.1f}m"

    def _seek_try_reacquire(self) -> bool:
        cur = self.nav.current
        goal = self._seek_goal_cell
        if cur is None or goal is None or goal not in self.nav.graph:
            return False
        now_m = time.monotonic()
        blocked = {c for c, exp in self._abandoned.items() if exp > now_m}
        pr = find_path_astar(self.nav.graph, cur.serial, goal, blacklist=blocked)
        path = None
        if pr.found:
            if len(pr.smoothed) > 1:
                path = list(pr.smoothed[1:])
            elif pr.serials:
                path = list(pr.serials)
            elif len(pr.full_serials) > 1:
                path = list(pr.full_serials[1:])
        if not path:
            return False
        self._seek_active = False
        self._path_queue = path
        self._follow_active = True
        self._follow_goal = goal
        self._follow_replans = 0
        self.state.target = None
        self.state.target_source = None
        self.state.last_progress_t = time.time()
        logger.info("voxel_explorer: seek reacquired A* path (%d cells) to %s, "
                    "resuming follow", len(path), goal)
        return True

    def _finish_seek(self, *, success: bool) -> None:
        self._seek_active = False
        self._follow_active = False
        self._path_queue.clear()
        self._follow_goal = None
        self._follow_replans = 0
        s = self.state
        if success and self._final_yaw_deg is not None:
            self._aligning = True
            self._align_start_t = time.time()
            s.action = "aligning"
            logger.info("voxel_explorer: seek done, aligning to %.1fdeg",
                        self._final_yaw_deg)
        else:
            if not success:
                self._final_yaw_deg = None
            s.action = "seek_done" if success else "seek_failed"
        self._send_osc(0.0, 0.0, run=False)
