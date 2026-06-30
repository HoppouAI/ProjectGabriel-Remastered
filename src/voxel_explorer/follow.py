"""Path-follow queue (drive-to-waypoint) primitives."""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from src.voxel_nav import Serial, serial_to_center


logger = logging.getLogger(__name__)


class FollowMixin:
    """Queue-driven path follow. Relies on attributes initialized by
    `VoxelExplorer.__init__`:
        self._lock, self._active, self.state, self._path_queue,
        self._follow_active, self._follow_label, self._follow_goal,
        self._follow_replans, self._final_yaw_deg, self._aligning,
        self._ec_multiplier, self.nav.
    Also calls self.start() and self._send_osc() from the main class.
    """

    def follow_path(self, serials: list[Serial], *, label: str = "",
                    final_yaw_deg: Optional[float] = None) -> None:
        """Drive along the given cell sequence. Replaces any current target.
        The explorer must be active; if not, start() is called first.
        If final_yaw_deg is given, the explorer rotates to that heading
        once the queue is empty before going inactive."""
        with self._lock:
            if not self._active:
                self.start()
            self._path_queue = list(serials)
            self._follow_active = True
            self._follow_label = label or ""
            self._final_yaw_deg = final_yaw_deg
            self._aligning = False
            # fresh trip, reset the seek fallback budget
            self._seek_active = False
            self._seek_engagements = 0
            # last cell of the queue is treated as the goal for replans
            self._follow_goal = serials[-1] if serials else None
            self._follow_replans = 0
            s = self.state
            s.target = None
            s.target_source = None
            s.e_count = 0.0
            s.last_distance = math.inf
            s.last_cell = None
            s.last_progress_t = time.time()
            self._ec_multiplier = 1.0
            logger.info("voxel_explorer: follow path label=%r len=%d final_yaw=%s",
                        self._follow_label, len(self._path_queue),
                        f"{final_yaw_deg:.1f}" if final_yaw_deg is not None else "none")

    def cancel_follow(self) -> None:
        with self._lock:
            if not self._follow_active and not self._aligning and not self._seek_active:
                return
            self._follow_active = False
            self._aligning = False
            self._seek_active = False
            self._final_yaw_deg = None
            self._path_queue.clear()
            self._follow_goal = None
            self._follow_replans = 0
            self.state.target = None
            self._send_osc(0.0, 0.0, run=False)
            self.state.action = "follow_cancel"
            logger.info("voxel_explorer: follow cancelled")

    @property
    def follow_status(self) -> dict:
        return {
            "active": self._follow_active or self._aligning or self._seek_active,
            "remaining": len(self._path_queue),
            "label": self._follow_label,
            "aligning": self._aligning,
            "seeking": self._seek_active,
        }

    def _follow_carrot(self, px: float, pz: float) -> tuple[float, float]:
        """Pure-pursuit lookahead point along [target] + queue, about
        FOLLOW_LOOKAHEAD meters ahead of the avatar. Steering at this carrot
        instead of the next cell center lets the follower arc through corners
        and flow between waypoints instead of stopping to pivot at each one.

        The carrot only extends through queued cells we still have voxel
        line-of-sight to from where we are, so it never aims past a wall
        corner and yanks the avatar into geometry."""
        s = self.state
        if s.target is None:
            return px, pz
        cur = self.nav.current
        from_cell = cur.serial if cur is not None else None
        graph = self.nav.graph
        # always head to the immediate target, then keep extending through
        # queued cells only while LOS holds from our current cell.
        chain: list[Serial] = [s.target]
        if from_cell is not None:
            for cell in self._path_queue:
                if not graph.has_line_of_sight(from_cell, cell):
                    break
                chain.append(cell)
        remaining = self.FOLLOW_LOOKAHEAD
        prev_x, prev_z = px, pz
        for cell in chain:
            wx, _wy, wz = serial_to_center(cell)
            seg = math.hypot(wx - prev_x, wz - prev_z)
            if seg >= remaining:
                if seg < 1e-6:
                    return wx, wz
                t = remaining / seg
                return prev_x + (wx - prev_x) * t, prev_z + (wz - prev_z) * t
            remaining -= seg
            prev_x, prev_z = wx, wz
        return prev_x, prev_z

    def _advance_follow_queue(self, current_serial: Serial) -> bool:
        """Pop cells off the follow queue until we find one we should actually
        drive toward. Skips cells we're already in (same XZ column).
        Sets state.target and returns True on success, False if the queue
        is exhausted.

        We used to also skip cells more than FOLLOW_MAX_CLIMB above us, but
        that ate entire upstairs/elevator paths. now if a cell is bogus
        we let _give_up_target catch it via eCount and replan instead."""
        s = self.state
        while self._path_queue:
            nxt = self._path_queue.pop(0)
            same_col = (current_serial[0] == nxt[0]
                        and current_serial[2] == nxt[2])
            if same_col or self.nav.bar_check(current_serial, nxt):
                continue
            s.target = nxt
            s.target_source = current_serial
            return True
        s.target = None
        s.target_source = None
        return False
