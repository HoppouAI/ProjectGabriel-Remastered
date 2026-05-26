"""A* preview + drive-to-waypoint follow + post-arrival yaw alignment."""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from src.voxel_explorer import VoxelExplorer
from src.voxel_nav import VoxelPathResult, find_path_astar, serial_to_center

logger = logging.getLogger(__name__)


class NavigationMixin:
    """pathfind_to (preview) + goto_* (active follow) + alignment."""

    def pathfind_to(self, gx: float, gy: float, gz: float) -> dict:
        """A* preview from current pose to goal world coords. Snaps both
        endpoints onto the nearest Reachable cell, but only if one is
        actually within snap range -- otherwise we'd silently pick some
        random cell across the map and 'pathfind' to there.
        If the closest start cell cant reach the goal (player standing on
        an isolated little island of reachable cells, or technically off
        the green grid), we fall back to the next-nearest reachable cells
        within 8m and try those too."""
        if self._last_pose is None:
            return {"found": False, "reason": "no current pose"}
        # gather up to 12 candidate start cells within 8m so we have some
        # backups if the literal closest one is stranded.
        starts = self._nav.graph.find_nearest_reachable(
            self._last_pose.x, self._last_pose.y, self._last_pose.z,
            max_distance=8.0, k=12,
        )
        if not starts:
            return {"found": False,
                    "reason": "your current position isnt on the map yet, "
                              "walk around to map this area first"}
        goal_node = self._nav.graph.find_closest(gx, gy, gz,
                                                  max_distance=2.5)
        if goal_node is None:
            return {"found": False,
                    "reason": "no mapped reachable cell near the goal -- "
                              "the waypoint is in an unmapped area"}
        # try each start in order. first one that yields a path wins. this
        # is cheap because A* short-circuits on the empty open set when
        # the start cant reach the goal.
        chosen_start = None
        result: VoxelPathResult | None = None
        for cand in starts:
            r = find_path_astar(self._nav.graph, cand.serial, goal_node.serial)
            if r.found:
                chosen_start = cand
                result = r
                break
        if result is None or chosen_start is None:
            return {"found": False, "reason": "no path"}
        return {
            "found": True,
            "start": list(chosen_start.serial),
            "goal": list(goal_node.serial),
            "full": [list(s) for s in result.full_serials],
            "filtered": [list(s) for s in result.serials],
            "cost": result.cost,
            "expanded": result.nodes_expanded,
            "start_snap_distance": math.sqrt(
                (serial_to_center(chosen_start.serial)[0] - self._last_pose.x) ** 2
                + (serial_to_center(chosen_start.serial)[2] - self._last_pose.z) ** 2
            ),
        }

    def pathfind_to_waypoint(self, name: str) -> dict:
        self._ensure_waypoints(self._world_id)
        wp = self._waypoints.get(name)
        if wp is None:
            return {"found": False, "reason": f"waypoint '{name}' not found"}
        return self.pathfind_to(wp.x, wp.y, wp.z)

    def _ensure_explorer_for_follow(self) -> None:
        """Make sure an explorer exists for follow mode, but DO NOT flip
        the public explore_enabled flag, so frontier-exploration stays off.
        We deliberately leave the explorer un-started here so the tick loop
        cant pick a discovery target in the window between construction
        and the follow_path call. follow_path will start() it atomically
        together with seeding the queue. The follow_only flag is set by
        the caller AFTER follow_path has been called so the auto-teardown
        check cant fire in the window between create and seed."""
        if not self._running:
            raise RuntimeError("mapping not running")
        if self._explorer is None:
            self._explorer = VoxelExplorer(self._nav, self._osc,
                                            learning_mode=True)
            self._explorer.force_run = self._force_run
            self._explorer.speed_mode = self._speed_mode
            logger.info("mapping: explorer created for path-follow")

    def _autostart_for_nav(self, timeout: float = 4.0) -> str:
        """Auto-start the mapping service if its not running yet, so the AI
        can call gotoWaypoint / saveWaypoint without anyone having to click
        Start Mapping in the WebUI first. Returns empty string on success,
        or an error message describing what blew up."""
        if self._running and self._last_pose is not None:
            return ""
        if not self._running:
            logger.info("mapping: auto-starting for nav request")
            state = self.start(explore=False)
            if not state.get("running"):
                err = state.get("error") or self._last_error \
                    or "could not auto-start mapping"
                return err
        # tick thread populates _last_pose at ~20Hz once the reader has a
        # frame. wait briefly so callers dont get a stale "no current pose".
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._last_pose is not None:
                return ""
            time.sleep(0.05)
        return "no pose yet, is VRChat focused with the shader on?"

    def goto_xyz(self, gx: float, gy: float, gz: float,
                 *, label: str = "",
                 final_yaw_deg: Optional[float] = None) -> dict:
        """A* from current pose to (gx,gy,gz), then drive there via OSC.
        If final_yaw_deg is given, the explorer rotates to that heading
        after arrival before going inactive."""
        err = self._autostart_for_nav()
        if err:
            return {"found": False, "reason": err}
        with self._lock:
            preview = self.pathfind_to(gx, gy, gz)
            if not preview.get("found"):
                return preview
            self._ensure_explorer_for_follow()
            # build cell-serial list -- include both full + filtered path
            full = preview.get("full") or []
            cells: list[tuple[int, int, int]] = [tuple(s) for s in full]  # type: ignore
            try:
                self._explorer.follow_path(cells, label=label or "goto",
                                           final_yaw_deg=final_yaw_deg)
            except Exception as exc:
                logger.exception("mapping: follow_path failed")
                return {"found": False, "reason": f"follow failed: {exc}"}
            # only flag follow-only AFTER the queue is seeded, otherwise
            # the tick loop teardown could fire in the gap between create
            # and seed (follow_status.active is False until follow_path).
            if not self._explore_enabled:
                self._explorer_follow_only = True
            preview["driving"] = True
            preview["label"] = label or "goto"
            return preview

    def goto_waypoint(self, name: str) -> dict:
        # autostart up front so the world id (and thus the waypoint store)
        # points at the real world, not the default placeholder.
        err = self._autostart_for_nav()
        if err:
            return {"found": False, "reason": err}
        self._ensure_waypoints(self._world_id)
        wp = self._waypoints.get(name)
        if wp is None:
            return {"found": False, "reason": f"waypoint '{name}' not found"}
        # the saved facing rides through to the explorer so it can rotate
        # to the heading once it arrives, in the same control loop as the
        # walking (no race with explorer teardown).
        return self.goto_xyz(wp.x, wp.y, wp.z, label=f"wp:{name}",
                             final_yaw_deg=float(wp.yaw))

    def cancel_goto(self) -> dict:
        with self._lock:
            self._pending_align_yaw = None
            try:
                self._osc.client.send_message("/input/LookHorizontal", 0.0)
            except Exception:
                pass
            if self._explorer is not None:
                try:
                    self._explorer.cancel_follow()
                except Exception:
                    logger.exception("mapping: cancel_follow failed")
            return self.get_state()

    def _drive_yaw_alignment(self, pose_yaw: float) -> None:
        """One tick of proportional yaw alignment via OSC LookHorizontal.
        Stops once we're within ~2deg of the target."""
        target = self._pending_align_yaw
        if target is None:
            return
        # shortest signed delta in (-180, 180]
        delta = (target - pose_yaw + 540.0) % 360.0 - 180.0
        if abs(delta) <= 2.0:
            try:
                self._osc.client.send_message("/input/LookHorizontal", 0.0)
            except Exception:
                pass
            self._pending_align_yaw = None
            logger.info("mapping: yaw aligned to %.1fdeg", target)
            return
        sign = 1.0 if delta > 0 else -1.0
        # proportional magnitude: ~0.08 floor so we actually turn at small
        # deltas, ramp up to 0.5 around 30deg+. positive = right (+yaw).
        mag = min(0.5, max(0.08, abs(delta) / 60.0))
        try:
            self._osc.client.send_message("/input/LookHorizontal", sign * mag)
        except Exception:
            pass
