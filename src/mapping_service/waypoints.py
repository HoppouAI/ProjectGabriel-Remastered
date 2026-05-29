"""Saved waypoint CRUD."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WaypointsMixin:
    """list / add / remove waypoints for the current world."""

    def list_waypoints(self) -> list[dict]:
        # auto-start so we resolve the actual current world id, otherwise
        # we'd list waypoints from whatever world was last loaded (or the
        # default) which gives the AI an empty list and it tells the user
        # there are no saved spots even when there are.
        self._autostart_for_nav()
        self._ensure_waypoints(self._world_id)
        with self._lock:
            return [w.to_dict() for w in self._waypoints.list()]

    def add_waypoint(self, name: str, note: str = "") -> dict:
        if not name or not name.strip():
            raise ValueError("waypoint name required")
        err = self._autostart_for_nav()
        if err:
            raise RuntimeError(err)
        if self._last_pose is None:
            raise RuntimeError("no current pose -- start mapping first")
        self._ensure_waypoints(self._world_id)
        p = self._last_pose
        wp = self._waypoints.add(
            name.strip(), p.x, p.z, y=p.y, yaw=p.yaw, note=note,
        )
        logger.info("mapping: added waypoint '%s' at (%.2f, %.2f, %.2f)",
                    wp.name, wp.x, wp.y, wp.z)
        return wp.to_dict()

    def remove_waypoint(self, name: str) -> bool:
        self._ensure_waypoints(self._world_id)
        ok = self._waypoints.remove(name)
        if ok:
            logger.info("mapping: removed waypoint '%s'", name)
        return ok
