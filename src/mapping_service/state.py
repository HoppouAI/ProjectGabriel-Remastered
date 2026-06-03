"""Read-only state queries for the WebUI / control server."""

from __future__ import annotations

import logging

from src.voxel_nav import NodeType

logger = logging.getLogger(__name__)


class StateMixin:
    """get_state / get_world_cells / follow_status -- all WebUI polling."""

    def get_state(self) -> dict:
        with self._lock:
            reach = wall = iffy = 0
            try:
                with self._nav.graph._lock:  # noqa: SLF001
                    for n in self._nav.graph.nodes.values():
                        if n.node_type == NodeType.REACHABLE:
                            reach += 1
                        elif n.node_type == NodeType.UNREACHABLE:
                            wall += 1
                        else:
                            iffy += 1
            except Exception:
                pass

            pose = None
            if self._last_pose is not None:
                p = self._last_pose
                pose = {"x": p.x, "y": p.y, "z": p.z, "yaw": p.yaw}

            target = None
            action = "idle"
            if self._explorer is not None:
                st = self._explorer.state
                if st.target is not None:
                    target = list(st.target)
                action = st.action

            stats = {}
            if self._reader is not None:
                try:
                    stats = self._reader.stats()
                except Exception:
                    pass

            return {
                "running": self._running,
                "explore": self._explore_enabled,
                "manual": self._manual_mapping,
                "world": self._world_id,
                "world_name": self._world_name or self._resolve_world_name(),
                "pose": pose,
                "target": target,
                "action": action,
                "counts": {"reach": reach, "wall": wall, "iffy": iffy,
                           "total": reach + wall + iffy},
                "decode_rate": stats.get("decode_rate", 0.0),
                "last_error": self._last_error,
                "settings": {
                    "tick_hz": self._tick_hz,
                    "force_run": self._force_run,
                    "manual_wall_distance": self.manual_wall_distance,
                    "manual_wall_ratio": self.manual_wall_ratio,
                },
                "follow": self.follow_status(),
            }

    def get_world_cells(self) -> dict:
        """Return all cells split by type. Heavy -- caller should poll
        slowly. Each cell is [sx, sy, sz].

        The payload is cached and only rebuilt when the graph revision (or
        the active world) actually changes, so idle polling over a 100k+
        cell map is basically free instead of re-iterating every node and
        re-encoding a multi-megabyte response several times a second."""
        try:
            rev = self._nav.graph.revision
        except Exception:
            rev = -1
        world = self._world_id
        cache = getattr(self, "_world_cells_cache", None)
        if (cache is not None and cache[0] == world and cache[1] == rev):
            return cache[2]

        reach: list[list[int]] = []
        wall: list[list[int]] = []
        iffy: list[list[int]] = []
        try:
            with self._nav.graph._lock:  # noqa: SLF001
                for serial, node in self._nav.graph.nodes.items():
                    item = [serial[0], serial[1], serial[2]]
                    if node.node_type == NodeType.REACHABLE:
                        reach.append(item)
                    elif node.node_type == NodeType.UNREACHABLE:
                        wall.append(item)
                    else:
                        iffy.append(item)
        except Exception:
            logger.exception("mapping: get_world_cells failed")
        payload = {"world": world, "rev": rev, "reach": reach,
                   "wall": wall, "iffy": iffy}
        self._world_cells_cache = (world, rev, payload)
        return payload

    def follow_status(self) -> dict:
        if self._explorer is None:
            return {"active": False, "remaining": 0, "label": ""}
        try:
            return self._explorer.follow_status
        except Exception:
            return {"active": False, "remaining": 0, "label": ""}
