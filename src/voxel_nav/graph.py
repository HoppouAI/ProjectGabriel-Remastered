"""Thread safe voxel `Graph` keyed by `Serial`."""

from __future__ import annotations

import math
import threading
from typing import Optional

from .coords import CELL_SIZE, Node, NodeType, Serial, serial_to_center, world_to_serial


class Graph:
    """Concurrent dict of Serial -> Node, scoped to a single VRChat world."""

    def __init__(self):
        self._nodes: dict[Serial, Node] = {}
        self._lock = threading.RLock()

    @property
    def nodes(self) -> dict[Serial, Node]:
        return self._nodes

    def add_node(self, node: Node) -> None:
        with self._lock:
            self._nodes[node.serial] = node

    def remove_node(self, serial: Serial) -> None:
        with self._lock:
            self._nodes.pop(serial, None)

    def find_node(self, x: float, y: float, z: float) -> Optional[Node]:
        s = world_to_serial(x, y, z)
        with self._lock:
            return self._nodes.get(s)

    def get(self, serial: Serial) -> Optional[Node]:
        with self._lock:
            return self._nodes.get(serial)

    def __contains__(self, serial: Serial) -> bool:
        with self._lock:
            return serial in self._nodes

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)

    def find_closest(self, x: float, y: float, z: float,
                     only_reachable: bool = True,
                     max_distance: float | None = None) -> Optional[Node]:
        """Nearest reachable node by squared distance to voxel center.
        If max_distance is given (meters), nodes farther than that are
        rejected and None is returned. Useful for snapping pathfind
        endpoints so a stale waypoint in an unmapped area doesnt silently
        snap to some random cell on the other side of the graph."""
        best: Optional[Node] = None
        best_d = math.inf
        limit_sq = math.inf if max_distance is None else max_distance * max_distance
        with self._lock:
            for node in self._nodes.values():
                if only_reachable and node.node_type != NodeType.REACHABLE:
                    continue
                cx, cy, cz = serial_to_center(node.serial)
                dx = cx - x; dy = cy - y; dz = cz - z
                d = dx*dx + dy*dy + dz*dz
                if d < best_d and d <= limit_sq:
                    best_d = d
                    best = node
        return best

    def find_nearest_reachable(self, x: float, y: float, z: float,
                                max_distance: float, k: int = 12,
                                ) -> list[Node]:
        """Return up to k nearest REACHABLE nodes within max_distance,
        sorted by distance. Used as fallback start candidates when the
        closest cell to the player turns out to be in an isolated little
        island that cant reach the actual goal."""
        limit_sq = max_distance * max_distance
        cands: list[tuple[float, Node]] = []
        with self._lock:
            for node in self._nodes.values():
                if node.node_type != NodeType.REACHABLE:
                    continue
                cx, cy, cz = serial_to_center(node.serial)
                dx = cx - x; dy = cy - y; dz = cz - z
                d = dx*dx + dy*dy + dz*dz
                if d <= limit_sq:
                    cands.append((d, node))
        cands.sort(key=lambda t: t[0])
        return [n for _, n in cands[:k]]

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "version": 1,
                "cell_size": CELL_SIZE,
                "nodes": [
                    {"s": list(n.serial), "t": int(n.node_type),
                     "l": n.label} for n in self._nodes.values()
                ],
            }

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        g = cls()
        for entry in data.get("nodes", ()):
            s = tuple(entry["s"])  # type: ignore[assignment]
            g._nodes[s] = Node(
                serial=s,
                node_type=NodeType(int(entry.get("t", 0))),
                label=entry.get("l", ""),
            )
        return g
