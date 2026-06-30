"""Thread safe voxel `Graph` keyed by `Serial`."""

from __future__ import annotations

import math
import threading
import time
from typing import Iterable, Optional

from .coords import (
    CELL_SIZE,
    Node,
    NodeType,
    Serial,
    _SQRT2,
    _SQRT3,
    _VERTICAL_PENALTY,
    serial_to_center,
    world_to_serial,
)

# how often to yield the GIL during O(n) scans, in node count.
_GIL_YIELD_EVERY = 2000


class Graph:
    """Concurrent dict of Serial -> Node, scoped to a single VRChat world."""

    def __init__(self):
        self._nodes: dict[Serial, Node] = {}
        self._lock = threading.RLock()
        # monotonic change counter, bumped on any mutation. lets the WebUI
        # state layer cache the heavy world-cells payload and skip rebuilding
        # it (and re-iterating 100k+ nodes) when nothing actually changed.
        self._rev = 0

    @property
    def nodes(self) -> dict[Serial, Node]:
        return self._nodes

    @property
    def revision(self) -> int:
        return self._rev

    def bump(self) -> None:
        """Mark the graph as changed without adding/removing a node, e.g.
        when a node's type is flipped in place."""
        with self._lock:
            self._rev += 1

    def add_node(self, node: Node) -> None:
        with self._lock:
            self._nodes[node.serial] = node
            self._rev += 1

    def remove_node(self, serial: Serial) -> None:
        with self._lock:
            self._nodes.pop(serial, None)
            self._rev += 1

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

    # ------------------------------------------------------------------
    # batch pathable neighbor lookup (single lock for 26 candidates)
    # ------------------------------------------------------------------
    def get_pathable_neighbors(
        self, serial: Serial,
    ) -> Iterable[tuple[Serial, float]]:
        """26-connected neighbors that are REACHABLE, with A* edge costs.
        Acquires the graph lock exactly once for all 26 lookups so A* and
        BFS callers dont thrash the lock 1.3M times per pathfind."""
        sx, sy, sz = serial
        # collect every candidate serial first so we can batch-lookup under
        # one lock. the cost is the same regardless of whether the cell
        # exists so we precompute costs too.
        items: list[tuple[Serial, float, bool]] = []
        for dy in (-1, 0, 1):
            ny = sy + dy
            same_layer = dy == 0
            ortho_cost = 1.0 if same_layer else _SQRT2
            for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                items.append(((sx + dx, ny, sz + dz),
                              ortho_cost + abs(dy) * _VERTICAL_PENALTY, False))
            diag_cost = _SQRT2 if same_layer else _SQRT3
            for dx, dz in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                items.append(((sx + dx, ny, sz + dz),
                              diag_cost + abs(dy) * _VERTICAL_PENALTY, True))
        with self._lock:
            nodes = self._nodes
            for cand, cost, is_diag in items:
                n = nodes.get(cand)
                if n is None or n.node_type != NodeType.REACHABLE:
                    continue
                if is_diag and self._corner_blocked(nodes, serial, cand):
                    continue
                yield cand, cost

    @staticmethod
    def _corner_blocked(
        nodes: dict[Serial, Node], a: Serial, b: Serial,
    ) -> bool:
        """True only if the diagonal step a->b would cut between two confirmed
        wall corners. Both endpoints are already known REACHABLE, so we only
        refuse the move when BOTH orthogonal corner cells are explicit
        UnReachable walls. Unmapped (missing) or Iffy corners dont block it:
        the footstep map is often just one cell wide and the avatar provably
        walked these diagonals when it laid the trail down, so treating
        unmapped corners as walls disconnected every thin diagonal path."""
        dx = b[0] - a[0]
        dz = b[2] - a[2]
        o1 = nodes.get((a[0] + dx, a[1], a[2]))
        if o1 is None or o1.node_type != NodeType.UNREACHABLE:
            return False
        o2 = nodes.get((a[0], a[1], a[2] + dz))
        if o2 is None or o2.node_type != NodeType.UNREACHABLE:
            return False
        return True

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
            for i, node in enumerate(self._nodes.values()):
                if only_reachable and node.node_type != NodeType.REACHABLE:
                    continue
                cx, cy, cz = serial_to_center(node.serial)
                dx = cx - x; dy = cy - y; dz = cz - z
                d = dx*dx + dy*dy + dz*dz
                if d < best_d and d <= limit_sq:
                    best_d = d
                    best = node
                if i % _GIL_YIELD_EVERY == 0:
                    time.sleep(0)
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
            for i, node in enumerate(self._nodes.values()):
                if node.node_type != NodeType.REACHABLE:
                    continue
                cx, cy, cz = serial_to_center(node.serial)
                dx = cx - x; dy = cy - y; dz = cz - z
                d = dx*dx + dy*dy + dz*dz
                if d <= limit_sq:
                    cands.append((d, node))
                if i % _GIL_YIELD_EVERY == 0:
                    time.sleep(0)
        cands.sort(key=lambda t: t[0])
        return [n for _, n in cands[:k]]

    def has_line_of_sight(self, a: Serial, b: Serial) -> bool:
        """True if every voxel the straight segment a->b passes through is a
        REACHABLE node. Conservative: unmapped or wall cells block sight, so a
        shortcut never cuts across geometry we havent confirmed walkable. 3D
        amanatides-woo voxel walk, acquires the lock once for the whole trace."""
        if a == b:
            return True
        ax, ay, az = a
        bx, by, bz = b
        sx = bx - ax
        sy = by - ay
        sz = bz - az
        stepx = 1 if sx > 0 else (-1 if sx < 0 else 0)
        stepy = 1 if sy > 0 else (-1 if sy < 0 else 0)
        stepz = 1 if sz > 0 else (-1 if sz < 0 else 0)
        inf = math.inf
        tmax_x = ((ax + (1 if stepx > 0 else 0)) - (ax + 0.5)) / sx if sx != 0 else inf
        tmax_y = ((ay + (1 if stepy > 0 else 0)) - (ay + 0.5)) / sy if sy != 0 else inf
        tmax_z = ((az + (1 if stepz > 0 else 0)) - (az + 0.5)) / sz if sz != 0 else inf
        tdx = abs(1.0 / sx) if sx != 0 else inf
        tdy = abs(1.0 / sy) if sy != 0 else inf
        tdz = abs(1.0 / sz) if sz != 0 else inf
        cx, cy, cz = ax, ay, az
        guard = abs(sx) + abs(sy) + abs(sz) + 4
        with self._lock:
            nodes = self._nodes
            while True:
                n = nodes.get((cx, cy, cz))
                if n is None or n.node_type != NodeType.REACHABLE:
                    return False
                if (cx, cy, cz) == (bx, by, bz):
                    return True
                if tmax_x <= tmax_y and tmax_x <= tmax_z:
                    cx += stepx
                    tmax_x += tdx
                elif tmax_y <= tmax_z:
                    cy += stepy
                    tmax_y += tdy
                else:
                    cz += stepz
                    tmax_z += tdz
                guard -= 1
                if guard < 0:
                    return False

    def to_dict(self) -> dict:
        with self._lock:
            nodes_out = []
            for i, n in enumerate(self._nodes.values()):
                nodes_out.append({
                    "s": list(n.serial), "t": int(n.node_type),
                    "l": n.label,
                })
                if i % _GIL_YIELD_EVERY == 0:
                    time.sleep(0)
            return {
                "version": 1,
                "cell_size": CELL_SIZE,
                "nodes": nodes_out,
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
