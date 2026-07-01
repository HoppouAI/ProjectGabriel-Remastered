"""Thread safe voxel `Graph` keyed by `Serial`."""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

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

# spatial index granularity: 8 cells per chunk axis = 2m chunks. nearest-node
# queries only touch chunks that intersect the search sphere instead of
# scanning all 100k+ nodes.
_CHUNK_SHIFT = 3
_CHUNK_M = CELL_SIZE * (1 << _CHUNK_SHIFT)


def _chunk_of(s: Serial) -> tuple[int, int, int]:
    return (s[0] >> _CHUNK_SHIFT, s[1] >> _CHUNK_SHIFT, s[2] >> _CHUNK_SHIFT)


def _build_neighbor_offsets() -> tuple[tuple[int, int, int, float, bool], ...]:
    out = []
    for dy in (-1, 0, 1):
        same_layer = dy == 0
        vert = abs(dy) * _VERTICAL_PENALTY
        ortho = (1.0 if same_layer else _SQRT2) + vert
        diag = (_SQRT2 if same_layer else _SQRT3) + vert
        for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            out.append((dx, dy, dz, ortho, False))
        for dx, dz in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            out.append((dx, dy, dz, diag, True))
    return tuple(out)


# (dx, dy, dz, cost, is_diagonal) x26, built once instead of per expansion.
_NEIGHBOR_OFFSETS = _build_neighbor_offsets()


class Graph:
    """Concurrent dict of Serial -> Node, scoped to a single VRChat world."""

    def __init__(self):
        self._nodes: dict[Serial, Node] = {}
        # chunk -> {serial: node} spatial index, kept in sync by add/remove.
        # in-place node_type flips dont need index updates since we store refs.
        self._chunks: dict[tuple[int, int, int], dict[Serial, Node]] = {}
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
            s = node.serial
            self._nodes[s] = node
            self._chunks.setdefault(_chunk_of(s), {})[s] = node
            self._rev += 1

    def remove_node(self, serial: Serial) -> None:
        with self._lock:
            if self._nodes.pop(serial, None) is not None:
                ck = _chunk_of(serial)
                bucket = self._chunks.get(ck)
                if bucket is not None:
                    bucket.pop(serial, None)
                    if not bucket:
                        del self._chunks[ck]
            self._rev += 1

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._chunks.clear()
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
    ) -> list[tuple[Serial, float]]:
        """26-connected neighbors that are REACHABLE, with A* edge costs.
        Acquires the graph lock exactly once for all 26 lookups so A* and
        BFS callers dont thrash the lock 1.3M times per pathfind. Returns a
        list (not a generator) so the lock is released before the caller
        does its per-neighbor work."""
        sx, sy, sz = serial
        out: list[tuple[Serial, float]] = []
        with self._lock:
            nodes = self._nodes
            get = nodes.get
            reachable = NodeType.REACHABLE
            for dx, dy, dz, cost, is_diag in _NEIGHBOR_OFFSETS:
                cand = (sx + dx, sy + dy, sz + dz)
                n = get(cand)
                if n is None or n.node_type != reachable:
                    continue
                if is_diag and self._corner_blocked(nodes, serial, cand):
                    continue
                out.append((cand, cost))
        return out

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

    def _nearby(self, x: float, y: float, z: float, max_distance: float,
                only_reachable: bool) -> list[tuple[float, Node]]:
        """All nodes within max_distance meters as (dist_sq, node), unsorted.
        Walks only the chunks that intersect the search sphere."""
        limit_sq = max_distance * max_distance
        qcx, qcy, qcz = _chunk_of(world_to_serial(x, y, z))
        r = int(max_distance / _CHUNK_M) + 1
        out: list[tuple[float, Node]] = []
        with self._lock:
            chunks = self._chunks
            reachable = NodeType.REACHABLE
            for cx in range(qcx - r, qcx + r + 1):
                for cy in range(qcy - r, qcy + r + 1):
                    for cz in range(qcz - r, qcz + r + 1):
                        bucket = chunks.get((cx, cy, cz))
                        if not bucket:
                            continue
                        for s, node in bucket.items():
                            if only_reachable and node.node_type != reachable:
                                continue
                            ccx, ccy, ccz = serial_to_center(s)
                            dx = ccx - x; dy = ccy - y; dz = ccz - z
                            d = dx*dx + dy*dy + dz*dz
                            if d <= limit_sq:
                                out.append((d, node))
        return out

    def find_closest(self, x: float, y: float, z: float,
                     only_reachable: bool = True,
                     max_distance: float | None = None) -> Optional[Node]:
        """Nearest reachable node by squared distance to voxel center.
        If max_distance is given (meters), nodes farther than that are
        rejected and None is returned. Useful for snapping pathfind
        endpoints so a stale waypoint in an unmapped area doesnt silently
        snap to some random cell on the other side of the graph."""
        if max_distance is not None:
            cands = self._nearby(x, y, z, max_distance, only_reachable)
            if not cands:
                return None
            return min(cands, key=lambda t: t[0])[1]
        # unbounded query has to scan everything, keep the legacy path.
        best: Optional[Node] = None
        best_d = math.inf
        with self._lock:
            for i, node in enumerate(self._nodes.values()):
                if only_reachable and node.node_type != NodeType.REACHABLE:
                    continue
                cx, cy, cz = serial_to_center(node.serial)
                dx = cx - x; dy = cy - y; dz = cz - z
                d = dx*dx + dy*dy + dz*dz
                if d < best_d:
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
        cands = self._nearby(x, y, z, max_distance, only_reachable=True)
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
            node = Node(
                serial=s,
                node_type=NodeType(int(entry.get("t", 0))),
                label=entry.get("l", ""),
            )
            g._nodes[s] = node
            g._chunks.setdefault(_chunk_of(s), {})[s] = node
        return g
