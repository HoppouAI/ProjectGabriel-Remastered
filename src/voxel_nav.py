"""3D voxel navigation, ported from the reference NodeManager + Pathfinding.

A walk-trail learning system. Whenever the avatar is grounded and we have
a fresh world pose, we mark that voxel as Reachable. Over time the graph
fills with the floor space the AI has actually traversed, including
multiple floors (Y axis). Pathfinding is 26-connected A* across that
graph with diagonal corner clipping prevention.

Cell size is 0.25 m to match exactly. Graphs persist per VRChat world
id to data/voxel_nav/<world_id>.json.

Coordinates:
    serial = (floor(x*4), floor(y*4), floor(z*4))   # int voxel coords
    center = (sx*0.25+0.125, sy*0.25+0.125, sz*0.25+0.125)
    position(corner) = (sx*0.25, sy*0.25, sz*0.25)
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


CELL_SIZE = 0.25
HALF_CELL = 0.125
SCALE = 4.0  # 1.0 / CELL_SIZE

_SQRT2 = 1.4142135
_SQRT3 = 1.7320508
_VERTICAL_PENALTY = 0.4142


class NodeType(IntEnum):
    REACHABLE = 0
    UNREACHABLE = 1
    IFFY = 2


Serial = tuple[int, int, int]


def world_to_serial(x: float, y: float, z: float) -> Serial:
    return (
        int(math.floor(x * SCALE)),
        int(math.floor(y * SCALE)),
        int(math.floor(z * SCALE)),
    )


def serial_to_center(s: Serial) -> tuple[float, float, float]:
    return (s[0] * CELL_SIZE + HALF_CELL,
            s[1] * CELL_SIZE + HALF_CELL,
            s[2] * CELL_SIZE + HALF_CELL)


def serial_to_position(s: Serial) -> tuple[float, float, float]:
    return (s[0] * CELL_SIZE, s[1] * CELL_SIZE, s[2] * CELL_SIZE)


@dataclass
class Node:
    serial: Serial
    node_type: NodeType = NodeType.REACHABLE
    is_turn: bool = False
    label: str = ""


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


# ---------------------------------------------------------------------------
# Pathfinding (port of reference Pathfinding.cs)
# ---------------------------------------------------------------------------

def _is_pathable(graph: Graph, s: Serial) -> bool:
    n = graph.get(s)
    return n is not None and n.node_type == NodeType.REACHABLE


def _corner_unreachable(graph: Graph, a: Serial, b: Serial) -> bool:
    """A diagonal step from a->b is blocked if BOTH orthogonal neighbors
    that share the corner are not reachable. Prevents wall clipping."""
    dx = b[0] - a[0]
    dz = b[2] - a[2]
    if _is_pathable(graph, (a[0] + dx, a[1], a[2])):
        return False
    if _is_pathable(graph, (a[0], a[1], a[2] + dz)):
        return False
    return True


def _neighbors(graph: Graph, serial: Serial) -> Iterable[tuple[Serial, float]]:
    sx, sy, sz = serial
    for dy in (-1, 0, 1):
        ny = sy + dy
        same_layer = (dy == 0)
        # orthogonal XZ moves
        ortho_cost = 1.0 if same_layer else _SQRT2
        for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cand = (sx + dx, ny, sz + dz)
            if _is_pathable(graph, cand):
                yield cand, ortho_cost + abs(dy) * _VERTICAL_PENALTY
        # diagonal XZ moves
        diag_cost = _SQRT2 if same_layer else _SQRT3
        for dx, dz in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            cand = (sx + dx, ny, sz + dz)
            if not _is_pathable(graph, cand):
                continue
            if _corner_unreachable(graph, serial, cand):
                continue
            yield cand, diag_cost + abs(dy) * _VERTICAL_PENALTY


def _heuristic(a: Serial, b: Serial) -> float:
    # straight line distance in cell units (matches Vector3.Distance on
    # node positions since each axis is a uniform scale).
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def _filter_path(path: list[Serial]) -> list[Serial]:
    """Collapse straight runs, keep only turn points and the final goal.

    Direction comparison is done on XZ only (Y changes don't count as a
    "turn" because vertical moves are common on stairs)."""
    if not path or len(path) < 2:
        return path[:]
    out: list[Serial] = []
    for i in range(1, len(path)):
        is_turn = (i == len(path) - 1)
        if i + 1 < len(path):
            a = (path[i + 1][0] - path[i][0], path[i + 1][2] - path[i][2])
            b = (path[i][0] - path[i - 1][0], path[i][2] - path[i - 1][2])
            if a != b:
                is_turn = True
        if is_turn:
            out.append(path[i])
    return out


@dataclass
class VoxelPathResult:
    found: bool
    serials: list[Serial] = field(default_factory=list)        # filtered (turn points)
    full_serials: list[Serial] = field(default_factory=list)   # every cell
    cost: float = 0.0
    nodes_expanded: int = 0

    @property
    def world_waypoints(self) -> list[tuple[float, float, float]]:
        return [serial_to_center(s) for s in self.serials]


def find_path_astar(graph: Graph, start: Serial, goal: Serial,
                    max_nodes: int = 50_000) -> VoxelPathResult:
    """A* over the voxel graph. 26-connected, prevents corner clipping.

    Both `start` and `goal` must already exist in the graph as Reachable
    nodes. Use `Graph.find_closest()` first if you need to snap a free
    world position onto the trail.
    """
    if start not in graph or goal not in graph:
        return VoxelPathResult(found=False)
    if start == goal:
        return VoxelPathResult(found=True, serials=[goal], full_serials=[goal])

    open_set: list[tuple[float, int, Serial]] = []
    came_from: dict[Serial, Serial] = {}
    g_score: dict[Serial, float] = {start: 0.0}
    counter = 0
    heapq.heappush(open_set, (_heuristic(start, goal), counter, start))
    expanded = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            full: list[Serial] = [current]
            while current in came_from:
                current = came_from[current]
                full.append(current)
            full.reverse()
            filtered = _filter_path(full)
            return VoxelPathResult(
                found=True, serials=filtered, full_serials=full,
                cost=g_score[goal], nodes_expanded=expanded,
            )

        expanded += 1
        if expanded > max_nodes:
            logger.warning("voxel_nav: A* hit max_nodes=%d, aborting", max_nodes)
            break

        for neighbor, cost in _neighbors(graph, current):
            tentative = g_score[current] + cost
            if tentative < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                f = tentative + _heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, counter, neighbor))

    return VoxelPathResult(found=False, nodes_expanded=expanded)


# ---------------------------------------------------------------------------
# Per-world manager (learning + persistence)
# ---------------------------------------------------------------------------

class VoxelNavManager:
    """Holds the current world's graph, learns trail from pose updates,
    persists per world id."""

    def __init__(self, data_dir: Path | str = "data/voxel_nav",
                 learning_mode: bool = True):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.learning_mode = learning_mode
        self.graph: Graph = Graph()
        self._world_id: Optional[str] = None
        self._current: Optional[Node] = None
        self._previous: Optional[Node] = None
        self._dirty = False
        self._lock = threading.RLock()
        # jump filter: if a single observe lands a huge distance from where
        # we just were, treat it as a pose glitch and require one repeat
        # before we trust it. otherwise transient bad reads paint stray
        # floating cells out in the void.
        self._pending_cell: Optional[Serial] = None
        self._pending_count: int = 0
        # cells past this many voxels from current count as a teleport
        # and need confirmation. ~2m at the 0.25m grid.
        self._jump_threshold: int = 8

    # --- world lifecycle ---------------------------------------------------
    def load_world(self, world_id: str) -> None:
        with self._lock:
            if world_id == self._world_id:
                return
            self.flush()
            self._world_id = world_id
            self._current = None
            self._previous = None
            path = self._data_dir / f"{world_id}.json"
            if path.exists():
                try:
                    self.graph = Graph.from_dict(json.loads(path.read_text("utf-8")))
                    logger.info("voxel_nav: loaded %d nodes for world %s",
                                len(self.graph), world_id)
                except Exception as exc:
                    logger.warning("voxel_nav: load failed for %s: %s", world_id, exc)
                    self.graph = Graph()
            else:
                self.graph = Graph()

    def flush(self) -> None:
        with self._lock:
            if not self._dirty or self._world_id is None:
                return
            path = self._data_dir / f"{self._world_id}.json"
            try:
                path.write_text(json.dumps(self.graph.to_dict()), encoding="utf-8")
                self._dirty = False
            except Exception as exc:
                logger.warning("voxel_nav: save failed: %s", exc)

    # --- learning ----------------------------------------------------------
    def observe(self, x: float, y: float, z: float, grounded: bool = True,
                interpolate: bool = True) -> Node:
        """Record that the avatar occupies this voxel. Returns the Node.

        With `interpolate=True` (default), if the previous observation was
        more than 1 voxel away (e.g. you walked fast between polls), fills
        in every voxel along the straight line between them so the trail
        stays connected for pathfinding.
        """
        serial = world_to_serial(x, y, z)
        with self._lock:
            # teleport / glitch guard: if we jumped way too far in one tick,
            # demand the same cell show up again before we commit it. this
            # kills the random floating green cubes from pose decoder hiccups.
            if self._current is not None:
                dx = abs(serial[0] - self._current.serial[0])
                dy = abs(serial[1] - self._current.serial[1])
                dz = abs(serial[2] - self._current.serial[2])
                if max(dx, dy, dz) > self._jump_threshold:
                    if self._pending_cell == serial:
                        self._pending_count += 1
                        if self._pending_count < 2:
                            return self._current
                    else:
                        self._pending_cell = serial
                        self._pending_count = 1
                        return self._current
            self._pending_cell = None
            self._pending_count = 0

            if interpolate and self._current is not None \
                    and self._current.serial != serial \
                    and self.learning_mode and grounded:
                self._fill_segment(self._current.serial, serial)

            existing = self.graph.get(serial)
            if existing is None:
                node = Node(serial=serial, node_type=NodeType.REACHABLE)
                if self.learning_mode and grounded:
                    self.graph.add_node(node)
                    self._dirty = True
            else:
                node = existing
                # reference impl: if we re-walk an UnReachable cell, demote to Iffy
                if self.learning_mode and grounded \
                        and node.node_type == NodeType.UNREACHABLE:
                    node.node_type = NodeType.IFFY
                    self._dirty = True
            if self._current is None or self._current.serial != serial:
                self._previous = self._current
                self._current = node
            return node

    def _fill_segment(self, a: Serial, b: Serial) -> None:
        """Bresenham-style 3D line fill between two voxels. Marks every cell
        on the line as Reachable. Skips the endpoints (caller handles
        them). Bails on huge gaps (>32 cells) to avoid teleport-glitches
        painting trails across the map."""
        dx = b[0] - a[0]; dy = b[1] - a[1]; dz = b[2] - a[2]
        steps = max(abs(dx), abs(dy), abs(dz))
        if steps <= 1 or steps > 32:
            return
        # if Y motion dominates the segment its almost always a glitch
        # (pose decoder noise on a step, brief fall, jump-in-place). dont
        # paint vertical green columns through the ceiling.
        horiz = max(abs(dx), abs(dz))
        if abs(dy) > horiz + 1:
            return
        for i in range(1, steps):
            t = i / steps
            sx = a[0] + int(round(dx * t))
            sy = a[1] + int(round(dy * t))
            sz = a[2] + int(round(dz * t))
            cell = (sx, sy, sz)
            if cell not in self.graph:
                self.graph.add_node(Node(serial=cell, node_type=NodeType.REACHABLE))
                self._dirty = True

    def mark_unreachable(self, serial: Serial) -> None:
        with self._lock:
            node = self.graph.get(serial)
            if node is None:
                node = Node(serial=serial, node_type=NodeType.UNREACHABLE)
                self.graph.add_node(node)
            else:
                node.node_type = NodeType.UNREACHABLE
            self._dirty = True

    def set_cell_type(self, serial: Serial, node_type: NodeType) -> Node:
        """Manual override for the WebUI editor. Creates the node if it
        doesnt exist, otherwise just flips its type."""
        with self._lock:
            node = self.graph.get(serial)
            if node is None:
                node = Node(serial=serial, node_type=node_type)
                self.graph.add_node(node)
            else:
                node.node_type = node_type
            self._dirty = True
            return node

    def delete_cell(self, serial: Serial) -> bool:
        """Manual delete from the WebUI editor. Returns True if a cell was
        actually removed."""
        with self._lock:
            existed = serial in self.graph
            if existed:
                self.graph.remove_node(serial)
                # if we just nuked the cell we thought we were standing in,
                # clear the cached current so the next observe rebuilds it.
                if self._current is not None and self._current.serial == serial:
                    self._current = None
                if self._previous is not None and self._previous.serial == serial:
                    self._previous = None
                self._dirty = True
            return existed

    # --- reference-style discovery helpers ------------------------------------
    def check_vertical(self, serial: Serial) -> bool:
        """reference CheckVertical: a candidate cell counts as 'already known' if
        the cell itself or its +Y / -Y neighbor exists in the graph. Used
        to find unexplored cardinal neighbors of a Reachable node while
        being tolerant to ~1 cell of floor height variation (stairs)."""
        sx, sy, sz = serial
        if serial in self.graph:
            return True
        if (sx, sy + 1, sz) in self.graph:
            return True
        if (sx, sy - 1, sz) in self.graph:
            return True
        return False

    def choose_discovery_target(self, node: Node, forward_xz: tuple[float, float],
                                ) -> Optional[Serial]:
        """reference CheckNodeForTarget: starting from the cardinal cell in front
        of the avatar, then rotating 90deg / 180deg / 270deg, return the
        first unexplored cell. None if all 4 cardinals are known."""
        fx, fz = forward_xz
        if abs(fx) > abs(fz):
            offset = (1 if fx >= 0 else -1, 0, 0)
        else:
            offset = (0, 0, 1 if fz >= 0 else -1)
        # try forward, then rotated +90, +180, +270
        for _ in range(4):
            cand = (node.serial[0] + offset[0],
                    node.serial[1] + offset[1],
                    node.serial[2] + offset[2])
            if not self.check_vertical(cand):
                return cand
            # rotate 90 deg in XZ: (x,z) -> (-z,x)
            offset = (-offset[2], 0, offset[0])
        return None

    def check_stack(self, forward_xz: tuple[float, float],
                    blacklist: Optional[set[Serial]] = None,
                    ) -> Optional[tuple[Serial, Node]]:
        """reference CheckStack: scan all Reachable nodes, find the one with the
        closest-to-current unexplored cardinal neighbor. Returns
        (target_serial, source_node) or None. `blacklist` skips candidate
        target cells the caller has temporarily given up on."""
        if self._current is None:
            return None
        cur_cx, _, cur_cz = serial_to_center(self._current.serial)
        best: Optional[tuple[Serial, Node]] = None
        best_d = math.inf
        with self.graph._lock:  # noqa: SLF001
            items = list(self.graph.nodes.values())
        for src in items:
            if src.node_type != NodeType.REACHABLE:
                continue
            cand = self.choose_discovery_target(src, forward_xz)
            if cand is None:
                continue
            if blacklist is not None and cand in blacklist:
                continue
            cx, _, cz = serial_to_center(cand)
            dx = cx - cur_cx; dz = cz - cur_cz
            d = dx * dx + dz * dz
            if d < best_d:
                best_d = d
                best = (cand, src)
        return best

    def is_pathable_neighbor(self, a: Serial, b: Serial) -> bool:
        """reference IsPathableNeighbor: b is within the 3x3x3 cube around a."""
        return (abs(a[0] - b[0]) <= 1
                and abs(a[1] - b[1]) <= 1
                and abs(a[2] - b[2]) <= 1)

    def bar_check(self, a: Serial, b: Serial) -> bool:
        """reference BarCheck: b matches a or a+/-1 on Y (same column tolerance)."""
        return (a[0] == b[0] and a[2] == b[2]
                and abs(a[1] - b[1]) <= 1)

    @property
    def current(self) -> Optional[Node]:
        return self._current

    @property
    def previous(self) -> Optional[Node]:
        return self._previous

    # --- planning ----------------------------------------------------------
    def plan_to(self, target_world_xyz: tuple[float, float, float],
                start_world_xyz: Optional[tuple[float, float, float]] = None,
                snap_target: bool = True) -> VoxelPathResult:
        with self._lock:
            if start_world_xyz is None and self._current is not None:
                start_serial = self._current.serial
            elif start_world_xyz is not None:
                start_node = self.graph.find_node(*start_world_xyz)
                if start_node is None:
                    start_node = self.graph.find_closest(*start_world_xyz)
                if start_node is None:
                    return VoxelPathResult(found=False)
                start_serial = start_node.serial
            else:
                return VoxelPathResult(found=False)

            goal_node = self.graph.find_node(*target_world_xyz)
            if goal_node is None and snap_target:
                goal_node = self.graph.find_closest(*target_world_xyz)
            if goal_node is None:
                return VoxelPathResult(found=False)
            return find_path_astar(self.graph, start_serial, goal_node.serial)
