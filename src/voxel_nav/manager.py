"""Per-world `VoxelNavManager`. Learning, persistence, discovery helpers."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from pathlib import Path
from typing import Optional

from .coords import Node, NodeType, Serial, serial_to_center, world_to_serial
from .graph import Graph
from .pathfinding import VoxelPathResult, _neighbors, find_path_astar


logger = logging.getLogger(__name__)


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
        # track last grounded state so a jump (false -> true transition)
        # forces us to drop _current/_previous before processing the
        # landing pose. otherwise interpolate would paint a trail from
        # takeoff to landing through wherever the jump arc went.
        self._last_grounded: bool = True

    # --- world lifecycle ---------------------------------------------------
    def load_world(self, world_id: str) -> None:
        with self._lock:
            if world_id == self._world_id:
                return
            self.flush()
            self._world_id = world_id
            self._current = None
            self._previous = None
            self._last_grounded = True
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
            # if we just landed (was airborne, now grounded), wipe the trail
            # tracker so the landing cell becomes a fresh anchor instead of
            # getting connected back to the takeoff cell with a fake walk
            # segment.
            if grounded and not self._last_grounded:
                self._current = None
                self._previous = None
                self._pending_cell = None
                self._pending_count = 0
            self._last_grounded = grounded
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
                # NOTE: the reference impl demotes an UnReachable cell to Iffy
                # whenever we walk through it. our pose decoder is noisier
                # than theirs (single-frame Y blips on stairs/teleports
                # can land "inside" a wall) so we leave walls alone here.
                # if a wall really should not be a wall, the user can flip
                # it in the editor or mark_iffy gets called explicitly by
                # the explorer when it gives up on a target.
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

    # --- cell type overrides ----------------------------------------------
    def mark_unreachable(self, serial: Serial) -> None:
        with self._lock:
            node = self.graph.get(serial)
            if node is None:
                node = Node(serial=serial, node_type=NodeType.UNREACHABLE)
                self.graph.add_node(node)
            else:
                node.node_type = NodeType.UNREACHABLE
            self._dirty = True

    def mark_iffy(self, serial: Serial) -> None:
        """reference NodeManager.MarkIffy: demote a cell to Iffy so future
        A* runs route around it, without commiting to a full wall mark.
        Used when the explorer gets stuck mid-follow but the cell might
        still be reachable from a different angle."""
        with self._lock:
            node = self.graph.get(serial)
            if node is None:
                node = Node(serial=serial, node_type=NodeType.IFFY)
                self.graph.add_node(node)
            elif node.node_type == NodeType.REACHABLE:
                node.node_type = NodeType.IFFY
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
        """Find the graph-nearest reachable cell that still has an unexplored
        cardinal neighbor. BFS outward from `_current` over the actual
        pathable graph using the same neighbor rules as the planner. This
        gives vacuum-like coverage (finish the connected region you're in
        before walking to a far-off pocket) instead of the old euclidean
        scoring which would happily pick a frontier across a wall just
        because it was closer in straight-line 3D."""
        if self._current is None:
            return None
        start = self._current.serial
        cur_cx, cur_cy, cur_cz = serial_to_center(start)
        visited: set[Serial] = {start}
        queue: deque[tuple[Serial, int]] = deque([(start, 0)])
        max_visits = 10000
        best: Optional[tuple[Serial, Node]] = None
        best_steps = math.inf
        best_d = math.inf
        while queue:
            serial, steps = queue.popleft()
            node = self.graph.get(serial)
            if node is None or node.node_type != NodeType.REACHABLE:
                continue
            cand = self.choose_discovery_target(node, forward_xz)
            if cand is not None and (blacklist is None or cand not in blacklist):
                cx, cy, cz = serial_to_center(cand)
                dx = cx - cur_cx
                dy = cy - cur_cy
                dz = cz - cur_cz
                d = dx * dx + dy * dy + dz * dz
                if steps < best_steps or (steps == best_steps and d < best_d):
                    best_steps = steps
                    best_d = d
                    best = (cand, node)
                # once we've found a frontier at depth N, finish draining
                # depth N for the euclidean tie-break and then stop.
                if best_steps < math.inf and steps > best_steps:
                    break
            if len(visited) >= max_visits:
                continue
            for nb, _cost in _neighbors(self.graph, serial):
                if nb in visited:
                    continue
                visited.add(nb)
                queue.append((nb, steps + 1))
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
