"""WebUI cell edits + voxel graph cleanup passes."""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.voxel_nav import NodeType

logger = logging.getLogger(__name__)


class EditingMixin:
    """3D viewer cell edits, stray-voxel cleanup, jump-artifact cleanup."""

    def edit_cell(self, sx: int, sy: int, sz: int, kind: str) -> dict:
        """Manually flip a voxel from the WebUI. `kind` is one of
        reach / wall / iffy / delete."""
        kind_norm = (kind or "").strip().lower()
        serial = (int(sx), int(sy), int(sz))
        type_map = {
            "reach": NodeType.REACHABLE,
            "reachable": NodeType.REACHABLE,
            "wall": NodeType.UNREACHABLE,
            "unreachable": NodeType.UNREACHABLE,
            "iffy": NodeType.IFFY,
        }
        if kind_norm == "delete":
            existed = self._nav.delete_cell(serial)
            self._nav.flush()
            logger.info("mapping: edit delete %s (existed=%s)", serial, existed)
            return {"result": "ok", "kind": "delete",
                    "cell": list(serial), "existed": existed}
        if kind_norm not in type_map:
            raise ValueError(f"unknown cell kind '{kind}'")
        node = self._nav.set_cell_type(serial, type_map[kind_norm])
        self._nav.flush()
        logger.info("mapping: edit set %s -> %s",
                    serial, node.node_type.name)
        return {"result": "ok", "kind": kind_norm,
                "cell": list(serial), "type": node.node_type.name}

    def edit_cells_bulk(self, cells: list[tuple[int, int, int]],
                         kind: str) -> dict:
        """Apply the same edit to many cells in one shot. Flushes once at
        the end so a 500-cell drag select doesnt write the json 500 times."""
        kind_norm = (kind or "").strip().lower()
        type_map = {
            "reach": NodeType.REACHABLE,
            "reachable": NodeType.REACHABLE,
            "wall": NodeType.UNREACHABLE,
            "unreachable": NodeType.UNREACHABLE,
            "iffy": NodeType.IFFY,
        }
        applied = 0
        if kind_norm == "delete":
            for c in cells:
                if self._nav.delete_cell((int(c[0]), int(c[1]), int(c[2]))):
                    applied += 1
        else:
            if kind_norm not in type_map:
                raise ValueError(f"unknown cell kind '{kind}'")
            nt = type_map[kind_norm]
            for c in cells:
                self._nav.set_cell_type((int(c[0]), int(c[1]), int(c[2])), nt)
                applied += 1
        self._nav.flush()
        logger.info("mapping: bulk edit %s applied=%d/%d",
                    kind_norm, applied, len(cells))
        return {"result": "ok", "kind": kind_norm, "applied": applied,
                "total": len(cells)}

    def cleanup_strays(self, *, min_component_size: int = 8,
                        dry_run: bool = False) -> dict:
        """Find connected components in the voxel graph and delete the tiny
        floating ones. Uses 26-connectivity so cells diagonally touching
        each other (eg stairs) count as connected.

        The biggest component is always kept (thats your main map). Any
        component with a waypoint or the avatars current cell is also
        kept regardless of size. Everything else gets nuked if its
        smaller than min_component_size.
        """
        with self._lock:
            # snapshot serials so we dont hold the graph lock while BFSing
            with self._nav.graph._lock:  # noqa: SLF001
                serials = set(self._nav.graph.nodes.keys())
            total_cells = len(serials)
            if total_cells == 0:
                return {"result": "ok", "components_total": 0,
                        "components_removed": 0, "cells_removed": 0,
                        "cells_kept": 0, "kept_due_to_waypoint": 0,
                        "kept_due_to_avatar": 0, "largest_component": 0,
                        "dry_run": bool(dry_run),
                        "min_component_size": int(min_component_size)}

            # 26-neighborhood (all dx,dy,dz in -1..1 except origin)
            neighbor_offsets = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ]

            # BFS connected components
            unseen = set(serials)
            components: list[set[tuple[int, int, int]]] = []
            bfs_counter = 0
            while unseen:
                start = next(iter(unseen))
                comp: set[tuple[int, int, int]] = set()
                stack = [start]
                while stack:
                    s = stack.pop()
                    if s in comp:
                        continue
                    comp.add(s)
                    unseen.discard(s)
                    sx, sy, sz = s
                    for dx, dy, dz in neighbor_offsets:
                        n = (sx + dx, sy + dy, sz + dz)
                        if n in unseen:
                            stack.append(n)
                    bfs_counter += 1
                    if bfs_counter % 2000 == 0:
                        time.sleep(0)
                components.append(comp)

            largest_size = max(len(c) for c in components) if components else 0

            # protected cells: waypoints + avatar
            protected_serials: set[tuple[int, int, int]] = set()
            wp_serials: set[tuple[int, int, int]] = set()
            if self._waypoints is not None:
                for wp in self._waypoints.list():
                    try:
                        from src.voxel_nav import world_to_serial as _w2s
                        wp_serials.add(_w2s(wp.x, wp.y, wp.z))
                    except Exception:
                        pass
            protected_serials.update(wp_serials)
            avatar_serial: Optional[tuple[int, int, int]] = None
            if self._last_pose is not None:
                try:
                    from src.voxel_nav import world_to_serial as _w2s
                    avatar_serial = _w2s(self._last_pose.x,
                                          self._last_pose.y,
                                          self._last_pose.z)
                    protected_serials.add(avatar_serial)
                except Exception:
                    pass

            to_remove: list[tuple[int, int, int]] = []
            removed_components = 0
            kept_due_to_waypoint = 0
            kept_due_to_avatar = 0
            for comp in components:
                if len(comp) >= max(1, int(min_component_size)):
                    continue
                if len(comp) == largest_size:
                    continue  # always keep the main map
                # protection checks
                if avatar_serial is not None and avatar_serial in comp:
                    kept_due_to_avatar += 1
                    continue
                if comp & wp_serials:
                    kept_due_to_waypoint += 1
                    continue
                to_remove.extend(comp)
                removed_components += 1

            if not dry_run and to_remove:
                for i, s in enumerate(to_remove):
                    self._nav.delete_cell(s)
                    if i % 3000 == 0:
                        time.sleep(0)
                self._nav.flush()

            logger.info("mapping: cleanup_strays components=%d removed=%d cells_removed=%d "
                        "kept_wp=%d kept_avatar=%d largest=%d min_size=%d dry_run=%s",
                        len(components), removed_components, len(to_remove),
                        kept_due_to_waypoint, kept_due_to_avatar,
                        largest_size, min_component_size, dry_run)

            return {
                "result": "ok",
                "components_total": len(components),
                "components_removed": removed_components,
                "cells_removed": len(to_remove),
                "cells_kept": total_cells - (0 if dry_run else len(to_remove)),
                "kept_due_to_waypoint": kept_due_to_waypoint,
                "kept_due_to_avatar": kept_due_to_avatar,
                "largest_component": largest_size,
                "min_component_size": int(min_component_size),
                "dry_run": bool(dry_run),
            }

    def cleanup_jump_artifacts(self, *, dry_run: bool = False) -> dict:
        """Remove floating single-cell-high voxels left over from old jump
        landings. The heuristic: a 'jump artifact' is a REACHABLE cell that
        sits exactly 1 cell above another REACHABLE cell at the same (x,z)
        AND has zero same-level REACHABLE horizontal neighbors (8-conn at
        same y). That pattern almost always means a brief mid-jump observe
        snapped to the floor+1 row.

        Protects waypoint cells and the avatars current cell. The cell
        directly below the artifact is fine, only the lonely one above is
        removed."""
        with self._lock:
            with self._nav.graph._lock:  # noqa: SLF001
                serials = set(self._nav.graph.nodes.keys())
                node_types = {s: self._nav.graph.nodes[s].node_type
                              for s in serials}

            # protected cells
            wp_serials: set[tuple[int, int, int]] = set()
            if self._waypoints is not None:
                for wp in self._waypoints.list():
                    try:
                        from src.voxel_nav import world_to_serial as _w2s
                        wp_serials.add(_w2s(wp.x, wp.y, wp.z))
                    except Exception:
                        pass
            avatar_serial: Optional[tuple[int, int, int]] = None
            if self._last_pose is not None:
                try:
                    from src.voxel_nav import world_to_serial as _w2s
                    avatar_serial = _w2s(self._last_pose.x,
                                          self._last_pose.y,
                                          self._last_pose.z)
                except Exception:
                    pass

            to_remove: list[tuple[int, int, int]] = []
            for s in serials:
                if node_types.get(s) != NodeType.REACHABLE:
                    continue
                if s in wp_serials or s == avatar_serial:
                    continue
                sx, sy, sz = s
                below = (sx, sy - 1, sz)
                if below not in serials:
                    continue
                if node_types.get(below) != NodeType.REACHABLE:
                    continue
                # must have zero same-level reachable horizontal neighbors
                lonely = True
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dz == 0:
                            continue
                        n = (sx + dx, sy, sz + dz)
                        if n in serials and node_types.get(n) == NodeType.REACHABLE:
                            lonely = False
                            break
                    if not lonely:
                        break
                if lonely:
                    to_remove.append(s)

            if not dry_run and to_remove:
                for s in to_remove:
                    self._nav.delete_cell(s)
                self._nav.flush()

            logger.info("mapping: cleanup_jump_artifacts removed=%d dry_run=%s",
                        len(to_remove), dry_run)
            return {
                "result": "ok",
                "cells_removed": len(to_remove),
                "cells_kept": len(serials) - (0 if dry_run else len(to_remove)),
                "dry_run": bool(dry_run),
            }
