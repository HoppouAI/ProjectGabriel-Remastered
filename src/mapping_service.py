"""Mapping + waypoint orchestration for the main webUI.

Wraps PoseExfilReader, VoxelNavManager, VoxelExplorer and WaypointStore
into one easy-to-poke service. The webUI hits this; nothing autostarts.

Designed to be created once in main.py and shoved into control_server's
shared_state under "mapping_service".
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.pose_decoder import GRID_W, GRID_H, PoseExfilReader, WorldPose
from src.voxel_nav import (
    NodeType, VoxelNavManager, VoxelPathResult,
    find_path_astar, world_to_serial, serial_to_center,
)
from src.voxel_explorer import VoxelExplorer
from src.waypoints import WaypointStore, Waypoint

logger = logging.getLogger(__name__)


@dataclass
class _RegionGuess:
    monitor_index: int
    abs_x: int
    abs_y: int
    cell: int


class MappingService:
    """Owns the mapping subsystem. All public methods are thread safe.

    Lifecycle:
        ms = MappingService(osc, instance_monitor=im)
        ms.start(explore=True)      # webUI clicks Start
        ms.add_waypoint("couch")    # adds at current pose
        ms.stop()                   # webUI clicks Stop

    Each call to start() rescans for the pose strip on the screen, so the
    user can move the strip around between sessions.
    """

    DEFAULT_WORLD = "default"

    def __init__(self, osc, *, instance_monitor=None,
                 data_dir: str | Path = "data/voxel_nav"):
        self._osc = osc
        self._instance_monitor = instance_monitor
        self._lock = threading.RLock()

        self._nav = VoxelNavManager(data_dir=data_dir, learning_mode=True)
        self._reader: Optional[PoseExfilReader] = None
        self._explorer: Optional[VoxelExplorer] = None
        self._waypoints: Optional[WaypointStore] = None

        self._world_id: str = self.DEFAULT_WORLD
        self._world_name: str = ""
        self._running = False
        self._explore_enabled = False
        # True when the explorer was spun up only for a path-follow (goto)
        # and should be torn down once the follow completes, rather than
        # falling through into frontier discovery.
        self._explorer_follow_only = False
        # if set (degrees), once the current follow finishes we rotate the
        # avatar to match this yaw and then sit still. used so waypoint
        # gotos land you facing the same way you were when you saved.
        self._pending_align_yaw: Optional[float] = None
        # manual mapping: user drives the avatar themselves, we just label
        # cells they walk through as Reachable (via nav.observe) and use the
        # forward raycast to flag the cell directly in front as a wall when
        # we get a near-zero reading. great for fast first-pass mapping.
        self._manual_mapping = False
        self._manual_wall_throttle: dict[tuple[int, int, int], float] = {}
        self._manual_debug_last: float = 0.0
        # tunables for the raycast wall trip (also exposed via API)
        self.manual_wall_distance = 0.35   # m -- at or below this counts as wall
        self.manual_wall_ratio = 0.07      # ratio fallback for short rays
        self.manual_ray_name = "Fwd"        # which named ray to read
        # while manual mapping is on, hard-lock the avatar's yaw to the
        # nearest cardinal (0/90/180/270) and strafe-correct so they walk
        # down the center of a single voxel row. user just pushes forward.
        self.manual_grid_snap = True
        self._last_error: str = ""
        self._region: Optional[_RegionGuess] = None
        self._last_pose: Optional[WorldPose] = None
        self._last_pose_t: float = 0.0
        self._tick_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # tunables exposed via the UI / api
        self._tick_hz: float = 20.0          # pose sample rate
        self._force_run: bool = False        # always sprint while exploring
        # default move speed for follow + explore. proxies through to the
        # explorer's speed_mode any time we (re)create one.
        self._speed_mode: str = "fast"

        # ensure waypoint store at least exists for the default world so the
        # UI can list/add even when mapping hasnt been started yet.
        self._ensure_waypoints(self._world_id)

    # ------------------------------------------------------------------
    # world id
    # ------------------------------------------------------------------
    def _resolve_world_id(self) -> str:
        if self._instance_monitor is not None:
            try:
                wid = getattr(self._instance_monitor, "world_id", "")
                if wid:
                    return wid
                # legacy: current_location is "world:instance", strip instance
                loc = getattr(self._instance_monitor, "current_location", "")
                if loc and ":" in loc:
                    return loc.split(":", 1)[0]
                if loc:
                    return loc
            except Exception:
                pass
        return self.DEFAULT_WORLD

    def _resolve_world_name(self) -> str:
        if self._instance_monitor is not None:
            try:
                return getattr(self._instance_monitor, "world_name", "") or ""
            except Exception:
                pass
        return ""

    def _ensure_waypoints(self, world_id: str) -> None:
        if self._waypoints is None or self._waypoints.world_id != world_id:
            self._waypoints = WaypointStore(world_id)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self, *, explore: bool = False) -> dict:
        """Find the pose strip, load the world, kick the tick loop."""
        with self._lock:
            if self._running:
                # allow toggling exploration without a full restart
                self._set_explore(explore)
                return self.get_state()

            world = self._resolve_world_id()
            self._world_id = world
            self._world_name = self._resolve_world_name()
            self._nav.load_world(world)
            self._ensure_waypoints(world)

            # scan for pose strip
            try:
                from scripts.test_pose_decoder_live import scan_and_decode
            except Exception as exc:
                self._last_error = f"scan import failed: {exc}"
                logger.exception("mapping: scan import failed")
                return self.get_state()

            try:
                result = scan_and_decode(8)
            except Exception as exc:
                self._last_error = f"strip scan crashed: {exc}"
                logger.exception("mapping: strip scan crashed")
                return self.get_state()

            if not isinstance(result, tuple) or result[0] != 0:
                self._last_error = ("could not find pose strip on screen. "
                                    "is VRChat focused with the shader on?")
                logger.warning("mapping: %s", self._last_error)
                return self.get_state()

            _, mi, ax, ay, cell = result
            self._region = _RegionGuess(mi, ax, ay, cell)

            region = {
                "left": ax, "top": ay,
                "width":  GRID_W * cell,
                "height": GRID_H * cell,
            }
            self._reader = PoseExfilReader(
                region=region, cell_size=cell,
                poll_hz=20.0, monitor_index=mi,
            )
            self._reader.start()

            if explore:
                self._explorer = VoxelExplorer(self._nav, self._osc,
                                                learning_mode=True)
                self._explorer.force_run = self._force_run
                self._explorer.speed_mode = self._speed_mode
                self._explorer.start()
                self._explore_enabled = True
            else:
                self._explorer = None
                self._explore_enabled = False

            self._stop_evt.clear()
            self._tick_thread = threading.Thread(
                target=self._run, daemon=True, name="mapping-tick")
            self._tick_thread.start()

            self._running = True
            self._last_error = ""
            logger.info("mapping: started (world=%s explore=%s)",
                        world, explore)
            return self.get_state()

    def stop(self) -> dict:
        with self._lock:
            if not self._running:
                return self.get_state()
            self._stop_evt.set()
            try:
                if self._explorer is not None:
                    self._explorer.stop()
            except Exception:
                logger.exception("mapping: explorer stop failed")
            self._explorer = None
            self._explore_enabled = False
            self._explorer_follow_only = False
            self._pending_align_yaw = None
            self._manual_mapping = False
            self._manual_wall_throttle.clear()
            try:
                if self._reader is not None:
                    self._reader.stop()
            except Exception:
                logger.exception("mapping: reader stop failed")
            self._reader = None
            try:
                self._nav.flush()
            except Exception:
                logger.exception("mapping: nav flush failed")
            # zero movement just in case
            try:
                self._osc.client.send_message("/input/Vertical", 0.0)
                self._osc.client.send_message("/input/Horizontal", 0.0)
                self._osc.client.send_message("/input/LookHorizontal", 0.0)
                self._osc.client.send_message("/input/Run", 0)
            except Exception:
                pass
            self._running = False
            logger.info("mapping: stopped")
            return self.get_state()

    def set_explore(self, enabled: bool) -> dict:
        with self._lock:
            self._set_explore(enabled)
            return self.get_state()

    def _set_explore(self, enabled: bool) -> None:
        if not self._running:
            # remember desired state for next start
            self._explore_enabled = enabled
            return
        if enabled and self._explorer is None:
            self._explorer = VoxelExplorer(self._nav, self._osc,
                                            learning_mode=True)
            self._explorer.force_run = self._force_run
            self._explorer.speed_mode = self._speed_mode
            self._explorer.start()
            self._explore_enabled = True
            self._explorer_follow_only = False
            logger.info("mapping: explorer enabled")
        elif enabled and self._explorer is not None:
            # explorer already running (likely from a goto) -- keep it but
            # stop treating it as follow-only so it can run discovery.
            self._explore_enabled = True
            self._explorer_follow_only = False
        elif not enabled and self._explorer is not None:
            try:
                self._explorer.stop()
            except Exception:
                logger.exception("mapping: explorer stop failed")
            self._explorer = None
            self._explore_enabled = False
            self._explorer_follow_only = False
            logger.info("mapping: explorer disabled")

    # ------------------------------------------------------------------
    # tick loop -- reads pose, feeds nav + explorer, persists periodically
    # ------------------------------------------------------------------
    def _handle_world_change(self, new_world: str) -> None:
        """Detected that VRChat moved us to a different world. Flush the
        old map, swap in the new one, and reset all per-world state so we
        dont observe the new pose into the old map (which creates a stray
        voxel out in the void of the new map at the old coords)."""
        with self._lock:
            old = self._world_id
            logger.info("mapping: world change %s -> %s, hot swapping",
                        old, new_world)
            # stop the explorer cold so it cant drive on a stale follow
            # queue thats indexed against the old map.
            if self._explorer is not None:
                try:
                    self._explorer.stop()
                except Exception:
                    logger.exception("mapping: explorer stop on world swap failed")
                self._explorer = None
            self._explore_enabled = False
            self._explorer_follow_only = False
            self._pending_align_yaw = None
            # flush + load. load_world also clears nav._current/_previous.
            try:
                self._nav.load_world(new_world)
            except Exception:
                logger.exception("mapping: load_world failed during swap")
            self._world_id = new_world
            self._world_name = self._resolve_world_name()
            self._ensure_waypoints(new_world)
            # forget the last pose so the next tick doesnt paint the old
            # coords into the new map.
            self._last_pose = None
            self._last_pose_t = 0.0
            self._manual_wall_throttle.clear()
            # zero movement just in case the avatar was mid-input.
            try:
                self._osc.client.send_message("/input/Vertical", 0.0)
                self._osc.client.send_message("/input/Horizontal", 0.0)
                self._osc.client.send_message("/input/LookHorizontal", 0.0)
                self._osc.client.send_message("/input/Run", 0)
            except Exception:
                pass

    def _run(self) -> None:
        last_flush = time.time()
        last_world_check = 0.0
        while not self._stop_evt.is_set():
            reader = self._reader
            if reader is None:
                break
            try:
                # check for VRChat world change ~every 2s. cheap, just a
                # string compare against the instance monitor.
                now_pre = time.time()
                if now_pre - last_world_check >= 2.0:
                    last_world_check = now_pre
                    try:
                        new_world = self._resolve_world_id()
                        if new_world and new_world != self._world_id:
                            self._handle_world_change(new_world)
                            continue  # skip this tick, dont use stale pose
                    except Exception:
                        logger.exception("mapping: world change probe failed")
                pose = reader.get()
                if pose is not None and pose.timestamp != self._last_pose_t:
                    self._last_pose_t = pose.timestamp
                    self._last_pose = pose
                    grounded = bool(getattr(self._osc, "grounded", True))
                    self._nav.observe(pose.x, pose.y, pose.z,
                                       grounded=grounded, interpolate=True)
                    if self._explorer is not None:
                        self._explorer.tick(pose.x, pose.y, pose.z, pose.yaw)
                        # if we only spun the explorer up for a goto, tear
                        # it down the moment the follow queue is empty so
                        # we dont silently slide into discovery mode.
                        if (self._explorer_follow_only
                                and not self._explore_enabled
                                and self._explorer is not None
                                and not self._explorer.follow_status.get("active")):
                            try:
                                self._explorer.stop()
                            except Exception:
                                logger.exception("mapping: explorer auto-stop failed")
                            self._explorer = None
                            self._explorer_follow_only = False
                            logger.info("mapping: explorer torn down after goto complete")
                    # yaw alignment runs after the explorer is gone so they
                    # dont fight over LookHorizontal.
                    if (self._pending_align_yaw is not None
                            and self._explorer is None):
                        self._drive_yaw_alignment(pose.yaw)
                    # manual mapping: only active when the user has explicitly
                    # toggled it on and the explorer isnt driving. uses the
                    # forward raycast to flag obvious walls.
                    if (self._manual_mapping
                            and self._explorer is None):
                        self._manual_mapping_tick(pose)
                        # grid lock runs only when manual is on, the
                        # explorer isnt driving, and theres no pending
                        # waypoint alignment fighting for LookHorizontal.
                        if (self.manual_grid_snap
                                and self._pending_align_yaw is None):
                            self._drive_grid_lock(pose)
                now = time.time()
                if now - last_flush >= 5.0:
                    self._nav.flush()
                    last_flush = now
            except Exception:
                logger.exception("mapping: tick failed")
            interval = 1.0 / max(1.0, min(self._tick_hz, 120.0))
            time.sleep(interval)
        # final flush
        try:
            self._nav.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # state queries
    # ------------------------------------------------------------------
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
        slowly. Each cell is [sx, sy, sz]."""
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
        return {"world": self._world_id, "reach": reach,
                "wall": wall, "iffy": iffy}

    # ------------------------------------------------------------------
    # waypoints
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # manual cell edits (3D viewer)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # stray voxel cleanup
    # ------------------------------------------------------------------
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
                for s in to_remove:
                    self._nav.delete_cell(s)
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

    # ------------------------------------------------------------------
    # world management (list / delete saved maps)
    # ------------------------------------------------------------------
    def update_settings(self, *, tick_hz: float | None = None,
                        force_run: bool | None = None,
                        manual_wall_distance: float | None = None,
                        manual_wall_ratio: float | None = None,
                        speed_mode: str | None = None) -> dict:
        """Live-tune mapping speed knobs. Returns the new settings dict."""
        with self._lock:
            if tick_hz is not None:
                self._tick_hz = max(5.0, min(float(tick_hz), 120.0))
            if force_run is not None:
                self._force_run = bool(force_run)
                if self._explorer is not None:
                    try:
                        self._explorer.force_run = self._force_run
                    except Exception:
                        pass
            if speed_mode is not None:
                mode = str(speed_mode).strip().lower()
                if mode not in ("walk", "fast", "run"):
                    raise ValueError(
                        f"speed_mode must be walk/fast/run, got {speed_mode!r}")
                self._speed_mode = mode
                if self._explorer is not None:
                    try:
                        self._explorer.speed_mode = mode
                    except Exception:
                        pass
            if manual_wall_distance is not None:
                self.manual_wall_distance = max(
                    0.02, min(float(manual_wall_distance), 2.0))
            if manual_wall_ratio is not None:
                self.manual_wall_ratio = max(
                    0.0, min(float(manual_wall_ratio), 1.0))
            return {
                "tick_hz": self._tick_hz,
                "force_run": self._force_run,
                "speed_mode": self._speed_mode,
                "manual_wall_distance": self.manual_wall_distance,
                "manual_wall_ratio": self.manual_wall_ratio,
            }

    def list_worlds(self) -> list[dict]:
        """All saved world maps on disk. Useful for the UI's delete menu."""
        out: list[dict] = []
        try:
            for p in self._nav._data_dir.glob("*.json"):  # noqa: SLF001
                world_id = p.stem
                size_kb = p.stat().st_size / 1024.0
                out.append({
                    "world": world_id,
                    "size_kb": round(size_kb, 1),
                    "is_current": world_id == self._world_id,
                })
        except Exception:
            logger.exception("mapping: list_worlds failed")
        out.sort(key=lambda w: w["world"])
        return out

    def delete_world(self, world_id: str | None = None) -> dict:
        """Delete a saved world map. If world_id is None or matches the
        current world, also clears the in-memory graph and stops mapping."""
        with self._lock:
            target = world_id or self._world_id
            target = target.strip()
            if not target:
                raise ValueError("world id required")

            is_current = (target == self._world_id)
            if is_current and self._running:
                # stop tick loop so we dont re-save immediately
                self._stop_evt.set()
                try:
                    if self._explorer is not None:
                        self._explorer.stop()
                except Exception:
                    pass
                self._explorer = None
                self._explore_enabled = False
                try:
                    if self._reader is not None:
                        self._reader.stop()
                except Exception:
                    pass
                self._reader = None
                self._running = False

            removed = False
            try:
                path = self._nav._data_dir / f"{target}.json"  # noqa: SLF001
                if path.exists():
                    path.unlink()
                    removed = True
            except Exception as exc:
                logger.exception("mapping: delete world file failed")
                raise RuntimeError(f"delete failed: {exc}") from exc

            if is_current:
                # wipe in-memory graph too so the viewer empties out
                try:
                    with self._nav.graph._lock:  # noqa: SLF001
                        self._nav.graph.nodes.clear()
                    self._nav._current = None  # noqa: SLF001
                    self._nav._previous = None  # noqa: SLF001
                    self._nav._dirty = False  # noqa: SLF001
                except Exception:
                    logger.exception("mapping: wipe graph failed")

            logger.info("mapping: deleted world '%s' (file_removed=%s)",
                        target, removed)
            return {"world": target, "removed": removed,
                    "was_current": is_current}

    # ------------------------------------------------------------------
    # pathfinding (A* preview only -- no driving)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # drive-to-waypoint (active path follow)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # manual mapping mode -- user walks, we listen to the fwd raycast
    # ------------------------------------------------------------------
    def set_manual_mapping(self, enabled: bool) -> dict:
        with self._lock:
            self._manual_mapping = bool(enabled)
            if not enabled:
                self._manual_wall_throttle.clear()
                # release any grid-lock outputs so user isnt stuck strafing
                try:
                    self._osc.client.send_message("/input/LookHorizontal", 0.0)
                    self._osc.client.send_message("/input/Horizontal", 0.0)
                except Exception:
                    pass
            logger.info("mapping: manual mapping %s",
                        "ON" if enabled else "off")
            return self.get_state()

    def _manual_mapping_tick(self, pose: WorldPose) -> None:
        """Read the forward raycast and, if it's reading a near-zero hit,
        mark the cell directly in front of us as a wall. Throttled per cell
        so we dont spam the graph with the same write every tick."""
        raycast = getattr(self._osc, "raycast_state", None)
        now_dbg = time.monotonic()
        debug_due = (now_dbg - self._manual_debug_last) >= 2.0
        if raycast is None:
            if debug_due:
                logger.warning("mapping(manual): no raycast_state on osc client")
                self._manual_debug_last = now_dbg
            return
        reading = raycast.get(self.manual_ray_name)
        if reading is None:
            if debug_due:
                try:
                    known = list(raycast._rays.keys())  # noqa: SLF001
                except Exception:
                    known = []
                logger.warning("mapping(manual): no ray named %r yet (known rays: %s)",
                               self.manual_ray_name, known)
                self._manual_debug_last = now_dbg
            return
        if reading.last_updated <= 0.0:
            # never got a single packet for this ray, nothing to act on
            if debug_due:
                logger.warning("mapping(manual): %s has no readings yet", self.manual_ray_name)
                self._manual_debug_last = now_dbg
            return
        if debug_due:
            logger.info("mapping(manual): %s hit=%s d=%.3f r=%.3f (thresh d<=%.2f r<=%.2f)",
                        self.manual_ray_name, reading.hit, reading.distance, reading.ratio,
                        self.manual_wall_distance, self.manual_wall_ratio)
            self._manual_debug_last = now_dbg
        # just trust the distance. user tunes the threshold via the slider,
        # if distance is at or below it we call it a wall. Hit and Ratio
        # both turned out to be unreliable on the test avatar.
        is_wall = reading.distance <= self.manual_wall_distance
        if not is_wall:
            return
        # compute cell ~1 voxel in front of us using pose yaw
        yaw_rad = math.radians(pose.yaw)
        fx = math.sin(yaw_rad)
        fz = math.cos(yaw_rad)
        # push half a cell past where the ray says the wall is, but clamp
        # so we always mark at least one cell ahead of us
        push = max(0.30, reading.distance + 0.15)
        wx = pose.x + fx * push
        wz = pose.z + fz * push
        cell = world_to_serial(wx, pose.y, wz)
        now = time.monotonic()
        last = self._manual_wall_throttle.get(cell, 0.0)
        if now - last < 2.0:
            return
        self._manual_wall_throttle[cell] = now
        try:
            self._nav.mark_unreachable(cell)
            logger.info("mapping: manual wall at %s (d=%.2f r=%.2f)",
                        cell, reading.distance, reading.ratio)
        except Exception:
            logger.exception("mapping: mark_unreachable failed")

    def _drive_grid_lock(self, pose: WorldPose) -> None:
        """Hard-lock yaw to nearest cardinal and strafe-correct lateral
        offset so the user walks straight down a single voxel row. Forward
        input is left alone, the user drives that themselves."""
        # --- yaw lock ---
        target_yaw = (round(pose.yaw / 90.0) * 90.0) % 360.0
        delta = (target_yaw - pose.yaw + 540.0) % 360.0 - 180.0
        if abs(delta) <= 0.5:
            yaw_out = 0.0
        else:
            sign = 1.0 if delta > 0 else -1.0
            # strong pull: 0.15 floor so we move even at small deltas, ramp
            # to full stick at ~15deg+ so big offsets snap fast.
            mag = min(1.0, max(0.15, abs(delta) / 15.0))
            yaw_out = sign * mag
        try:
            self._osc.client.send_message("/input/LookHorizontal", yaw_out)
        except Exception:
            pass

        # --- lateral lock ---
        # if the user is actively strafing themselves (avatar local VelocityX
        # is well above floor), back off so they can hop to a neighbouring
        # row. once they let go we re-center on whichever row they ended up
        # on. VelocityX is in avatar-local frame so positive = strafing right.
        user_strafe = abs(getattr(self._osc, "velocity_x", 0.0))
        if user_strafe > 0.25:
            try:
                self._osc.client.send_message("/input/Horizontal", 0.0)
            except Exception:
                pass
            return
        # right vector for the snapped cardinal: facing +Z (yaw 0) -> right=+X.
        # right_x = cos(yaw), right_z = -sin(yaw)
        yaw_rad = math.radians(target_yaw)
        rx = math.cos(yaw_rad)
        rz = -math.sin(yaw_rad)
        # cell center under the player
        cell = world_to_serial(pose.x, pose.y, pose.z)
        cx, _cy, cz = serial_to_center(cell)
        # signed lateral offset, positive means player is to the RIGHT of center
        lateral = (pose.x - cx) * rx + (pose.z - cz) * rz
        if abs(lateral) <= 0.03:
            strafe_out = 0.0
        else:
            # negative strafe to correct rightward drift, positive for leftward
            sign = -1.0 if lateral > 0 else 1.0
            mag = min(0.6, max(0.1, abs(lateral) / 0.12))
            strafe_out = sign * mag
        try:
            self._osc.client.send_message("/input/Horizontal", strafe_out)
        except Exception:
            pass

    def follow_status(self) -> dict:
        if self._explorer is None:
            return {"active": False, "remaining": 0, "label": ""}
        try:
            return self._explorer.follow_status
        except Exception:
            return {"active": False, "remaining": 0, "label": ""}
