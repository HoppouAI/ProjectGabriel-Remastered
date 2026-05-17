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
    def _run(self) -> None:
        last_flush = time.time()
        while not self._stop_evt.is_set():
            reader = self._reader
            if reader is None:
                break
            try:
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
        self._ensure_waypoints(self._world_id)
        with self._lock:
            return [w.to_dict() for w in self._waypoints.list()]

    def add_waypoint(self, name: str, note: str = "") -> dict:
        if not name or not name.strip():
            raise ValueError("waypoint name required")
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
    # world management (list / delete saved maps)
    # ------------------------------------------------------------------
    def update_settings(self, *, tick_hz: float | None = None,
                        force_run: bool | None = None,
                        manual_wall_distance: float | None = None,
                        manual_wall_ratio: float | None = None) -> dict:
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
            if manual_wall_distance is not None:
                self.manual_wall_distance = max(
                    0.02, min(float(manual_wall_distance), 2.0))
            if manual_wall_ratio is not None:
                self.manual_wall_ratio = max(
                    0.0, min(float(manual_wall_ratio), 1.0))
            return {
                "tick_hz": self._tick_hz,
                "force_run": self._force_run,
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
        random cell across the map and 'pathfind' to there."""
        if self._last_pose is None:
            return {"found": False, "reason": "no current pose"}
        # snap radius: 4m for start (we're standing IN the cell so it
        # should be very close), wider 6m for goal so a waypoint dropped
        # a bit off a mapped corridor still snaps in.
        start_node = self._nav.graph.find_closest(
            self._last_pose.x, self._last_pose.y, self._last_pose.z,
            max_distance=4.0,
        )
        if start_node is None:
            return {"found": False,
                    "reason": "your current position isnt on the map yet, "
                              "walk around to map this area first"}
        goal_node = self._nav.graph.find_closest(gx, gy, gz,
                                                  max_distance=6.0)
        if goal_node is None:
            return {"found": False,
                    "reason": "no mapped reachable cell near the goal -- "
                              "the waypoint is in an unmapped area"}
        result: VoxelPathResult = find_path_astar(
            self._nav.graph, start_node.serial, goal_node.serial,
        )
        if not result.found:
            return {"found": False, "reason": "no path"}
        return {
            "found": True,
            "start": list(start_node.serial),
            "goal": list(goal_node.serial),
            "full": [list(s) for s in result.full_serials],
            "filtered": [list(s) for s in result.serials],
            "cost": result.cost,
            "expanded": result.nodes_expanded,
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
        """Make sure an explorer exists and is started, but DO NOT flip the
        public explore_enabled flag, so frontier-exploration stays off."""
        if not self._running:
            raise RuntimeError("mapping not running")
        if self._explorer is None:
            self._explorer = VoxelExplorer(self._nav, self._osc,
                                            learning_mode=True)
            self._explorer.force_run = self._force_run
            self._explorer.start()
            # mark this explorer as follow-only ONLY if the user hasnt
            # explicitly enabled frontier exploration. otherwise leave it
            # alone so explore-mode stays alive after the goto finishes.
            if not self._explore_enabled:
                self._explorer_follow_only = True
            logger.info("mapping: explorer spun up for path-follow")

    def goto_xyz(self, gx: float, gy: float, gz: float,
                 *, label: str = "") -> dict:
        """A* from current pose to (gx,gy,gz), then drive there via OSC."""
        with self._lock:
            preview = self.pathfind_to(gx, gy, gz)
            if not preview.get("found"):
                return preview
            self._ensure_explorer_for_follow()
            # build cell-serial list -- include both full + filtered path
            full = preview.get("full") or []
            cells: list[tuple[int, int, int]] = [tuple(s) for s in full]  # type: ignore
            try:
                self._explorer.follow_path(cells, label=label or "goto")
            except Exception as exc:
                logger.exception("mapping: follow_path failed")
                return {"found": False, "reason": f"follow failed: {exc}"}
            preview["driving"] = True
            preview["label"] = label or "goto"
            return preview

    def goto_waypoint(self, name: str) -> dict:
        self._ensure_waypoints(self._world_id)
        wp = self._waypoints.get(name)
        if wp is None:
            return {"found": False, "reason": f"waypoint '{name}' not found"}
        # remember the saved facing so we can rotate to it once we arrive.
        # cleared again on cancel/stop or when alignment completes.
        with self._lock:
            self._pending_align_yaw = float(wp.yaw)
        return self.goto_xyz(wp.x, wp.y, wp.z, label=f"wp:{name}")

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
