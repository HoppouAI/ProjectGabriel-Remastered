"""Start / stop / world-swap + the tick loop."""

from __future__ import annotations

import logging
import threading
import time

from src.pose_decoder import GRID_W, GRID_H, PoseExfilReader
from src.voxel_explorer import VoxelExplorer

from ._base import _RegionGuess

logger = logging.getLogger(__name__)


class LifecycleMixin:
    """start/stop, explore toggle, world-change handler, the tick thread."""

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
        last_gap_fill = time.time()
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
                # periodic interior hole-fill while actively exploring. the
                # passive footstep map always speckles the floor with single
                # cell gaps; close them every 20s so coverage looks solid and
                # the frontier picker stops chasing un-walkable interior holes.
                if (self._explore_enabled and self._explorer is not None
                        and now - last_gap_fill >= 20.0):
                    last_gap_fill = now
                    try:
                        filled = self._nav.fill_interior_gaps()
                        if filled:
                            logger.info("mapping: hole-fill closed %d interior "
                                        "gap cells", filled)
                    except Exception:
                        logger.exception("mapping: hole-fill failed")
            except Exception:
                logger.exception("mapping: tick failed")
            interval = 1.0 / max(1.0, min(self._tick_hz, 120.0))
            time.sleep(interval)
        # final flush
        try:
            self._nav.flush()
        except Exception:
            pass
