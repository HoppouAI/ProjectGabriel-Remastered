"""Manual mapping mode -- user drives, we listen to the forward raycast."""

from __future__ import annotations

import logging
import math
import time

from src.pose_decoder import WorldPose
from src.voxel_nav import serial_to_center, world_to_serial

logger = logging.getLogger(__name__)


class ManualMappingMixin:
    """Toggle + per-tick wall detect + grid-lock yaw/strafe assist."""

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
