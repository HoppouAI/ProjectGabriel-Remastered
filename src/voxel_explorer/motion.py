"""OSC drive + low level motion helpers (CheckImpeded, align, send_osc)."""

from __future__ import annotations

import logging

from src.voxel_nav import Serial


logger = logging.getLogger(__name__)


class MotionMixin:
    """Methods that touch OSC inputs or compare poses tick-to-tick.

    Relies on attributes initialized by `VoxelExplorer.__init__`:
        self.osc, self.state, self._last_pose, self._final_yaw_deg,
        self._last_send_forward, self._last_send_turn, self._last_send_run.
    """

    def _check_impeded(self, forward: float, turn: float,
                       pose_x: float, pose_z: float,
                       fx: float, fz: float,
                       current_serial: Serial) -> bool:
        """reference Wander.CheckImpeded: we're stuck if we asked for motion
        but our pose hasnt budged since the last tick."""
        s = self.state
        if s.target is None or current_serial == s.target:
            return False
        # not actually trying to move
        if forward < 0.1 and -0.5 < turn < 0.5:
            return False
        if self._last_pose is None:
            return False
        lx, lz, lfx, lfz = self._last_pose
        if lx != pose_x or lz != pose_z:
            return False
        # reference tolerates 0.5 wobble on forward vector before declaring stuck
        if lfx < fx - 0.5 or lfx > fx + 0.5:
            return False
        if lfz < fz - 0.5 or lfz > fz + 0.5:
            return False
        return True

    def _drive_final_yaw(self, pose_yaw_deg: float) -> bool:
        """Rotate toward `_final_yaw_deg` via LookHorizontal. Returns True
        once we're inside the deadband (caller should release)."""
        target = self._final_yaw_deg
        if target is None:
            return True
        delta = (target - pose_yaw_deg + 540.0) % 360.0 - 180.0
        if abs(delta) <= 3.0:
            return True
        sign = 1.0 if delta > 0 else -1.0
        # bigger floor than the old mapping-side aligner so vrchat actually
        # registers the turn at small angles. ramps up to 0.6 by ~45deg.
        mag = min(0.6, max(0.18, abs(delta) / 45.0))
        # push lookhorizontal every tick (not just on change) so face tracker
        # or other systems writing 0 to lookhorizontal cant strand us.
        try:
            c = self.osc.client
            c.send_message("/input/Vertical", 0.0)
            c.send_message("/input/LookHorizontal", float(sign * mag))
            c.send_message("/input/Run", 0)
            self._last_send_forward = 0.0
            self._last_send_turn = sign * mag
            self._last_send_run = False
        except Exception:
            logger.exception("voxel_explorer: align OSC send failed")
        return False

    def _send_osc(self, forward: float, turn: float, run: bool) -> None:
        # reference SendOSC: dedupe each channel against last value
        c = self.osc.client
        forward = max(-1.0, min(1.0, forward))
        turn = max(-1.0, min(1.0, turn))
        if forward != self._last_send_forward:
            c.send_message("/input/Vertical", float(forward))
            self._last_send_forward = forward
        if turn != self._last_send_turn:
            c.send_message("/input/LookHorizontal", float(turn))
            self._last_send_turn = turn
        if run != self._last_send_run:
            c.send_message("/input/Run", 1 if run else 0)
            self._last_send_run = run
