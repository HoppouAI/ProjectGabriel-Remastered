"""reference-style trail explorer (main class).

Python port of the reference walk-to-target + discovery target loop,
swapping the reference depth camera for our pose strip and using direct
OSC inputs (`/input/Vertical`, `/input/LookHorizontal`) to drive the
avatar.

Behavior (matches exactly):
    1. If no target: pick the cardinal cell in front of current Reachable
       node that has not been visited (and isnt right above/below a known
       cell). Falls back to scanning the whole graph for the closest
       unexplored cardinal of any Reachable node.
    2. Steer toward target center: turn until facing the cell, then walk
       forward. Run when far, walk when close.
    3. Track an "impede counter" eCount: every frame we are stuck without
       getting closer it ticks up. When it crosses a threshold, mark the
       target UnReachable (reference MarkTargetUnReachable) and pick a new one.
    4. When the current voxel matches the target (BarCheck): success,
       clear target.

The graph fills in passively because `VoxelNavManager.observe()` is being
called from the host loop with each pose tick.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

from src.voxel_nav import NodeType, Serial, VoxelNavManager, serial_to_center

from .follow import FollowMixin
from .motion import MotionMixin
from .raycast_assist import RaycastAssistMixin
from .state import ExplorerState
from .targeting import TargetingMixin


logger = logging.getLogger(__name__)


class VoxelExplorer(FollowMixin, TargetingMixin, MotionMixin, RaycastAssistMixin):
    """Drives the avatar via OSC to fill in `nav.graph` reference style.

    Call `tick(pose)` at ~20Hz from the same loop that calls
    `nav.observe(pose)`. The explorer assumes `osc.client.send_message`
    is the standard SimpleUDPClient.
    """

    # reference hardcoded constants from Wander.cs / NodeManager.cs
    FACING_THRESHOLD = 0.98          # dot(forward, to_target) past this = walk
    E_COUNT_GIVE_UP = 20             # eCount > this = target is UnReachable
    TURN_DEADZONE = 0.001            # |cross| <= this = no turn
    FOLLOW_MAX_CLIMB = 1             # skip queued cells more than this many
                                     # voxels above current (cant jump walls)
    # wallclock progress watchdog. reference relies on eCount > 20 which works at
    # their ~60Hz, but at our 20Hz with small targets the forward output sits
    # right at 0.1 and CheckImpeded gates out so eCount never grows. if we
    # have a target and our voxel cell hasnt changed this many seconds, we
    # give up the same way an eCount blowout would.
    NO_PROGRESS_TIMEOUT = 4.0

    def __init__(self, nav: VoxelNavManager, osc, *, learning_mode: bool = True):
        self.nav = nav
        self.osc = osc
        self.learning_mode = learning_mode
        self.force_run = False
        # movement speed mode: "walk" (half speed, no sprint),
        # "fast" (full vertical, no sprint, default normal walk),
        # "run" (full vertical + sprint). default fast so the AI moves
        # at normal walk speed instead of the old proportional crawl.
        self.speed_mode: str = "fast"
        # path-follow mode (used by drive-to-waypoint).
        # when active we step through _path_queue instead of asking the
        # nav manager for new exploration targets.
        self._path_queue: list[Serial] = []
        self._follow_active: bool = False
        self._follow_label: str = ""
        # original goal cell for the active follow. used to replan when
        # we get stuck mid-route, demoting the failed cell to Iffy and
        # re-routing around it.
        self._follow_goal: Optional[Serial] = None
        self._follow_replans: int = 0
        self._follow_replan_limit: int = 4
        # when the queue empties we optionally rotate to a saved facing
        # before going inactive (mirrors waypoint-mode in the ref impl).
        self._final_yaw_deg: Optional[float] = None
        self._aligning: bool = False
        self._align_start_t: float = 0.0
        self._align_timeout_s: float = 10.0
        self.state = ExplorerState()
        self._active = False
        self._last_send_forward = 0.0
        self._last_send_turn = 0.0
        self._last_send_run = False
        self._ec_multiplier = 1.0
        self._last_pose = None  # (x, z, fx, fz) for CheckImpeded
        self._lock = threading.RLock()
        # short-lived blacklist of distant targets we recently abandoned
        # without marking them as walls. keeps check_stack from immediately
        # re-picking the same dead-end cell every tick. cell -> expiry mono.
        self._abandoned: dict[Serial, float] = {}
        self._abandon_ttl = 30.0

    # ----------------------------------------------------------------------
    # lifecycle
    # ----------------------------------------------------------------------
    def start(self) -> None:
        self._active = True
        self.state = ExplorerState()
        self.state.last_progress_t = time.time()
        self._ec_multiplier = 1.0
        self._last_pose = None
        # force first OSC send by invalidating dedupe state
        self._last_send_forward = float("nan")
        self._last_send_turn = float("nan")
        self._last_send_run = None
        logger.info("voxel_explorer: started")

    def stop(self) -> None:
        self._active = False
        self._send_osc(0.0, 0.0, run=False)
        logger.info("voxel_explorer: stopped")

    # ----------------------------------------------------------------------
    # main tick
    # ----------------------------------------------------------------------
    def tick(self, pose_x: float, pose_y: float, pose_z: float,
             pose_yaw_deg: float) -> None:
        if not self._active:
            return
        s = self.state

        # aligning to a final facing after the path completed. runs before
        # everything else so we dont accidentally pick a new target while
        # we still have a heading to settle on.
        if self._aligning:
            timed_out = (time.time() - self._align_start_t) > self._align_timeout_s
            if timed_out:
                logger.warning("voxel_explorer: align timeout, giving up at yaw=%.1f target=%.1f",
                                pose_yaw_deg,
                                self._final_yaw_deg if self._final_yaw_deg is not None else 0.0)
            if timed_out or self._drive_final_yaw(pose_yaw_deg):
                self._aligning = False
                self._final_yaw_deg = None
                s.action = "aligned" if not timed_out else "align_timeout"
                self._send_osc(0.0, 0.0, run=False)
            return

        # forward XZ vector from yaw (decoder convention: 0deg=+Z, 90deg=+X)
        yaw_rad = math.radians(pose_yaw_deg)
        fx = math.sin(yaw_rad)
        fz = math.cos(yaw_rad)

        current = self.nav.current
        if current is None:
            # no pose lock yet, do nothing
            self._send_osc(0.0, 0.0, run=False)
            s.action = "wait_pose"
            self._last_pose = (pose_x, pose_z, fx, fz)
            return

        # check target reached. in follow mode we're more lenient: if the
        # target is in the same XZ column we're already in (any Y), treat it
        # as reached, because we cant physically walk to a cell directly
        # above/below us without changing X/Z first. otherwise we'd loop
        # forever trying to descend a step that the engine snaps us off of.
        if s.target is not None:
            reached = self.nav.bar_check(current.serial, s.target)
            if not reached and self._follow_active:
                if current.serial[0] == s.target[0] and current.serial[2] == s.target[2]:
                    reached = True
            if reached:
                logger.info("voxel_explorer: reached target %s", s.target)
                s.target = None
                s.target_source = None
                s.e_count = 0.0
                s.last_distance = math.inf
                self._ec_multiplier = 1.0
                s.action = "reached"
                # path-follow: pop the next waypoint cell off the queue,
                # skipping any cells we're already on top of in XZ.
                if self._follow_active:
                    if self._advance_follow_queue(current.serial):
                        s.last_progress_t = time.time()
                        s.action = f"follow next ({len(self._path_queue)} left)"
                    else:
                        self._follow_active = False
                        self._follow_goal = None
                        self._follow_replans = 0
                        if self._final_yaw_deg is not None:
                            self._aligning = True
                            self._align_start_t = time.time()
                            s.action = "aligning"
                            logger.info("voxel_explorer: follow done, aligning to %.1fdeg",
                                        self._final_yaw_deg)
                        else:
                            s.action = "follow_done"
                            logger.info("voxel_explorer: follow path complete (%s)",
                                        self._follow_label)
                        self._send_osc(0.0, 0.0, run=False)
                        self._last_pose = (pose_x, pose_z, fx, fz)
                        return

        # need a new target? only discover from a Reachable current.
        if s.target is None:
            if self._follow_active:
                # follow mode: try to grab the next cell from the queue
                # (e.g. we just gave up on a step that was unreachable).
                if self._advance_follow_queue(current.serial):
                    s.e_count = 0.0
                    s.last_distance = math.inf
                    s.last_cell = None
                    s.last_progress_t = time.time()
                    s.action = f"follow next ({len(self._path_queue)} left)"
                else:
                    self._follow_active = False
                    self._follow_goal = None
                    self._follow_replans = 0
                    if self._final_yaw_deg is not None:
                        self._aligning = True
                        self._align_start_t = time.time()
                        s.action = "aligning"
                        logger.info("voxel_explorer: follow done, aligning to %.1fdeg",
                                    self._final_yaw_deg)
                    else:
                        s.action = "follow_done"
                    self._send_osc(0.0, 0.0, run=False)
                    self._last_pose = (pose_x, pose_z, fx, fz)
                    return
            if s.target is None:
                if current.node_type != NodeType.REACHABLE:
                    self._send_osc(0.0, 0.0, run=False)
                    s.action = "wait_reachable"
                    self._last_pose = (pose_x, pose_z, fx, fz)
                    return
                self._choose_target(current, (fx, fz))
                if s.target is None:
                    self._send_osc(0.0, 0.0, run=False)
                    s.action = "no_target"
                    self._last_pose = (pose_x, pose_z, fx, fz)
                    return

        # --- DoMotion (1-1 with reference Wander.DoMotion) -------------------
        tx, _, tz = serial_to_center(s.target)
        dx = tx - pose_x
        dz = tz - pose_z
        mag = math.hypot(dx, dz)
        if mag < 1e-6:
            ndx, ndz = fx, fz
        else:
            ndx = dx / mag
            ndz = dz / mag

        # reference Utils.CrossProd on Vector2 = look.x*to.y - look.y*to.x.
        # they pack worldX -> .x, worldZ -> .y, so this is fx*ndz - fz*ndx.
        cross = fx * ndz - fz * ndx
        dot = fx * ndx + fz * ndz
        flag = dot > self.FACING_THRESHOLD

        if dot < 0.0:
            # behind us: hard turn one way
            cross = -1.0 if cross < 0 else 1.0
        elif flag:
            # nearly aligned: soften turn with cross^1.5 (signed)
            if cross < 0:
                cross = -1.0 * (abs(cross) ** 1.5)
            else:
                cross = cross ** 1.5

        if cross < -self.TURN_DEADZONE or cross > self.TURN_DEADZONE:
            # reference keeps a minimum turn magnitude of 0.5 to defeat deadzone
            if cross < 0:
                turn = 0.5 - 0.5 * cross
            else:
                turn = -0.5 - 0.5 * cross
        else:
            turn = 0.0

        fwd_scale = min(mag * 0.75, 1.0) * 0.9 + 0.1
        if flag:
            forward = max(0.0, min(fwd_scale * dot, 1.0))
        else:
            forward = 0.0

        # raycast-assisted smoothing: only while following an A* path so the
        # avatar flows around door frames/furniture instead of ramming them
        # and stalling into a replan. frontier mapping deliberately skips this
        # so it can still approach geometry to learn it.
        ra_clearance = None
        if self._follow_active:
            turn, forward, ra_clearance = self._apply_raycast_assist(turn, forward, mag)

        # speed mode controls Run (sprint) + walk-mode dampener. we keep
        # the proportional forward scaling so the avatar doesnt overshoot
        # short single-cell hops and spin around to recover.
        mode = (self.speed_mode or "fast").lower()
        if mode == "walk":
            forward = forward * 0.5
            run = False
        elif mode == "run":
            run = True
        else:  # "fast" (default)
            run = mag >= 2.0 or bool(getattr(self, "force_run", False))
        # dont sprint straight into something close even if the cell is far
        if ra_clearance is not None and ra_clearance < self.RA_RUN_MIN_CLEAR:
            run = False
        self._send_osc(forward, turn, run=run)

        # --- eCount bookkeeping (reference WalkToTarget tail) ----------------
        if flag:
            if mag > s.last_distance:
                s.e_count += self._ec_multiplier * 2.0
            if self._check_impeded(forward, turn, pose_x, pose_z, fx, fz, current.serial):
                s.e_count += self._ec_multiplier
            else:
                s.e_count = max(0.0, s.e_count - 1.0)
        s.last_distance = mag

        # wallclock no-progress watchdog. update progress timer whenever our
        # current voxel cell changes (real movement between voxels), otherwise
        # fire give_up when we've been frozen in the same cell too long.
        # this catches the case reference misses where forward output sits at the
        # 0.1 CheckImpeded threshold and eCount never accumulates.
        now = time.time()
        cur_serial = current.serial
        if s.last_cell is None or s.last_cell != cur_serial:
            s.last_cell = cur_serial  # reuse field, stores serial now
            s.last_progress_t = now
            # we actually moved between voxels, so the perch-stuck counter
            # should reset; the next give-up here is unrelated to the last.
            s.consec_giveups_in_cell = 0
            s.giveup_cell = None
        stuck_for = now - s.last_progress_t

        s.action = (f"walk d={mag:.2f} e={s.e_count:.0f} "
                    f"f={forward:.2f} t={turn:+.2f} stuck={stuck_for:.1f}")

        if s.e_count > self.E_COUNT_GIVE_UP:
            self._give_up_target("e_count")
        elif stuck_for > self.NO_PROGRESS_TIMEOUT:
            self._give_up_target(f"no_xz_progress_{stuck_for:.1f}s")
            s.last_progress_t = time.time()

        self._last_pose = (pose_x, pose_z, fx, fz)
