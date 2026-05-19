"""reference-style trail explorer.

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
from dataclasses import dataclass
from typing import Optional

from src.voxel_nav import (
    NodeType, Serial, VoxelNavManager, find_path_astar, serial_to_center,
)

logger = logging.getLogger(__name__)


@dataclass
class ExplorerState:
    target: Optional[Serial] = None
    target_source: Optional[Serial] = None     # the Reachable node we
                                               # discovered the target from
    e_count: float = 0.0
    last_distance: float = math.inf
    last_cell: Optional[Serial] = None         # for no-progress watchdog
    last_progress_t: float = 0.0
    action: str = "idle"


class VoxelExplorer:
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
    # path-follow mode (drive-to-waypoint)
    # ----------------------------------------------------------------------
    def follow_path(self, serials: list[Serial], *, label: str = "") -> None:
        """Drive along the given cell sequence. Replaces any current target.
        The explorer must be active; if not, start() is called first."""
        with self._lock:
            if not self._active:
                self.start()
            self._path_queue = list(serials)
            self._follow_active = True
            self._follow_label = label or ""
            # last cell of the queue is treated as the goal for replans
            self._follow_goal = serials[-1] if serials else None
            self._follow_replans = 0
            s = self.state
            s.target = None
            s.target_source = None
            s.e_count = 0.0
            s.last_distance = math.inf
            s.last_cell = None
            s.last_progress_t = time.time()
            self._ec_multiplier = 1.0
            logger.info("voxel_explorer: follow path label=%r len=%d",
                        self._follow_label, len(self._path_queue))

    def cancel_follow(self) -> None:
        with self._lock:
            if not self._follow_active:
                return
            self._follow_active = False
            self._path_queue.clear()
            self._follow_goal = None
            self._follow_replans = 0
            self.state.target = None
            self._send_osc(0.0, 0.0, run=False)
            self.state.action = "follow_cancel"
            logger.info("voxel_explorer: follow cancelled")

    @property
    def follow_status(self) -> dict:
        return {
            "active": self._follow_active,
            "remaining": len(self._path_queue),
            "label": self._follow_label,
        }

    # ----------------------------------------------------------------------
    # main tick
    # ----------------------------------------------------------------------
    def tick(self, pose_x: float, pose_y: float, pose_z: float,
             pose_yaw_deg: float) -> None:
        if not self._active:
            return
        s = self.state

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
                    self._send_osc(0.0, 0.0, run=False)
                    s.action = "follow_done"
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

        run = mag >= 2.0 or bool(getattr(self, "force_run", False))
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
        stuck_for = now - s.last_progress_t

        s.action = (f"walk d={mag:.2f} e={s.e_count:.0f} "
                    f"f={forward:.2f} t={turn:+.2f} stuck={stuck_for:.1f}")

        if s.e_count > self.E_COUNT_GIVE_UP:
            self._give_up_target("e_count")
        elif stuck_for > self.NO_PROGRESS_TIMEOUT:
            self._give_up_target(f"no_xz_progress_{stuck_for:.1f}s")
            s.last_progress_t = time.time()

        self._last_pose = (pose_x, pose_z, fx, fz)

    # ----------------------------------------------------------------------
    # CheckImpeded -- reference Wander.CheckImpeded
    # ----------------------------------------------------------------------
    def _check_impeded(self, forward: float, turn: float,
                       pose_x: float, pose_z: float,
                       fx: float, fz: float,
                       current_serial: Serial) -> bool:
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

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def _choose_target(self, current, forward_xz: tuple[float, float]) -> None:
        s = self.state
        cand = self.nav.choose_discovery_target(current, forward_xz)
        if cand is not None:
            s.target = cand
            s.target_source = current.serial
            s.e_count = 0.0
            s.last_distance = math.inf
            s.last_cell = None
            s.last_progress_t = time.time()
            logger.info("voxel_explorer: discover cardinal %s from %s",
                        cand, current.serial)
            return
        # prune expired abandons before asking the nav for a stack pick
        now_m = time.monotonic()
        if self._abandoned:
            self._abandoned = {c: t for c, t in self._abandoned.items() if t > now_m}
        stack = self.nav.check_stack(forward_xz,
                                     blacklist=set(self._abandoned.keys()) or None)
        if stack is None:
            logger.info("voxel_explorer: no unexplored cells remain")
            return
        cand, src = stack
        # if the source is current we can just walk
        if src.serial == current.serial:
            s.target = cand
            s.target_source = current.serial
            s.e_count = 0.0
            s.last_distance = math.inf
            s.last_cell = None
            s.last_progress_t = time.time()
            logger.info("voxel_explorer: discover stack target %s via %s",
                        cand, src.serial)
            return
        # route through the graph instead of walking in a straight line.
        # without this we tried to head straight toward a target that might
        # be on a totally different floor and just smashed into walls.
        # mirrors reference NodeManager.CheckStack which always SetPaths to
        # the source node before chasing the cardinal.
        pr = find_path_astar(self.nav.graph, current.serial, src.serial)
        if not pr.found or not pr.full_serials:
            logger.info("voxel_explorer: no graph path to %s for target %s, "
                        "blacklisting", src.serial, cand)
            self._abandoned[cand] = time.monotonic() + self._abandon_ttl
            return
        # queue = every hop after current, then the cardinal unexplored cell
        # as the final step
        queue: list[Serial] = list(pr.full_serials[1:]) + [cand]
        self._path_queue = queue
        self._follow_active = True
        self._follow_label = f"explore -> {cand}"
        if not self._advance_follow_queue(current.serial):
            # _advance_follow_queue couldnt find a valid next cell (everything
            # too high to climb maybe). abandon and move on.
            self._follow_active = False
            self._path_queue.clear()
            self._abandoned[cand] = time.monotonic() + self._abandon_ttl
            logger.info("voxel_explorer: stack route to %s had no climbable "
                        "step, blacklisting", cand)
            return
        s.e_count = 0.0
        s.last_distance = math.inf
        s.last_cell = None
        s.last_progress_t = time.time()
        logger.info("voxel_explorer: route to stack target %s via %d hops "
                    "through %s", cand, len(queue), src.serial)

    def _advance_follow_queue(self, current_serial: Serial) -> bool:
        """Pop cells off the follow queue until we find one we should actually
        drive toward. Skips cells we're already in (same XZ column), and
        cells that are too high above us to climb (likely a wall, not a
        step). Sets state.target and returns True on success, False if the
        queue is exhausted."""
        s = self.state
        while self._path_queue:
            nxt = self._path_queue.pop(0)
            same_col = (current_serial[0] == nxt[0]
                        and current_serial[2] == nxt[2])
            if same_col or self.nav.bar_check(current_serial, nxt):
                continue
            # too high to climb? skip it. dont skip downward steps because
            # falling is fine.
            if nxt[1] - current_serial[1] > self.FOLLOW_MAX_CLIMB:
                logger.info("voxel_explorer: skipping follow cell %s "
                            "(too high above current %s)",
                            nxt, current_serial)
                continue
            s.target = nxt
            s.target_source = current_serial
            return True
        s.target = None
        s.target_source = None
        return False

    def _give_up_target(self, why: str) -> None:
        s = self.state
        failed = s.target
        cur = self.nav.current
        if failed is not None and self.learning_mode:
            is_neighbor = (cur is not None
                           and self.nav.is_pathable_neighbor(cur.serial, failed))
            # follow mode + still have a goal to reach: just demote the
            # blocked cell to Iffy so the replanner avoids it. dont commit
            # to a full wall since it might be reachable from another angle.
            if self._follow_active and self._follow_goal is not None:
                if is_neighbor:
                    logger.info("voxel_explorer: marking %s Iffy mid-follow "
                                "(%s)", failed, why)
                    self.nav.mark_iffy(failed)
                else:
                    logger.info("voxel_explorer: abandon distant follow cell "
                                "%s (%s), no wall mark", failed, why)
                    self._abandoned[failed] = time.monotonic() + self._abandon_ttl
            elif is_neighbor:
                logger.info("voxel_explorer: marking %s UnReachable (%s)",
                            failed, why)
                self.nav.mark_unreachable(failed)
            else:
                logger.info("voxel_explorer: abandon distant target %s (%s) "
                            "without wall mark", failed, why)
                self._abandoned[failed] = time.monotonic() + self._abandon_ttl
        s.target = None
        s.target_source = None
        s.e_count = 0.0
        s.last_distance = math.inf
        s.last_cell = None
        # try to replan in follow mode rather than just blindly walking into
        # the next queued cell (which is probably behind the same obstacle).
        if (self._follow_active and self._follow_goal is not None
                and cur is not None):
            if self._follow_replans >= self._follow_replan_limit:
                logger.warning("voxel_explorer: follow replan limit hit (%d), "
                               "cancelling follow %r",
                               self._follow_replan_limit, self._follow_label)
                self._follow_active = False
                self._path_queue.clear()
                self._follow_goal = None
                self._follow_replans = 0
            else:
                self._follow_replans += 1
                pr = find_path_astar(self.nav.graph, cur.serial,
                                     self._follow_goal)
                if pr.found and len(pr.full_serials) > 1:
                    self._path_queue = list(pr.full_serials[1:])
                    s.last_progress_t = time.time()
                    logger.info("voxel_explorer: follow replan #%d ok, "
                                "%d cells to goal %s",
                                self._follow_replans,
                                len(self._path_queue), self._follow_goal)
                else:
                    logger.warning("voxel_explorer: follow replan #%d failed, "
                                   "no path from %s to %s, cancelling",
                                   self._follow_replans, cur.serial,
                                   self._follow_goal)
                    self._follow_active = False
                    self._path_queue.clear()
                    self._follow_goal = None
                    self._follow_replans = 0
        self._send_osc(0.0, 0.0, run=False)

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
