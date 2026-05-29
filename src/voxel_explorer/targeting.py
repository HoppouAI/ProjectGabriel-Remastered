"""Target selection + give-up handling. Wraps nav's discovery helpers and
the short-lived `_abandoned` blacklist."""

from __future__ import annotations

import logging
import math
import time

from src.voxel_nav import NodeType, find_path_astar


logger = logging.getLogger(__name__)


class TargetingMixin:
    """Pick the next exploration target and handle abandoning a stuck one.

    Relies on attributes initialized by `VoxelExplorer.__init__`:
        self.nav, self.state, self.learning_mode, self._follow_active,
        self._follow_goal, self._follow_replans, self._follow_replan_limit,
        self._follow_label, self._path_queue, self._abandoned,
        self._abandon_ttl.
    Calls self._advance_follow_queue() and self._send_osc().
    """

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
        queue: list = list(pr.full_serials[1:]) + [cand]
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
                if pr.found and (pr.serials or len(pr.full_serials) > 1):
                    # filtered turn-points first (smooth, fewer stalls), fall
                    # back to the full cell list when filtering emptied it.
                    self._path_queue = list(pr.serials) if pr.serials \
                        else list(pr.full_serials[1:])
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
