"""Raycast-assisted steering for drive-to-waypoint path follow.

The A* path tells us WHERE to go (a sequence of reachable voxel cells), but
driving straight at each cell center with no obstacle sense makes the avatar
ram door frames / furniture, stall, blow the eCount watchdog, and replan. That
stop-replan churn is what makes directed navigation feel sluggish compared to
the reactive wanderer.

This mixin blends the engine-truth VRCRaycast distances (same sensors the
wanderer steers from) onto the toward-target drive: damp forward speed as we
close on something ahead, steer toward the more open side, and never sprint
into a near wall. It only runs during an active path-follow so frontier
mapping can still bump up against geometry to learn it.
"""

from __future__ import annotations


class RaycastAssistMixin:
    # master toggle, overridable per-instance
    ra_enabled: bool = True

    # below this forward clearance (m) halt forward and hard-steer out
    RA_STOP_DIST = 0.7
    # start damping forward below this clearance (m)
    RA_SLOW_DIST = 2.0
    # side "open" reference distance (m); a no-hit side counts as this
    RA_SIDE_REF = 3.0
    # max avoidance turn we add on top of the toward-target turn
    RA_AVOID_GAIN = 0.9
    # dont sprint when forward clearance is under this (m)
    RA_RUN_MIN_CLEAR = 2.5
    # within this distance of the target cell, stop steering for avoidance so
    # we can actually home in on waypoints that sit near a wall
    RA_NEAR_TARGET_RELAX = 1.5

    def _ra_ray(self, name):
        state = getattr(self.osc, "raycast_state", None) if self.osc else None
        if state is None:
            return None
        r = state.get(name)
        if r is None or not r.is_fresh():
            return None
        return r

    def _ra_forward_clearance(self):
        """Forward clearance in meters. Prefers FwdNear (hip ray) for the
        stop decision, falls back to Fwd (head ray). None if no ray data."""
        near = self._ra_ray("FwdNear")
        head = self._ra_ray("Fwd")
        if near is not None and near.hit:
            return near.distance
        if head is not None and head.hit:
            return head.distance
        if near is not None or head is not None:
            if head is not None:
                return max(head.distance, 5.0)
            return max(near.distance, 1.5)
        return None

    def _ra_side(self, name):
        r = self._ra_ray(name)
        if r is None:
            return None
        if not r.hit:
            return max(r.distance, self.RA_SIDE_REF)
        return r.distance

    def _ra_side_scores(self):
        ref = self.RA_SIDE_REF
        left_vals = [v for v in (self._ra_side("LeftFwd"), self._ra_side("Left"))
                     if v is not None]
        right_vals = [v for v in (self._ra_side("RightFwd"), self._ra_side("Right"))
                      if v is not None]
        left_score = min(left_vals) if left_vals else ref
        right_score = min(right_vals) if right_vals else ref
        return left_score, right_score

    def _apply_raycast_assist(self, turn, forward, mag):
        """Blend reactive raycast avoidance onto the toward-target drive.

        Returns (turn, forward, clearance). clearance is None when no ray
        data is available (so the caller leaves the run decision alone).
        Positive turn = turn right, matching the toward-target convention.
        """
        if not self.ra_enabled:
            return turn, forward, None
        clearance = self._ra_forward_clearance()
        if clearance is None:
            return turn, forward, None
        # close to the goal: let the base homing logic land it, just report
        # clearance so we still avoid sprinting the last meter.
        if mag <= self.RA_NEAR_TARGET_RELAX:
            return turn, forward, clearance
        if clearance >= self.RA_SLOW_DIST:
            return turn, forward, clearance

        left_score, right_score = self._ra_side_scores()
        span = max(self.RA_SLOW_DIST - self.RA_STOP_DIST, 0.1)
        damp = max(0.0, min(1.0, (clearance - self.RA_STOP_DIST) / span))
        forward = forward * (0.2 + 0.8 * damp)

        denom = max(left_score + right_score, 0.1)
        gradient = (right_score - left_score) / denom  # + => right is open
        closeness = 1.0 - damp  # 0 far .. 1 at stop dist
        avoid = gradient * self.RA_AVOID_GAIN * closeness
        turn = max(-1.0, min(1.0, turn + avoid))

        if clearance < self.RA_STOP_DIST:
            forward = min(forward, 0.0)
            side_dir = 1.0 if right_score >= left_score else -1.0
            turn = side_dir * max(abs(turn), 0.6)

        return turn, forward, clearance
