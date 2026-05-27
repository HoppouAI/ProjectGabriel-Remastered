"""Reactive steering decision: continuous gradient + smart escapes."""

from __future__ import annotations

import random
import time


class DecisionMixin:
    def _decide(self):
        cfg = self._cfg
        now = time.monotonic()
        ref = cfg["side_reference"]

        clearance = self._forward_clearance()
        drop = self._drop_ahead()
        # forward-cone (45deg) and pure side rays
        leftfwd_d = self._side_distance("LeftFwd", default=ref)
        rightfwd_d = self._side_distance("RightFwd", default=ref)
        left_d = self._side_distance("Left", default=ref)
        right_d = self._side_distance("Right", default=ref)
        back_r = self._ray("Back")
        back_clear = (back_r is None) or (not back_r.hit) or (back_r.distance > 0.8)

        # rolling clearance history (last 1.0s) for closing-rate estimate
        if clearance is not None:
            self._clearance_history.append((now, clearance))
        cutoff = now - 1.0
        self._clearance_history = [x for x in self._clearance_history if x[0] >= cutoff]
        closing_rate = 0.0  # meters per second of clearance shrinking
        if len(self._clearance_history) >= 2:
            t0, c0 = self._clearance_history[0]
            t1, c1 = self._clearance_history[-1]
            dt = max(t1 - t0, 0.05)
            closing_rate = max(0.0, (c0 - c1) / dt)

        # combined side scores (lower of pure-side and forward-side per side)
        left_score = min(leftfwd_d, left_d)
        right_score = min(rightfwd_d, right_d)

        # escalation: drop old wall hits
        self._recent_wall_hits = [
            t for t in self._recent_wall_hits if t >= now - cfg["escalation_window"]
        ]

        # ---- dead-end detection ----
        dead_end = (
            clearance is not None
            and clearance < cfg["deadend_distance"]
            and left_score < cfg["deadend_distance"]
            and right_score < cfg["deadend_distance"]
        )
        # escalation overrides regular wall mode with a u-turn
        escalated = len(self._recent_wall_hits) >= cfg["escalation_threshold"]

        # ---- hard overrides first ----
        if drop:
            if now > self._committed_turn_until:
                self._stuck_turn_dir = -1.0 if left_score > right_score else 1.0
                self._committed_turn_dir = self._stuck_turn_dir
                self._committed_turn_until = now + cfg["ledge_commit_seconds"]
            target_turn = self._committed_turn_dir * cfg["turn_speed_avoid"]
            target_forward = -0.1 if back_clear else 0.0
            action = "ledge"

        elif dead_end or (
            escalated
            and clearance is not None
            and clearance < cfg["slow_distance"]
        ):
            # u-turn: hold a single direction long enough to spin ~180
            if now > self._committed_turn_until:
                self._stuck_turn_dir = -1.0 if left_score > right_score else 1.0
                self._committed_turn_dir = self._stuck_turn_dir
                self._committed_turn_until = now + cfg["uturn_commit_seconds"]
                self._recent_wall_hits.clear()
            target_turn = self._committed_turn_dir * cfg["turn_speed_avoid"]
            target_forward = 0.0  # spin in place
            action = "uturn"

        elif clearance is not None and clearance < cfg["stop_distance"]:
            if now > self._committed_turn_until:
                self._stuck_turn_dir = -1.0 if left_score > right_score else 1.0
                self._committed_turn_dir = self._stuck_turn_dir
                self._committed_turn_until = now + cfg["wall_commit_seconds"]
                self._recent_wall_hits.append(now)
            target_turn = self._committed_turn_dir * cfg["turn_speed_avoid"]
            # only back up if back is clear, otherwise just turn in place
            target_forward = -0.3 if back_clear else 0.0
            action = "wall"

        else:
            # continuous gradient: speed scales with forward clearance,
            # turn scales with side imbalance plus a closer-side push-away
            if clearance is None:
                fwd_norm = 1.0
            else:
                span = max(cfg["cruise_distance"] - cfg["stop_distance"], 0.1)
                fwd_norm = max(0.0, min(1.0, (clearance - cfg["stop_distance"]) / span))
            # predictive brake: if clearance is closing fast, slow more
            if closing_rate > 0.5:
                fwd_norm *= max(0.3, 1.0 - closing_rate * 0.3)
            target_forward = cfg["forward_speed"] * (0.25 + 0.75 * fwd_norm)

            denom = max(left_score + right_score, 0.1)
            gradient = (right_score - left_score) / denom  # +1 means right is open
            closer = min(left_score, right_score)
            closeness = max(0.0, min(1.0, 1.0 - closer / ref))
            steer_strength = closeness ** 0.7
            target_turn = gradient * cfg["turn_speed_steer"] * steer_strength

            # predictive turn bump: if closing fast on something, steer harder
            if closing_rate > 0.8 and abs(gradient) > 0.05:
                target_turn *= 1.0 + min(1.0, (closing_rate - 0.8) * 0.7)

            if clearance is not None and clearance < cfg["slow_distance"]:
                action = "approach" if abs(target_turn) < 0.15 else "veer"
            else:
                action = "walking"

            wide_open = (
                clearance is None or clearance >= cfg["cruise_distance"]
            ) and closeness < 0.2
            if wide_open:
                straight_for = now - self._last_straight_time
                force_turn = straight_for > cfg["max_straight_time"]
                do_random = (
                    straight_for > cfg["min_straight_time"]
                    and random.random() < cfg["random_turn_chance"]
                )
                if force_turn or do_random:
                    direction = random.choice([-1.0, 1.0])
                    target_turn = direction * random.uniform(0.4, cfg["turn_speed_random"])
                    self._committed_turn_dir = direction
                    self._committed_turn_until = now + random.uniform(0.6, 1.2)
                    self._last_straight_time = now
                    action = "explore"
            else:
                if abs(target_turn) > 0.15:
                    self._last_straight_time = now

        # honor commit window: do not flip turn direction mid-escape
        if now < self._committed_turn_until and self._committed_turn_dir != 0.0:
            if (target_turn > 0) != (self._committed_turn_dir > 0) or target_turn == 0.0:
                target_turn = self._committed_turn_dir * max(
                    abs(target_turn), cfg["turn_speed_steer"]
                )

        # velocity-based stuck detection (only while trying to walk forward)
        velocity_stuck = False
        if (
            self.osc is not None
            and getattr(self.osc, "velocity_received", False)
            and target_forward > 0.2
            and action in ("walking", "approach", "veer", "explore")
        ):
            vel_z = abs(self.osc.velocity_z)
            if vel_z < cfg["stuck_velocity_threshold"]:
                self._stuck_frames += 1
            else:
                self._stuck_frames = 0

            if self._stuck_frames >= cfg["stuck_frames_to_reverse"]:
                if self._stuck_frames == cfg["stuck_frames_to_reverse"]:
                    self._stuck_turn_dir = random.choice([-1.0, 1.0])
                target_turn = self._stuck_turn_dir * cfg["turn_speed_avoid"]
                target_forward = -0.3
                action = "stuck"
                velocity_stuck = True
                if self._stuck_frames >= cfg["stuck_frames_to_jump"]:
                    self._do_jump()
                    self._stuck_frames = 0
        else:
            self._stuck_frames = 0

        # smoothing -- instant for hard avoidance, EMA for cruise
        alpha = cfg["smoothing_alpha"]
        sharp = action in ("wall", "ledge", "uturn", "stuck", "explore") or velocity_stuck
        if sharp:
            self._smoothed_turn = target_turn
        else:
            self._smoothed_turn = alpha * target_turn + (1 - alpha) * self._smoothed_turn
        self._smoothed_forward = alpha * target_forward + (1 - alpha) * self._smoothed_forward

        if action != "walking":
            self._last_straight_time = now
        self._current_action = action
        return self._smoothed_turn, self._smoothed_forward
