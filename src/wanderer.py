"""Autonomous wandering using avatar-mounted VRCRaycasts.

Replaces the old depth-model based wanderer. Reads engine-truth distance
readings published by VRCRaycast components on the avatar (see
unity_assets/AVATAR_SETUP.md) via VRChatOSC.raycast_state and steers
purely from those, no screen capture and no neural net.

Public API (kept stable for callers):
    Wanderer(config, osc=None)
    .active                  -> bool property
    .preload()               -> no-op now, kept so main.py doesnt break
    .start() / .stop()       -> dict result
    .pause() / .resume()
    .on_speech_activity()
    .on_ai_speaking()
    ._face_tracker_ref       -> assigned by main.py
    ._emotion_system_ref     -> assigned by main.py
"""

from __future__ import annotations

import logging
import random
import threading
import time

logger = logging.getLogger(__name__)

TARGET_FPS = 20  # raycasts come in at avatar tick rate, we can poll fast

DEFAULT_CFG = {
    # forward speed when path is clear
    "forward_speed": 0.6,
    # turn rates
    "turn_speed_avoid": 1.0,    # max turn rate when escaping a wall
    "turn_speed_steer": 0.6,    # max gradient-bias turn while moving
    "turn_speed_random": 0.6,   # exploration turns

    # raycast clearance thresholds (meters)
    "stop_distance": 0.8,       # below this on FwdNear -> reverse + commit turn
    "slow_distance": 2.0,       # start scaling speed down below this
    "cruise_distance": 3.5,     # full speed above this on Fwd
    "side_reference": 3.0,      # side distances are normalized against this

    # how long to stay in dedicated escape modes (seconds)
    "wall_commit_seconds": 1.4,
    "ledge_commit_seconds": 1.0,
    "uturn_commit_seconds": 2.5,    # full ~180 turn when dead-ended

    # dead-end detection (all forward-cone rays below this -> u-turn)
    "deadend_distance": 1.4,
    # escalation: repeated wall hits in this window promote to u-turn
    "escalation_window": 6.0,
    "escalation_threshold": 2,      # this many walls in window -> u-turn

    # exploration behavior
    "random_turn_chance": 0.02,
    "min_straight_time": 12.0,
    "max_straight_time": 25.0,
    "jump_chance": 0.012,

    # stuck detection
    "stuck_velocity_threshold": 0.05,
    "stuck_frames_to_reverse": 10,  # ~0.5s at 20fps
    "stuck_frames_to_jump": 40,     # ~2s at 20fps

    # smoothing
    "smoothing_alpha": 0.5,

    # auto-resume after silence
    "auto_resume_seconds": 30.0,

    # map-aware (curiosity) wandering. enabled automatically when the
    # mapping service is running and the graph has enough cells.
    "map_mode_min_cells": 30,            # need at least this many reachable cells
    "map_mode_min_radius": 3.0,          # candidate cells must be at least this far
    "map_mode_max_radius": 20.0,         # ...and no further than this
    "map_mode_sample_count": 80,         # how many candidates to score per pick
    "map_mode_softmax_temp": 0.6,        # higher = more random pick
    "map_mode_dwell_min_s": 0.8,         # pause between picks (min)
    "map_mode_dwell_max_s": 2.5,         # pause between picks (max)
    "map_mode_visit_radius_m": 1.5,      # cells within this of pose marked visited
    "map_mode_recency_cap_s": 600.0,     # recency score saturates here
    "map_mode_w_recency": 1.0,           # weights for the score components
    "map_mode_w_frontier": 1.2,
    "map_mode_w_distance": 0.35,
    "map_mode_stale_timeout_s": 25.0,    # bail and re-pick if follow stalls
    "map_mode_waypoint_chance": 0.2,     # roll per-pick to visit a saved waypoint instead
    "map_mode_waypoint_min_dist": 2.5,   # skip waypoints closer than this (we're already there)
}


class Wanderer:
    """Raycast-driven VRChat wanderer."""

    def __init__(self, config, osc=None):
        self.config = config
        self.osc = osc
        self._active = False
        self._thread = None
        self._lock = threading.Lock()

        # external refs wired up by main.py
        self._face_tracker_ref = None
        self._emotion_system_ref = None
        self._mapping_service_ref = None

        # map-mode state
        self._map_state = "idle"        # idle | following | dwell
        self._map_visits: dict = {}     # serial -> last visit timestamp
        self._map_dwell_until: float = 0.0
        self._map_follow_start: float = 0.0
        self._map_target_cell = None
        self._map_last_pick_failed: float = 0.0

        # pause / resume state
        self._paused = False
        self._auto_paused = False
        self._resume_timer = None

        # navigation state
        self._smoothed_turn = 0.0
        self._smoothed_forward = 0.0
        self._last_straight_time = 0.0
        self._committed_turn_dir = 0.0
        self._committed_turn_until = 0.0
        self._stuck_frames = 0
        self._stuck_turn_dir = 1.0
        self._current_action = "idle"
        # DropFwd is only trusted as a ledge sensor after we have seen it
        # hit the ground at least once. otherwise a missing/misconfigured
        # DropFwd ray would make us think we are always on a cliff.
        self._dropfwd_ever_hit = False
        # rolling history of forward clearance for predictive steering
        self._clearance_history = []  # list of (timestamp, clearance)
        # recent wall hits, for escalation to u-turn when we keep bouncing
        self._recent_wall_hits = []  # list of timestamps

        # config dict, with optional yaml overrides under wanderer.*
        self._cfg = dict(DEFAULT_CFG)
        try:
            user_overrides = self.config.get("wanderer", default={}) or {}
            if isinstance(user_overrides, dict):
                for k, v in user_overrides.items():
                    if k in self._cfg and isinstance(v, (int, float, bool)):
                        self._cfg[k] = v
        except Exception:
            pass

        self._resume_delay = float(self._cfg["auto_resume_seconds"])

    @property
    def active(self):
        return self._active

    # ------------------------------------------------------------------
    # legacy compat: depth model preload no longer needed
    # ------------------------------------------------------------------
    def preload(self):
        # kept so main.py call site doesnt break, raycasts dont need preloading
        return

    # ------------------------------------------------------------------
    # raycast readers
    # ------------------------------------------------------------------
    def _ray(self, name):
        state = getattr(self.osc, "raycast_state", None) if self.osc else None
        if state is None:
            return None
        r = state.get(name)
        if r is None or not r.is_fresh():
            return None
        return r

    def _forward_clearance(self):
        """Effective forward clearance in meters. Prefers FwdNear (1.5m hip
        ray) for stopping, falls back to Fwd (5m head ray). Returns None if
        neither ray is reporting yet."""
        near = self._ray("FwdNear")
        head = self._ray("Fwd")
        if near is not None and near.hit:
            return near.distance
        if head is not None and head.hit:
            return head.distance
        if near is not None or head is not None:
            # no hit on either = wide open, return a sensible large value
            if head is not None:
                return max(head.distance, 5.0)
            return max(near.distance, 1.5)
        return None

    def _side_blocked(self, name, threshold):
        r = self._ray(name)
        if r is None:
            return False
        return r.hit and r.distance < threshold

    def _side_distance(self, name, default=None):
        r = self._ray(name)
        if r is None:
            return default
        # for steering we treat "no hit" as max reference distance
        if not r.hit:
            return max(r.distance, self._cfg["side_reference"])
        return r.distance

    def _drop_ahead(self):
        r = self._ray("DropFwd")
        if r is None:
            return False
        if r.hit:
            self._dropfwd_ever_hit = True
            return False
        # only trust "miss = ledge" once we have proof the ray works,
        # otherwise a bad ray config would lock us in reverse forever
        return self._dropfwd_ever_hit

    # ------------------------------------------------------------------
    # decision (continuous gradient style + smart escapes)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # OSC
    # ------------------------------------------------------------------
    def _send_osc(self, turn, forward):
        if not self.osc or self._paused:
            return
        client = self.osc.client
        client.send_message("/input/LookHorizontal", float(max(-1, min(1, turn))))
        client.send_message("/input/Vertical", float(max(-1, min(1, forward))))
        client.send_message("/input/Horizontal", 0.0)
        client.send_message("/input/Run", 0)

    def _zero_osc(self):
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/LookHorizontal", 0.0)
        client.send_message("/input/LookVertical", 0.0)
        client.send_message("/input/Vertical", 0.0)
        client.send_message("/input/Horizontal", 0.0)
        client.send_message("/input/Run", 0)

    def _do_jump(self):
        if not self.osc:
            return
        client = self.osc.client
        client.send_message("/input/Jump", 1)
        time.sleep(0.05)
        client.send_message("/input/Jump", 0)

    # ------------------------------------------------------------------
    # pause / resume hooks (called from gemini receive loop on speech)
    # ------------------------------------------------------------------
    def pause(self):
        with self._lock:
            self._paused = True
            self._zero_osc()

    def resume(self):
        with self._lock:
            self._paused = False
            self._auto_paused = False
            self._cancel_resume_timer()

    def on_speech_activity(self):
        if not self._active or self._paused:
            return
        with self._lock:
            self._paused = True
            self._auto_paused = True
            self._zero_osc()
        self._reset_resume_timer()

    def on_ai_speaking(self):
        if self._active and self._paused and self._auto_paused:
            self._reset_resume_timer()

    def _reset_resume_timer(self):
        self._cancel_resume_timer()
        t = threading.Timer(self._resume_delay, self._auto_resume)
        t.daemon = True
        t.start()
        self._resume_timer = t

    def _cancel_resume_timer(self):
        t = self._resume_timer
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass
            self._resume_timer = None

    def _auto_resume(self):
        if self._active and self._paused and self._auto_paused:
            self.resume()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self):
        if self._active and self._thread and self._thread.is_alive():
            return {"result": "ok", "message": "already wandering"}

        if self._face_tracker_ref and self._face_tracker_ref.active:
            self._face_tracker_ref.stop()
        if self._emotion_system_ref:
            self._emotion_system_ref.set_wandering(True)

        self._active = True
        self._smoothed_turn = 0.0
        self._smoothed_forward = 0.0
        self._last_straight_time = time.monotonic()
        self._committed_turn_dir = 0.0
        self._committed_turn_until = 0.0
        self._stuck_frames = 0
        self._current_action = "starting"
        self._dropfwd_ever_hit = False

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="wanderer")
        self._thread.start()
        return {"result": "ok", "message": "started wandering"}

    def stop(self):
        if not self._active:
            return {"result": "ok", "message": "not wandering"}

        self._active = False
        self._paused = False
        self._auto_paused = False
        self._cancel_resume_timer()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self._emotion_system_ref:
            self._emotion_system_ref.set_wandering(False)
        if self._face_tracker_ref and not self._face_tracker_ref.active:
            self._face_tracker_ref.start()

        return {"result": "ok", "message": "stopped wandering"}

    # ------------------------------------------------------------------
    # map-aware wandering (curiosity exploration)
    # ------------------------------------------------------------------
    def _map_available(self):
        """True when the mapping service is live and the graph is rich enough
        for us to actually plan paths through it."""
        ms = self._mapping_service_ref
        if ms is None or not getattr(ms, "_running", False):
            return False
        nav = getattr(ms, "_nav", None)
        if nav is None or nav.graph is None:
            return False
        # need a current pose lock so we have a starting cell
        if getattr(nav, "current", None) is None:
            return False
        try:
            n = len(nav.graph)
        except Exception:
            return False
        return n >= int(self._cfg["map_mode_min_cells"])

    def _explorer_busy(self):
        ms = self._mapping_service_ref
        ex = getattr(ms, "_explorer", None) if ms else None
        if ex is None:
            return False
        st = getattr(ex, "follow_status", None)
        if isinstance(st, dict):
            return bool(st.get("active"))
        return bool(getattr(st, "active", False))

    def _mark_visited(self, now):
        """Stamp the current and nearby reachable cells as recently visited."""
        from src.voxel_nav import serial_to_center, SCALE
        ms = self._mapping_service_ref
        nav = getattr(ms, "_nav", None) if ms else None
        if nav is None or nav.current is None:
            return
        cur_serial = nav.current.serial
        cx, cy, cz = cur_serial
        try:
            r_cells = max(1, int(self._cfg["map_mode_visit_radius_m"] * SCALE))
        except Exception:
            r_cells = 4
        with nav.graph._lock:  # noqa: SLF001
            for serial in nav.graph.nodes.keys():
                vx, vy, vz = serial
                if abs(vx - cx) + abs(vz - cz) <= r_cells and abs(vy - cy) <= 2:
                    self._map_visits[serial] = now

    def _score_candidate(self, serial, node, cur_xyz, now):
        """Higher score = better wander target. Combines recency-of-visit,
        frontier-ness (how surrounded by mapped neighbors), and distance."""
        from src.voxel_nav import serial_to_center
        cfg = self._cfg
        nav = self._mapping_service_ref._nav  # noqa: SLF001
        try:
            wx, wy, wz = serial_to_center(serial)
        except Exception:
            return None
        cx, cy, cz = cur_xyz
        dx, dz = wx - cx, wz - cz
        dist = (dx * dx + dz * dz) ** 0.5
        if dist < cfg["map_mode_min_radius"] or dist > cfg["map_mode_max_radius"]:
            return None
        # vertical sanity: don't pick cells more than ~3m above/below us
        if abs(wy - cy) > 3.0:
            return None
        last_visit = self._map_visits.get(serial, 0.0)
        recency = min(cfg["map_mode_recency_cap_s"],
                      now - last_visit if last_visit else cfg["map_mode_recency_cap_s"])
        recency_n = recency / cfg["map_mode_recency_cap_s"]
        # frontier: count how many of the 8 horizontal neighbors are missing
        vx, vy, vz = serial
        missing = 0
        nodes = nav.graph.nodes
        for ddx in (-1, 0, 1):
            for ddz in (-1, 0, 1):
                if ddx == 0 and ddz == 0:
                    continue
                if (vx + ddx, vy, vz + ddz) not in nodes:
                    missing += 1
        frontier_n = missing / 8.0
        # distance is normalized within band
        span = cfg["map_mode_max_radius"] - cfg["map_mode_min_radius"]
        dist_n = (dist - cfg["map_mode_min_radius"]) / max(span, 0.1)
        return (cfg["map_mode_w_recency"] * recency_n
                + cfg["map_mode_w_frontier"] * frontier_n
                + cfg["map_mode_w_distance"] * dist_n)

    def _pick_map_target(self):
        """Curiosity-driven target pick. Returns (score, serial, node) or None."""
        from src.voxel_nav import serial_to_center, NodeType
        ms = self._mapping_service_ref
        nav = ms._nav  # noqa: SLF001
        if nav.current is None:
            return None
        try:
            cur_world = serial_to_center(nav.current.serial)
        except Exception:
            return None
        now = time.monotonic()
        sample_n = int(self._cfg["map_mode_sample_count"])
        with nav.graph._lock:  # noqa: SLF001
            all_nodes = [(s, n) for s, n in nav.graph.nodes.items()
                         if n.node_type == NodeType.REACHABLE]
        if not all_nodes:
            return None
        if len(all_nodes) > sample_n:
            sample = random.sample(all_nodes, sample_n)
        else:
            sample = all_nodes
        scored = []
        for serial, node in sample:
            s = self._score_candidate(serial, node, cur_world, now)
            if s is not None:
                scored.append((s, serial, node))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: min(15, len(scored))]
        temp = max(0.05, float(self._cfg["map_mode_softmax_temp"]))
        import math
        max_s = top[0][0]
        weights = [math.exp((s - max_s) / temp) for s, _, _ in top]
        total = sum(weights)
        if total <= 0:
            return top[0]
        r = random.random() * total
        acc = 0.0
        for w, entry in zip(weights, top):
            acc += w
            if r <= acc:
                return entry
        return top[0]

    def _start_map_follow(self, node):
        """Kick off an explorer follow_path to the chosen node via goto_xyz."""
        from src.voxel_nav import serial_to_center
        ms = self._mapping_service_ref
        try:
            wx, wy, wz = serial_to_center(node.serial)
        except Exception:
            return False
        # set casual speed before kicking off so the explorer reads it
        try:
            if ms._explorer is not None:  # noqa: SLF001
                ms._explorer.speed_mode = "walk"  # noqa: SLF001
        except Exception:
            pass
        try:
            res = ms.goto_xyz(wx, wy, wz, label="wander")
        except Exception:
            logger.exception("wanderer: goto_xyz crashed")
            return False
        if not res or not res.get("found"):
            return False
        return True

    def _try_waypoint_visit(self, now):
        """Pick a random saved waypoint for the current world and head to it.
        Returns True if a follow was started."""
        from src.voxel_nav import serial_to_center
        cfg = self._cfg
        ms = self._mapping_service_ref
        wpstore = getattr(ms, "_waypoints", None) if ms else None
        if wpstore is None:
            return False
        try:
            waypoints = list(wpstore.list())
        except Exception:
            return False
        if not waypoints:
            return False
        # filter out ones we're already standing on / way too close
        nav = ms._nav  # noqa: SLF001
        cur_world = None
        if nav.current is not None:
            try:
                cur_world = serial_to_center(nav.current.serial)
            except Exception:
                cur_world = None
        if cur_world is not None:
            cx, _, cz = cur_world
            min_d = cfg["map_mode_waypoint_min_dist"]
            waypoints = [w for w in waypoints
                         if ((w.x - cx) ** 2 + (w.z - cz) ** 2) ** 0.5 >= min_d]
        if not waypoints:
            return False
        wp = random.choice(waypoints)
        try:
            if ms._explorer is not None:  # noqa: SLF001
                ms._explorer.speed_mode = "walk"  # noqa: SLF001
        except Exception:
            pass
        try:
            res = ms.goto_xyz(wp.x, wp.y, wp.z, label=f"wander:wp:{wp.name}",
                              final_yaw_deg=float(wp.yaw))
        except Exception:
            logger.exception("wanderer: goto_xyz waypoint crashed")
            return False
        if not res or not res.get("found"):
            return False
        self._map_target_cell = None
        self._map_follow_start = now
        self._map_state = "following"
        self._current_action = "map_wp"
        logger.info("wanderer: visiting waypoint '%s' at (%.1f, %.1f, %.1f)",
                    wp.name, wp.x, wp.y, wp.z)
        return True

    def _tick_map_mode(self, now):
        """One iteration of map-aware wandering. Returns True if it drove the
        avatar this tick (so reactive logic should be skipped)."""
        cfg = self._cfg
        state_name = self._map_state

        # while the explorer is driving we just monitor and update visit map
        if self._explorer_busy():
            self._map_state = "following"
            self._mark_visited(now)
            # stall guard: if a single follow takes too long, cancel and re-pick
            if self._map_follow_start and (now - self._map_follow_start) > cfg["map_mode_stale_timeout_s"]:
                logger.info("wanderer: follow stalled, cancelling and re-picking")
                try:
                    self._mapping_service_ref.cancel_goto()
                except Exception:
                    pass
                self._map_state = "dwell"
                self._map_dwell_until = now + 1.0
                self._map_follow_start = 0.0
            return True

        # not driving: dwell, then pick a new target
        if state_name == "following":
            # just finished a follow, briefly dwell
            self._map_state = "dwell"
            self._map_dwell_until = now + random.uniform(
                cfg["map_mode_dwell_min_s"], cfg["map_mode_dwell_max_s"]
            )
            self._zero_osc()
            return True

        if state_name == "dwell" and now < self._map_dwell_until:
            self._zero_osc()
            return True

        # time to pick
        # roll for a waypoint visit first if any are saved for this world
        if random.random() < cfg["map_mode_waypoint_chance"]:
            if self._try_waypoint_visit(now):
                return True

        pick = self._pick_map_target()
        if pick is None:
            # backoff: don't hammer pick every frame if there's nothing useful
            self._map_last_pick_failed = now
            return False
        score, serial, node = pick
        if not self._start_map_follow(node):
            self._map_last_pick_failed = now
            return False
        self._map_target_cell = serial
        self._map_follow_start = now
        self._map_state = "following"
        self._current_action = "map_goto"
        try:
            from src.voxel_nav import serial_to_center
            wx, wy, wz = serial_to_center(node.serial)
            logger.info("wanderer: heading to (%.1f, %.1f, %.1f) score=%.2f",
                        wx, wy, wz, score)
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def _run_loop(self):
        logger.info("Wanderer started (raycast mode, target %s fps)", TARGET_FPS)
        frame_interval = 1.0 / TARGET_FPS
        log_counter = 0
        warned_no_rays = False
        prev_mode = None

        try:
            while self._active:
                t0 = time.perf_counter()

                if self._paused:
                    self._zero_osc()
                    time.sleep(0.25)
                    continue

                # prefer map-aware curiosity wandering when we have a map
                if self._map_available():
                    if prev_mode != "map":
                        logger.info("wanderer: switching to map-aware mode")
                        prev_mode = "map"
                        self._map_state = "idle"
                    drove = self._tick_map_mode(time.monotonic())
                    if drove:
                        elapsed = time.perf_counter() - t0
                        sleep_time = frame_interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue
                    # if map-mode couldn't pick a target, fall through to reactive
                else:
                    if prev_mode != "reactive":
                        if prev_mode is not None:
                            logger.info("wanderer: switching to reactive mode")
                        prev_mode = "reactive"

                # sanity check that raycasts are actually streaming
                state = getattr(self.osc, "raycast_state", None) if self.osc else None
                if state is None or not state.get_all():
                    if not warned_no_rays:
                        logger.warning(
                            "Wanderer: no raycast params seen yet. is the sensor "
                            "rig on the avatar and VRChat OSC enabled?"
                        )
                        warned_no_rays = True
                    self._zero_osc()
                    time.sleep(0.5)
                    continue
                warned_no_rays = False

                turn, forward = self._decide()
                self._send_osc(turn, forward)

                log_counter += 1
                if log_counter <= 5 or log_counter % (TARGET_FPS * 3) == 0:
                    clearance = self._forward_clearance()
                    vel_z = self.osc.velocity_z if self.osc else 0.0
                    clr_s = ("%.2fm" % clearance) if clearance is not None else "n/a"
                    logger.info(
                        "Wanderer: %-9s clr=%s turn=%+.2f fwd=%+.2f velZ=%+.2f stuck=%d",
                        self._current_action, clr_s, turn, forward, vel_z, self._stuck_frames,
                    )

                elapsed = time.perf_counter() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception:
            logger.exception("Wanderer loop crashed")
        finally:
            self._zero_osc()
            self._active = False
