"""Map-aware curiosity wandering. Used when the mapping service is live."""

from __future__ import annotations

import logging
import random
import time

logger = logging.getLogger("src.wanderer")


class MapModeMixin:
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
        from src.voxel_nav import SCALE
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
        # hard frontier guard: dont pick cells right on the edge of the
        # mapped area, otherwise we look like we're trying to expand the map.
        if missing > int(cfg["map_mode_max_missing_neighbors"]):
            return None
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
