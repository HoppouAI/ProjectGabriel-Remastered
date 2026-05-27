"""Lifecycle (start/stop) and the main wander loop."""

from __future__ import annotations

import logging
import threading
import time

from .config import TARGET_FPS

logger = logging.getLogger("src.wanderer")


class LoopMixin:
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
