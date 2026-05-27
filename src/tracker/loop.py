"""Main tracker loop, runs in a background thread at TARGET_FPS."""

import logging
import time

from .config import FRAME_H, FRAME_W, TARGET_FPS

logger = logging.getLogger("src.tracker")


class LoopMixin:
    def _run_loop(self):
        import cv2

        # init screen capture FIRST so the early-bettercam DXGI context
        # is consumed before torch/CUDA loading can mess with it
        capture_fn = self._init_screen_capture()
        if capture_fn is None:
            self._active = False
            return

        # now load model (this triggers torch/CUDA import). if preload() was
        # called at startup it returns instantly, otherwise blocks here.
        if not self._preload_ready.is_set():
            logger.info("Waiting for model preload...")
            self._preload_ready.wait(timeout=60)
        import torch
        self._ensure_model()

        logger.info(f"Player tracker started, target {TARGET_FPS} FPS")
        frame_interval = 1.0 / TARGET_FPS

        try:
            while self._active:
                t0 = time.perf_counter()

                frame = capture_fn()
                if frame is None:
                    time.sleep(0.001)
                    continue

                resized = cv2.resize(frame, (FRAME_W, FRAME_H))

                with torch.no_grad():
                    results = self.model.track(
                        resized,
                        persist=not self._first_frame,
                        tracker="bytetrack.yaml",
                        conf=self._cfg["confidence_threshold"],
                        iou=self._cfg["iou_threshold"],
                        classes=[0],
                        max_det=self._cfg["max_detections"],
                        verbose=False,
                        half=self._use_half,
                    )
                self._first_frame = False

                detections = self._parse_results(results)
                self._update_tracking(detections)
                self._send_osc()

                if self._vision_debug:
                    self._push_debug_frame(resized, results, detections)

                self._maybe_refresh_tracker_state(torch)

                self._frame_count += 1
                elapsed = time.perf_counter() - self._fps_timer
                if elapsed >= 2.0:
                    self._fps = self._frame_count / elapsed
                    logger.info(
                        f"Tracker: {self._fps:.1f} FPS | "
                        f"target_id={self._locked_id} | "
                        f"area={self._current_target_area:.4f}"
                    )
                    self._frame_count = 0
                    self._fps_timer = time.perf_counter()

                dt = time.perf_counter() - t0
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)

        except Exception as e:
            logger.error(f"Tracker loop error: {e}", exc_info=True)
        finally:
            self._zero_osc()
            self._cleanup_inference_cache(torch if "torch" in locals() else None)
            self._close_screen_capture()
            self._active = False
            logger.info(f"Player tracker stopped (last avg {self._fps:.1f} FPS)")
