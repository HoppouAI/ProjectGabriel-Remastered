"""YOLO model loading, warmup, preload, and periodic cache cleanup."""

import logging
import shutil
import threading
import time
from pathlib import Path

from .config import FRAME_H, FRAME_W, MODEL_DIR, MODEL_NAME

logger = logging.getLogger("src.tracker")


class ModelMixin:
    def preload(self):
        """Pre-load the YOLO model + warmup in a background thread.
        Non-blocking, call at startup so startfollow is instant later."""
        def _do_preload():
            try:
                logger.debug("Background preload: loading YOLO model...")
                self._ensure_model()
                logger.debug("Background preload: model ready")
            except Exception as e:
                logger.error(f"Background preload failed: {e}")
            finally:
                self._preload_ready.set()

        t = threading.Thread(target=_do_preload, daemon=True, name="yolo-preload")
        t.start()

    def _ensure_model(self):
        if self.model is not None:
            return

        import numpy as np
        import torch
        from ultralytics import YOLO

        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / MODEL_NAME

        if not model_path.exists():
            logger.info(f"Downloading {MODEL_NAME} to {model_dir}...")
            temp_model = YOLO(MODEL_NAME)
            default_path = Path(MODEL_NAME)
            if default_path.exists():
                shutil.move(str(default_path), str(model_path))
            self.model = temp_model
        else:
            self.model = YOLO(str(model_path))

        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA available: {gpu_name} ({vram:.1f} GB)")
            self.model.to(device)
            self._use_half = True  # FP16 handled via half= in track()
        else:
            device = "cpu"
            self._use_half = False
            logger.warning(
                "CUDA not available, running on CPU (expect <10 FPS). "
                "Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )
            self.model.to(device)

        logger.info(f"{MODEL_NAME} loaded on {device} (FP16={self._use_half})")

        # warmup inference, JIT compiles kernels and allocates buffers
        logger.debug("Running warmup inference...")
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.track(
                dummy, persist=False, conf=0.5, classes=[0],
                max_det=5, verbose=False, half=self._use_half,
            )
        logger.debug("Warmup done")

    def _cleanup_inference_cache(self, torch_module=None):
        try:
            predictor = getattr(self.model, "predictor", None) if self.model else None
            if predictor is not None and hasattr(predictor, "results"):
                predictor.results = None
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Tracker cache cleanup skipped: {e}")
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def _maybe_refresh_tracker_state(self, torch_module):
        now = time.perf_counter()
        cleanup_interval = float(self._cfg.get("cache_cleanup_interval", 300.0))
        if cleanup_interval > 0 and now >= self._next_cache_cleanup:
            self._cleanup_inference_cache(torch_module)
            self._next_cache_cleanup = now + cleanup_interval

        reset_interval = float(self._cfg.get("tracker_reset_interval", 1800.0))
        if reset_interval > 0 and now >= self._next_tracker_reset:
            self._first_frame = True
            self._locked_id = None
            self._lock_lost_time = None
            self._next_tracker_reset = now + reset_interval
            logger.info("Tracker state refreshed to keep long sessions stable")
