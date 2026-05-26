"""YOLO + boxmot DeepOCSORT model loading and lifecycle."""

import logging
import shutil
import threading
import time
from pathlib import Path

from ._constants import FRAME_H, FRAME_W, MODEL_DIR, MODEL_NAME

logger = logging.getLogger(__name__)


class ModelMixin:
    def preload(self):
        """Pre-load the YOLO model + warmup in a background thread.

        Non-blocking, call at startup so startFollow is instant later.
        Does NOT preload the boxmot ReID model -- that gets loaded when
        startfollow runs because it needs the same device as torch."""
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
            self._use_half = True
        else:
            device = "cpu"
            self._use_half = False
            logger.warning(
                "CUDA not available -- running on CPU (expect <10 FPS). "
                "Install PyTorch CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )
            self.model.to(device)

        self._torch_device = device
        logger.info(f"{MODEL_NAME} loaded on {device} (FP16={self._use_half})")

        # warmup inference (jit compile kernels, allocate buffers)
        logger.debug("Running warmup inference...")
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.predict(
                dummy, conf=0.5, classes=[0],
                max_det=5, verbose=False, half=self._use_half,
            )
        logger.debug("Warmup done")

    def _ensure_boxmot_tracker(self):
        """Initialise the boxmot DeepOCSORT tracker. Lazy because the ReID
        model needs to load AFTER torch/CUDA is already up."""
        if self._bot_tracker is not None:
            return

        try:
            from boxmot.trackers.tracker_zoo import create_tracker
        except ImportError:
            logger.error(
                "boxmot is not installed. Run: pip install boxmot. "
                "Falling back to no ReID -- ids will not persist across occlusion."
            )
            self._bot_tracker = False
            return

        reid_name = self._cfg.get("reid_model", "osnet_x0_25_msmt17.pt")
        reid_path = Path(MODEL_DIR) / reid_name
        reid_path.parent.mkdir(parents=True, exist_ok=True)

        device = getattr(self, "_torch_device", "cpu")
        # boxmot wants 'cuda:0' or 'cpu' style strings here
        bot_device = "cuda:0" if device == "cuda" else "cpu"
        use_half = bool(self._cfg.get("reid_half", True)) and device == "cuda"

        try:
            self._bot_tracker = create_tracker(
                tracker_type="deepocsort",
                reid_weights=reid_path,
                device=bot_device,
                half=use_half,
                per_class=False,
            )
            logger.info(
                f"boxmot DeepOCSORT ready (reid={reid_name}, device={bot_device}, half={use_half})"
            )
        except Exception as e:
            logger.error(f"Failed to init boxmot DeepOCSORT: {e}", exc_info=True)
            self._bot_tracker = False

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
            # dont nuke the lock here -- boxmot keeps the reid embedding
            # so the same person should keep their id even if we drop the
            # tracker object. recreate it next frame.
            self._bot_tracker = None
            self._next_tracker_reset = now + reset_interval
            logger.info("Tracker state refreshed to keep long sessions stable")
