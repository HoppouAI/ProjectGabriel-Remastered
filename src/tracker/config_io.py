"""Tracker config file I/O (config.json next to the yolo weights)."""

import json
import logging
from pathlib import Path

from .config import DEFAULT_CFG, MODEL_DIR

logger = logging.getLogger("src.tracker")


class ConfigIOMixin:
    def _load_config(self):
        config_path = Path(MODEL_DIR) / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    self._cfg.update(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load tracker config: {e}")

    def _save_config(self):
        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump(self._cfg, f, indent=2)

    def reload_config(self):
        # re-read config.json from disk and swap in the new dict. cheap, safe under GIL
        new_cfg = dict(DEFAULT_CFG)
        config_path = Path(MODEL_DIR) / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    new_cfg.update(json.load(f))
            except Exception as e:
                logger.warning(f"reload_config failed: {e}")
                return False
        self._cfg = new_cfg
        logger.info("Tracker config hot-reloaded")
        return True

    def get_config(self):
        return dict(self._cfg)
