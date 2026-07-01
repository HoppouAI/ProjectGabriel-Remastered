"""Settings tuning + saved-world list/delete."""

from __future__ import annotations

import logging

from ._base import normalize_speed

logger = logging.getLogger(__name__)


class WorldsMixin:
    """update_settings, list_worlds, delete_world."""

    def update_settings(self, *, tick_hz: float | None = None,
                        force_run: bool | None = None,
                        manual_wall_distance: float | None = None,
                        manual_wall_ratio: float | None = None,
                        speed_mode: str | None = None) -> dict:
        """Live-tune mapping speed knobs. Returns the new settings dict."""
        with self._lock:
            if tick_hz is not None:
                self._tick_hz = max(5.0, min(float(tick_hz), 120.0))
            if force_run is not None:
                self._force_run = bool(force_run)
                if self._explorer is not None:
                    try:
                        self._explorer.force_run = self._force_run
                    except Exception:
                        pass
            if speed_mode is not None:
                mode = normalize_speed(speed_mode)
                if mode is None:
                    raise ValueError(
                        f"speed_mode must be slow/normal/fast/sprint, got {speed_mode!r}")
                self._speed_mode = mode
                if self._explorer is not None:
                    try:
                        self._explorer.speed_mode = mode
                    except Exception:
                        pass
            if manual_wall_distance is not None:
                self.manual_wall_distance = max(
                    0.02, min(float(manual_wall_distance), 2.0))
            if manual_wall_ratio is not None:
                self.manual_wall_ratio = max(
                    0.0, min(float(manual_wall_ratio), 1.0))
            return {
                "tick_hz": self._tick_hz,
                "force_run": self._force_run,
                "speed_mode": self._speed_mode,
                "manual_wall_distance": self.manual_wall_distance,
                "manual_wall_ratio": self.manual_wall_ratio,
            }

    def list_worlds(self) -> list[dict]:
        """All saved world maps on disk. Useful for the UI's delete menu."""
        out: list[dict] = []
        try:
            for p in self._nav._data_dir.glob("*.json"):  # noqa: SLF001
                world_id = p.stem
                size_kb = p.stat().st_size / 1024.0
                out.append({
                    "world": world_id,
                    "size_kb": round(size_kb, 1),
                    "is_current": world_id == self._world_id,
                })
        except Exception:
            logger.exception("mapping: list_worlds failed")
        out.sort(key=lambda w: w["world"])
        return out

    def delete_world(self, world_id: str | None = None) -> dict:
        """Delete a saved world map. If world_id is None or matches the
        current world, also clears the in-memory graph and stops mapping."""
        with self._lock:
            target = world_id or self._world_id
            target = target.strip()
            if not target:
                raise ValueError("world id required")

            is_current = (target == self._world_id)
            if is_current and self._running:
                # stop tick loop so we dont re-save immediately
                self._stop_evt.set()
                try:
                    if self._explorer is not None:
                        self._explorer.stop()
                except Exception:
                    pass
                self._explorer = None
                self._explore_enabled = False
                try:
                    if self._reader is not None:
                        self._reader.stop()
                except Exception:
                    pass
                self._reader = None
                self._running = False

            removed = False
            try:
                path = self._nav._data_dir / f"{target}.json"  # noqa: SLF001
                if path.exists():
                    path.unlink()
                    removed = True
            except Exception as exc:
                logger.exception("mapping: delete world file failed")
                raise RuntimeError(f"delete failed: {exc}") from exc

            if is_current:
                # wipe in-memory graph too so the viewer empties out
                try:
                    self._nav.graph.clear()
                    self._nav._current = None  # noqa: SLF001
                    self._nav._previous = None  # noqa: SLF001
                    self._nav._dirty = False  # noqa: SLF001
                except Exception:
                    logger.exception("mapping: wipe graph failed")

            logger.info("mapping: deleted world '%s' (file_removed=%s)",
                        target, removed)
            return {"world": target, "removed": removed,
                    "was_current": is_current}
