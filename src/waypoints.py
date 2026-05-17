"""Named waypoints per VRChat world.

Stores user-named points like "spawn", "couch", "bar", per world ID, so the
AI can navigate back to them using the A* planner. Backed by a single JSON
file per world under data/waypoints/<world_id>.json.

World ID is whatever string the caller provides -- usually the VRChat
instance/world id from the instance monitor, or a manual label if the
caller doesnt know it yet.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data/waypoints")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    yaw: float = 0.0          # optional facing direction in degrees
    note: str = ""            # freeform AI/user note
    created_at: float = 0.0   # unix seconds
    updated_at: float = 0.0   # unix seconds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Waypoint":
        return cls(
            name=str(d["name"]),
            x=float(d["x"]),
            y=float(d.get("y", 0.0)),
            z=float(d["z"]),
            yaw=float(d.get("yaw", 0.0)),
            note=str(d.get("note", "")),
            created_at=float(d.get("created_at", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
        )


def _safe_world_id(world_id: str) -> str:
    cleaned = _SAFE_NAME.sub("_", world_id.strip())
    return cleaned[:120] or "unknown_world"


class WaypointStore:
    """Loads/saves a single world's waypoints. Thread safe."""

    def __init__(self, world_id: str, root: Path | None = None):
        self.world_id = _safe_world_id(world_id)
        self.root = root or DEFAULT_ROOT
        self._lock = threading.RLock()
        self._waypoints: dict[str, Waypoint] = {}
        self._path = self.root / f"{self.world_id}.json"
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("waypoints: failed to load %s: %s", self._path, e)
            return
        for entry in data.get("waypoints", []):
            try:
                w = Waypoint.from_dict(entry)
                self._waypoints[w.name.lower()] = w
            except Exception as e:
                logger.warning("waypoints: skipping bad entry: %s", e)

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "world_id": self.world_id,
            "waypoints": [w.to_dict() for w in self._waypoints.values()],
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------
    def add(
        self,
        name: str,
        x: float,
        z: float,
        *,
        y: float = 0.0,
        yaw: float = 0.0,
        note: str = "",
    ) -> Waypoint:
        key = name.strip().lower()
        if not key:
            raise ValueError("waypoint name cannot be empty")
        now = time.time()
        with self._lock:
            existing = self._waypoints.get(key)
            wp = Waypoint(
                name=name.strip(),
                x=float(x), y=float(y), z=float(z),
                yaw=float(yaw),
                note=note,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._waypoints[key] = wp
            self._save()
        return wp

    def remove(self, name: str) -> bool:
        key = name.strip().lower()
        with self._lock:
            if key not in self._waypoints:
                return False
            del self._waypoints[key]
            self._save()
        return True

    def get(self, name: str) -> Waypoint | None:
        with self._lock:
            return self._waypoints.get(name.strip().lower())

    def list(self) -> list[Waypoint]:
        with self._lock:
            return list(self._waypoints.values())

    def nearest(self, x: float, z: float) -> Waypoint | None:
        best: tuple[float, Waypoint] | None = None
        with self._lock:
            for w in self._waypoints.values():
                d = (w.x - x) ** 2 + (w.z - z) ** 2
                if best is None or d < best[0]:
                    best = (d, w)
        return best[1] if best else None
