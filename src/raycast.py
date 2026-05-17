"""Raycast state tracking.

Listens for VRChat avatar OSC parameters produced by `VRCRaycast` components
and keeps the latest reading per named ray. The avatar exports three params
per raycast:

    /avatar/parameters/<name>_Hit       bool
    /avatar/parameters/<name>_Distance  float (meters)
    /avatar/parameters/<name>_Ratio     float 0..1

The python side just reads them as they stream in over the existing OSC
listener socket. No screen capture, no math, raycast does the work in
engine.

This module is intentionally side-effect free, register the handlers from
wherever you already own the python-osc dispatcher (see `VRChatClient`).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Pattern

logger = logging.getLogger(__name__)

# the three suffixes VRChat appends to each VRCRaycast Parameter prefix
_FIELD_HIT = "Hit"
_FIELD_DISTANCE = "Distance"
_FIELD_RATIO = "Ratio"
_VALID_FIELDS = (_FIELD_HIT, _FIELD_DISTANCE, _FIELD_RATIO)

# VRCFury Full Controller prefixes merged params with VF<id>_ to namespace
# them. We strip this by default so consumers see clean ray names.
DEFAULT_STRIP_PREFIX = re.compile(r"^VF\d+_")


@dataclass
class RayReading:
    """Latest snapshot for a single named raycast."""
    name: str
    hit: bool = False
    distance: float = 0.0
    ratio: float = 0.0
    last_updated: float = 0.0  # monotonic seconds

    def is_fresh(self, max_age_seconds: float = 1.0) -> bool:
        """True if updated within the given window. Stale readings should be
        treated as 'no signal' rather than 'no hit'."""
        if self.last_updated <= 0.0:
            return False
        return (time.monotonic() - self.last_updated) <= max_age_seconds


class RaycastState:
    """Thread-safe registry of named raycast readings.

    The dispatcher feeds OSC values in from a background thread, consumers
    (wanderer, tracker, AI tools) read from the foreground. All access is
    locked.
    """

    def __init__(self, strip_prefix: str | Pattern | None = DEFAULT_STRIP_PREFIX):
        self._lock = threading.RLock()
        self._rays: dict[str, RayReading] = {}
        # consumers can pre-declare expected ray names so `get` always returns
        # something even before the first OSC packet lands. Otherwise we just
        # learn names as they show up.
        self._declared: set[str] = set()
        # optional prefix scrubber, used to drop VRCFury's VF<id>_ namespace
        if strip_prefix is None:
            self._strip_prefix: Pattern | None = None
        elif isinstance(strip_prefix, str):
            self._strip_prefix = re.compile(strip_prefix)
        else:
            self._strip_prefix = strip_prefix

    # --- declaration -----------------------------------------------------

    def declare(self, name: str) -> None:
        """Pre-register a ray name. Optional, but makes `get` always return a
        RayReading instead of None and helps the wanderer treat 'no data yet'
        as 'no obstacle' on startup."""
        with self._lock:
            self._declared.add(name)
            if name not in self._rays:
                self._rays[name] = RayReading(name=name)

    def declare_many(self, names: Iterable[str]) -> None:
        for name in names:
            self.declare(name)

    # --- updates from OSC ------------------------------------------------

    def update(self, parameter_name: str, value) -> bool:
        """Apply an incoming `/avatar/parameters/<x>` value. Returns True if
        the param matched the `<RayName>_<Field>` pattern and was consumed,
        False otherwise so the caller can route unknown params elsewhere."""
        if not parameter_name:
            return False
        # drop the VRCFury (or user-configured) namespace prefix if present
        if self._strip_prefix is not None:
            parameter_name = self._strip_prefix.sub("", parameter_name, count=1)
        # split on the LAST underscore so ray names with underscores work too
        # (e.g. "Drop_Left_Hit" -> name="Drop_Left", field="Hit")
        idx = parameter_name.rfind("_")
        if idx <= 0 or idx == len(parameter_name) - 1:
            return False
        name = parameter_name[:idx]
        field_name = parameter_name[idx + 1 :]
        if field_name not in _VALID_FIELDS:
            return False

        with self._lock:
            reading = self._rays.get(name)
            if reading is None:
                reading = RayReading(name=name)
                self._rays[name] = reading
            try:
                if field_name == _FIELD_HIT:
                    reading.hit = bool(value)
                elif field_name == _FIELD_DISTANCE:
                    reading.distance = max(0.0, float(value))
                elif field_name == _FIELD_RATIO:
                    reading.ratio = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                logger.debug(
                    "raycast: bad value for %s = %r", parameter_name, value
                )
                return False
            reading.last_updated = time.monotonic()
        return True

    # --- reads -----------------------------------------------------------

    def get(self, name: str) -> RayReading | None:
        with self._lock:
            reading = self._rays.get(name)
            if reading is None:
                return None
            # return a copy so callers can't mutate our state
            return RayReading(
                name=reading.name,
                hit=reading.hit,
                distance=reading.distance,
                ratio=reading.ratio,
                last_updated=reading.last_updated,
            )

    def get_all(self) -> dict[str, RayReading]:
        with self._lock:
            return {
                name: RayReading(
                    name=r.name,
                    hit=r.hit,
                    distance=r.distance,
                    ratio=r.ratio,
                    last_updated=r.last_updated,
                )
                for name, r in self._rays.items()
            }

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._rays.keys())

    # --- python-osc integration -----------------------------------------

    def register_handlers(
        self,
        dispatcher,
        prefix: str = "/avatar/parameters/",
        ray_names: Iterable[str] | None = None,
    ) -> None:
        """Wire this state object into a python-osc Dispatcher.

        If `ray_names` is provided, registers explicit handlers per
        `<prefix><name>_<field>` address (fastest path, recommended for
        production). Otherwise installs a default handler that inspects every
        avatar parameter and consumes the ones matching the raycast suffix
        pattern, leaving the rest for other handlers.
        """
        if ray_names:
            seen = set()
            for name in ray_names:
                if name in seen:
                    continue
                seen.add(name)
                self.declare(name)
                for field_name in _VALID_FIELDS:
                    address = f"{prefix}{name}_{field_name}"
                    dispatcher.map(address, self._osc_callback)
            logger.info(
                "raycast: registered explicit handlers for %d rays", len(seen)
            )
            return

        # fallback: catch-all on avatar parameters
        dispatcher.set_default_handler(self._osc_default_handler, needs_reply_address=False)
        logger.info("raycast: registered default handler (sniff mode)")

    def _osc_callback(self, address: str, *args) -> None:
        if not args:
            return
        # strip "/avatar/parameters/" prefix
        idx = address.rfind("/")
        param = address[idx + 1 :] if idx >= 0 else address
        self.update(param, args[0])

    def _osc_default_handler(self, address: str, *args) -> None:
        if not address.startswith("/avatar/parameters/"):
            return
        if not args:
            return
        param = address[len("/avatar/parameters/") :]
        self.update(param, args[0])


# --- helper queries for navigation ---------------------------------------

def forward_blocked(
    state: RaycastState,
    ray_name: str = "Fwd",
    threshold_meters: float = 1.0,
    max_age_seconds: float = 1.0,
) -> bool:
    """True if the named forward ray is hit and the distance is under threshold.
    Stale or missing readings are treated as 'unknown', NOT 'blocked', so the
    wanderer doesn't lock up if the avatar isn't reporting yet."""
    r = state.get(ray_name)
    if r is None or not r.is_fresh(max_age_seconds):
        return False
    return r.hit and r.distance > 0.0 and r.distance <= threshold_meters


def drop_ahead(
    state: RaycastState,
    ray_name: str = "DropFwd",
    safe_distance_meters: float = 1.5,
    max_age_seconds: float = 1.0,
) -> bool:
    """True if a forward-downward ray reports either no hit or a hit beyond
    the safe distance (meaning the floor is missing or too far below). Used to
    keep the wanderer from yeeting itself off ledges."""
    r = state.get(ray_name)
    if r is None or not r.is_fresh(max_age_seconds):
        return False
    # no hit at all means open air below -> definitely a drop
    if not r.hit:
        return True
    return r.distance > safe_distance_meters


def pick_clear_direction(
    state: RaycastState,
    candidates: Iterable[tuple[str, float]],
    min_clearance_meters: float = 2.0,
) -> str | None:
    """Given an iterable of `(ray_name, heading_degrees)` candidates, return
    the heading whose ray reports the most clearance, or None if nothing is
    clear enough.

    Returns the ray NAME (caller maps name back to heading). This intentionally
    doesn't return the heading directly so callers can attach extra metadata
    to each ray name (animation, priority, etc) without changing this API.
    """
    best_name = None
    best_distance = min_clearance_meters
    for name, _heading in candidates:
        r = state.get(name)
        if r is None or not r.is_fresh():
            continue
        # no hit = wide open, treat as max
        effective = r.distance if r.hit else 1000.0
        if effective >= best_distance:
            best_distance = effective
            best_name = name
    return best_name
