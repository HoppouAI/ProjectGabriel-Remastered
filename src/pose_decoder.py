"""Pose exfiltration via shader-rendered pixel grid.

VRChat doesn't expose the player's world position/rotation over OSC, but
shaders running on the avatar DO have it via `unity_ObjectToWorld`. We use
a tiny quad parented to the avatar that renders a fixed grid of pixels at
a known on-screen location. The pixel values encode the avatar's world
pose. Python screen-captures that grid with mss and decodes it back to
floats.

This adopts a fullscreen quad / Overlay-queue shader trick to dodge sRGB
gamma: each data pixel writes pure 0.0 or pure 1.0 per channel (one bit
per channel per pixel), which round-trips cleanly through any tonemapping.

Encoding (matches `unity_assets/shaders/PoseExfilScreen.shader`):

    Grid is 34 cells wide x 2 cells tall.

    Row 0 (top, "position" row):
      cells 0..31 = bits 0..31 of (position.x, position.y, position.z)
                    encoded in (R, G, B) channels respectively
      cell 32     = RED marker   (255, 0, 0)
      cell 33     = GREEN marker (0, 255, 0)

    Row 1 (bottom, "forward" row, used to derive yaw):
      cells 0..31 = bits 0..31 of (forward.x, forward.y, forward.z)
      cell 32     = GREEN marker (0, 255, 0)
      cell 33     = RED marker   (255, 0, 0)

Each float is packed as: uint32 = (float + 5000) * 100
  -> 1cm precision, +-5000m range, plenty for any VRChat world.

Yaw is recovered as atan2(forward.x, forward.z) in degrees.

If screen capture grabs the grid mid-write or the shader isn't on screen,
the marker pixels won't match and `decode_strip` returns None.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# grid layout. must match the shader exactly.
GRID_W = 34         # cells across (32 data + 2 markers)
GRID_H = 2          # rows (position + forward)
DATA_CELLS = 32     # data cells per row (32 bits per channel = uint32)

# each logical cell is CELL_SIZE x CELL_SIZE physical screen pixels so the
# grid is big enough to see and survives any resolution scaling. must match
# _CellSize in PoseExfilScreen.shader.
DEFAULT_CELL_SIZE = 8

# float packing (matches packFloat in the shader):
#   uint = (float + 5000) * 100      -> 1cm precision, +-5000m range
_PACK_OFFSET = 5000.0
_PACK_SCALE = 100.0

# threshold for deciding bit=1 vs bit=0 on a sampled channel value (0..255).
# anything brighter than 127 counts as on. with pure 0/1 output we expect
# either ~0 or ~255 so 127 is comfortably in the middle.
_BIT_THRESHOLD = 127


@dataclass
class WorldPose:
    """Player world pose decoded from the shader grid."""
    x: float        # meters
    y: float        # meters
    z: float        # meters
    yaw: float      # degrees, 0..360 (Unity Y-axis rotation)
    timestamp: float  # monotonic seconds when decoded


# --- packing helpers ---------------------------------------------------

def _pack_float(f: float) -> int:
    """Pack a float (meters) to uint32 the same way the shader does."""
    return int((f + _PACK_OFFSET) * _PACK_SCALE) & 0xFFFFFFFF


def _unpack_float(u: int) -> float:
    return (u / _PACK_SCALE) - _PACK_OFFSET


# --- encoding (used by tests and offline tooling) ----------------------

def encode_pose(pose: WorldPose) -> bytes:
    """Pack a pose into the logical cell stream the shader would emit.

    Output is `GRID_W * GRID_H * 3` bytes, RGB per cell, row-major. Data
    cells contain 0 or 255 per channel; marker cells contain pure red or
    green. Used by the test suite for round-trip checks.

    Note: forward vector is reconstructed from yaw (forward.y = 0 assumed,
    since we only encode a yaw angle in the high-level WorldPose).
    """
    import math

    out = bytearray(GRID_W * GRID_H * 3)

    px = _pack_float(pose.x)
    py = _pack_float(pose.y)
    pz = _pack_float(pose.z)

    yaw_rad = math.radians(pose.yaw)
    fx = math.sin(yaw_rad)
    fy = 0.0
    fz = math.cos(yaw_rad)
    fxu = _pack_float(fx)
    fyu = _pack_float(fy)
    fzu = _pack_float(fz)

    def write_cell(row: int, col: int, r: int, g: int, b: int) -> None:
        idx = (row * GRID_W + col) * 3
        out[idx]     = r
        out[idx + 1] = g
        out[idx + 2] = b

    # row 0: position bits
    for col in range(DATA_CELLS):
        r = 255 if (px >> col) & 1 else 0
        g = 255 if (py >> col) & 1 else 0
        b = 255 if (pz >> col) & 1 else 0
        write_cell(0, col, r, g, b)
    write_cell(0, 32, 255, 0, 0)   # RED marker
    write_cell(0, 33, 0, 255, 0)   # GREEN marker

    # row 1: forward bits
    for col in range(DATA_CELLS):
        r = 255 if (fxu >> col) & 1 else 0
        g = 255 if (fyu >> col) & 1 else 0
        b = 255 if (fzu >> col) & 1 else 0
        write_cell(1, col, r, g, b)
    write_cell(1, 32, 0, 255, 0)   # GREEN marker
    write_cell(1, 33, 255, 0, 0)   # RED marker

    return bytes(out)


def decode_strip(pixels: bytes, *, timestamp: float | None = None) -> WorldPose | None:
    """Decode the cell grid back to a WorldPose. Returns None if the marker
    pixels don't match (e.g. grid off screen, torn capture, wrong region).

    `pixels` should be `GRID_W * GRID_H * 3` bytes in RGB order, one byte
    triple per logical cell.
    """
    import math

    expected = GRID_W * GRID_H * 3
    if len(pixels) < expected:
        return None

    def cell(row: int, col: int) -> tuple[int, int, int]:
        idx = (row * GRID_W + col) * 3
        return pixels[idx], pixels[idx + 1], pixels[idx + 2]

    # validate markers. these are the only "magic" we have; if they're wrong
    # we either grabbed the wrong place on screen or the strip isn't visible.
    r0a, g0a, b0a = cell(0, 32)
    r0b, g0b, b0b = cell(0, 33)
    r1a, g1a, b1a = cell(1, 32)
    r1b, g1b, b1b = cell(1, 33)

    # marker pixels: dominant channel must be clearly above the others.
    # we don't require pure 0 on the other channels because sRGB roundtrip
    # and tonemapping can lift them a bit.
    def is_red(r: int, g: int, b: int)   -> bool: return r > 150 and r - g > 80 and r - b > 80
    def is_green(r: int, g: int, b: int) -> bool: return g > 150 and g - r > 80 and g - b > 80

    if not (is_red(r0a, g0a, b0a) and is_green(r0b, g0b, b0b)
            and is_green(r1a, g1a, b1a) and is_red(r1b, g1b, b1b)):
        return None

    def read_row(row: int) -> tuple[int, int, int]:
        x = y = z = 0
        for col in range(DATA_CELLS):
            r, g, b = cell(row, col)
            if r > _BIT_THRESHOLD: x |= (1 << col)
            if g > _BIT_THRESHOLD: y |= (1 << col)
            if b > _BIT_THRESHOLD: z |= (1 << col)
        return x, y, z

    pxu, pyu, pzu = read_row(0)
    fxu, _fyu, fzu = read_row(1)

    pos_x = _unpack_float(pxu)
    pos_y = _unpack_float(pyu)
    pos_z = _unpack_float(pzu)
    fwd_x = _unpack_float(fxu)
    fwd_z = _unpack_float(fzu)

    yaw = math.degrees(math.atan2(fwd_x, fwd_z))
    if yaw < 0:
        yaw += 360.0

    return WorldPose(
        x=pos_x,
        y=pos_y,
        z=pos_z,
        yaw=yaw,
        timestamp=timestamp if timestamp is not None else time.monotonic(),
    )


# --- live screen capture loop -------------------------------------------

class PoseExfilReader:
    """Background thread that screen-captures the pose grid and exposes the
    latest decoded pose. Uses mss so it doesn't fight bettercam (which is
    already owned by `src.tracker`).

    Region defaults to the top-left GRID_W*cell x GRID_H*cell rectangle.
    Configure to match wherever the avatar shader actually renders the
    grid on screen.
    """

    def __init__(
        self,
        *,
        region: dict | None = None,
        poll_hz: float = 20.0,
        monitor_index: int = 1,
        cell_size: int = DEFAULT_CELL_SIZE,
    ):
        # mss region dict: {"left": x, "top": y, "width": w, "height": h}
        # default region covers the full GRID_W x GRID_H cell grid in the
        # top-left corner.
        self._cell_size = max(1, int(cell_size))
        self._region = region or {
            "left": 0, "top": 0,
            "width":  GRID_W * self._cell_size,
            "height": GRID_H * self._cell_size,
        }
        self._poll_interval = 1.0 / max(1.0, poll_hz)
        self._monitor_index = monitor_index

        self._lock = threading.RLock()
        self._latest: WorldPose | None = None
        self._consecutive_failures = 0
        self._total_decodes = 0
        self._total_attempts = 0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="pose-exfil"
        )
        self._thread.start()
        logger.info(
            "pose_decoder: started, region=%s @ %.1f Hz",
            self._region, 1.0 / self._poll_interval,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def configure_region(self, region: dict) -> None:
        """Update the screen region on the fly. Useful when the user moves
        the strip to a different corner or changes resolution."""
        with self._lock:
            self._region = dict(region)

    def get(self) -> WorldPose | None:
        with self._lock:
            return self._latest

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_attempts": self._total_attempts,
                "total_decodes": self._total_decodes,
                "consecutive_failures": self._consecutive_failures,
                "decode_rate": (
                    self._total_decodes / self._total_attempts
                    if self._total_attempts else 0.0
                ),
            }

    def _run(self) -> None:
        # import mss lazily so this module stays importable without it
        try:
            import mss  # type: ignore
        except ImportError:
            logger.error("pose_decoder: mss not installed, exfil disabled")
            return

        with mss.mss() as sct:
            while not self._stop.is_set():
                tick_started = time.monotonic()
                try:
                    with self._lock:
                        region = dict(self._region)
                        cell = self._cell_size
                    raw = sct.grab(region)
                    # sample the center pixel of each logical cell so we
                    # ignore bilinear edge bleed between cells.
                    pixels = bytearray()
                    for row in range(GRID_H):
                        cy = row * cell + cell // 2
                        if cy >= raw.height:
                            break
                        for col in range(GRID_W):
                            cx = col * cell + cell // 2
                            if cx >= raw.width:
                                break
                            r, g, b = raw.pixel(cx, cy)
                            pixels.extend([r, g, b])
                    pose = decode_strip(bytes(pixels), timestamp=tick_started)
                except Exception as e:
                    logger.debug("pose_decoder: capture/decode error: %s", e)
                    pose = None

                with self._lock:
                    self._total_attempts += 1
                    if pose is not None:
                        self._latest = pose
                        self._total_decodes += 1
                        self._consecutive_failures = 0
                    else:
                        self._consecutive_failures += 1

                # warn occasionally if we've been failing a long time
                if self._consecutive_failures > 0 and self._consecutive_failures % 200 == 0:
                    logger.warning(
                        "pose_decoder: %d consecutive failures, is the shader on screen?",
                        self._consecutive_failures,
                    )

                # pace ourselves
                elapsed = time.monotonic() - tick_started
                sleep_for = self._poll_interval - elapsed
                if sleep_for > 0:
                    self._stop.wait(timeout=sleep_for)
