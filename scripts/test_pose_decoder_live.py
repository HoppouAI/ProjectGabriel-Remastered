"""Live pose grid decoder smoke test.

Scans every monitor for the distinctive marker pattern (RED-over-GREEN
beside GREEN-over-RED at the right edge of the grid), then samples the
full GRID_W x GRID_H cell grid and decodes it.

Usage:
    .venv\\Scripts\\python.exe scripts\\test_pose_decoder_live.py
    .venv\\Scripts\\python.exe scripts\\test_pose_decoder_live.py --watch
    .venv\\Scripts\\python.exe scripts\\test_pose_decoder_live.py --cell 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pose_decoder import (  # noqa: E402
    DEFAULT_CELL_SIZE,
    GRID_W,
    GRID_H,
    decode_strip,
)


def sample_grid(arr, left, top, cell):
    """Sample the GRID_W x GRID_H cell centers starting at (left, top).
    Returns bytes ready for decode_strip, or None if out of bounds.
    arr is HxWx4 BGRA from mss."""
    h, w = arr.shape[:2]
    if left + GRID_W * cell > w or top + GRID_H * cell > h:
        return None
    pixels = bytearray()
    for row in range(GRID_H):
        cy = top + row * cell + cell // 2
        for col in range(GRID_W):
            cx = left + col * cell + cell // 2
            px = arr[cy, cx]
            pixels.extend([int(px[2]), int(px[1]), int(px[0])])
    return bytes(pixels)


def find_grid_origin(arr, cell, max_hits=5):
    """Find the grid by locating the marker rectangles at the right edge.

    Markers (each marker is one cell wide / tall, `cell` px square):
      row 0: RED   then  GREEN
      row 1: GREEN then  RED

    Returns list of (left, top) top-left grid origins.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    r = arr[..., 2].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 0].astype(int)

    red_mask   = (r > 150) & (g < 100) & (b < 100) & (r - g > 80) & (r - b > 80)
    green_mask = (g > 150) & (r < 100) & (b < 100) & (g - r > 80) & (g - b > 80)

    h, w = r.shape
    candidates = []

    # find each red rectangle. iterate every red pixel that has no red pixel
    # immediately to its left or above it (top-left corner of a cluster).
    red_left_edge = red_mask & ~np.roll(red_mask, 1, axis=1)
    red_top_edge  = red_mask & ~np.roll(red_mask, 1, axis=0)
    red_corner    = red_left_edge & red_top_edge

    ys, xs = np.where(red_corner)
    seen = set()
    for y, x in zip(ys.tolist(), xs.tolist()):
        # measure this red cluster's width by scanning right
        rw = 0
        while x + rw < w and red_mask[y, x + rw]:
            rw += 1
        rh = 0
        while y + rh < h and red_mask[y + rh, x]:
            rh += 1
        if rw < 2 or rh < 2:
            continue  # too small, probably random red pixel

        # cell size estimate from the marker
        est_cell = max(rw, rh)

        # this red cluster might be the row 0 cell 32 marker. check the
        # neighbors: GREEN one cell right, GREEN one cell below, RED diag.
        x_right = x + est_cell
        y_down  = y + est_cell
        if x_right + est_cell > w or y_down + est_cell > h:
            continue
        # sample a pixel a few px into each expected neighbor cell
        probe = est_cell // 2
        if not green_mask[y + probe,         x_right + probe]: continue
        if not green_mask[y_down + probe,    x         + probe]: continue
        if not red_mask  [y_down + probe,    x_right   + probe]: continue

        # red cluster starts at column 32*cell from grid left
        left = x - 32 * est_cell
        top  = y
        if left < 0 or top < 0:
            continue
        key = (left, top, est_cell)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((left, top, est_cell))
        if len(candidates) >= max_hits:
            break
    return candidates


def scan_and_decode(cell):
    try:
        import mss
        import numpy as np  # noqa: F401
    except ImportError as e:
        print(f"ERROR: missing dep {e}. need mss + numpy.")
        return 1

    import numpy as np

    with mss.mss() as sct:
        for mi, mon in enumerate(sct.monitors):
            if mi == 0:
                continue  # virtual all-monitors entry
            print(f"--- monitor {mi}: {mon} ---")
            shot = sct.grab(mon)
            arr = np.array(shot, dtype=np.uint8)  # BGRA HxWx4
            candidates = find_grid_origin(arr, cell)
            if not candidates:
                print("  no marker pattern found")
                continue
            print(f"  found {len(candidates)} candidate grid origin(s)")
            for left, top, est_cell in candidates:
                pixels = sample_grid(arr, left, top, est_cell)
                if pixels is None:
                    continue
                # always print raw cells for debugging
                print(f"  trying origin ({left},{top}) cell={est_cell}")
                for row in range(GRID_H):
                    cells_str = []
                    for col in range(GRID_W):
                        idx = (row * GRID_W + col) * 3
                        r_, g_, b_ = pixels[idx], pixels[idx+1], pixels[idx+2]
                        bits = f"{int(r_>127)}{int(g_>127)}{int(b_>127)}"
                        cells_str.append(bits)
                    print(f"    row {row}: " + " ".join(cells_str))
                pose = decode_strip(pixels)
                if pose:
                    abs_x = mon["left"] + left
                    abs_y = mon["top"] + top
                    print(f"  HIT at monitor {mi} ({left},{top}) abs ({abs_x},{abs_y})  cell={est_cell}")
                    print(f"       x={pose.x:+.3f} y={pose.y:+.3f} z={pose.z:+.3f} yaw={pose.yaw:6.2f}deg")
                    return 0, mi, abs_x, abs_y, est_cell
                else:
                    print(f"  markers ok at ({left},{top}) but full decode failed")
    print()
    print("No grid decoded. Likely causes:")
    print("  - shader not visible (HUD not on avatar / wrong avatar)")
    print("  - wrong --cell size (try doubling or halving)")
    print("  - render scale is fractional so cells are smudged")
    return 2, None, 0, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=DEFAULT_CELL_SIZE,
                    help=f"shader cell size in physical pixels (default {DEFAULT_CELL_SIZE})")
    ap.add_argument("--watch", action="store_true",
                    help="loop forever, printing decoded pose at 4 Hz")
    args = ap.parse_args()

    if not args.watch:
        result = scan_and_decode(args.cell)
        return result[0] if isinstance(result, tuple) else result

    # watch: find once then keep sampling that same spot
    result = scan_and_decode(args.cell)
    if not isinstance(result, tuple) or result[0] != 0:
        return result[0] if isinstance(result, tuple) else result
    _, mon_index, abs_x, abs_y, est_cell = result

    try:
        import mss
        import numpy as np
    except ImportError as e:
        print(f"ERROR: missing dep {e}")
        return 1

    with mss.mss() as sct:
        mon = sct.monitors[mon_index]
        region = {
            "left":   abs_x,
            "top":    abs_y,
            "width":  GRID_W * est_cell,
            "height": GRID_H * est_cell,
        }
        try:
            while True:
                shot = sct.grab(region)
                arr = np.array(shot, dtype=np.uint8)
                pixels = sample_grid(arr, 0, 0, est_cell)
                pose = decode_strip(pixels) if pixels else None
                if pose:
                    print(f"\rx={pose.x:+7.3f} y={pose.y:+7.3f} z={pose.z:+7.3f} yaw={pose.yaw:6.2f}deg    ",
                          end="", flush=True)
                else:
                    print("\r[decode failed]                                                ",
                          end="", flush=True)
                time.sleep(0.25)
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
