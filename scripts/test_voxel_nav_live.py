"""Live test: pose decoder -> voxel nav trail learning.

Auto-finds the on-screen pose strip, then starts PoseExfilReader and
feeds every successful pose into a VoxelNavManager. Prints a live stats
line showing current voxel, total learned cells, and bounding box.

Walk around in VRChat and watch the graph fill up. Press Ctrl+C to stop;
a snapshot is saved to data/voxel_nav/<world_id>.json so you can reload.

Usage:
    .venv\\Scripts\\python.exe scripts\\test_voxel_nav_live.py
    .venv\\Scripts\\python.exe scripts\\test_voxel_nav_live.py --world myroom --plan-back
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pose_decoder import GRID_W, GRID_H, PoseExfilReader  # noqa: E402
from src.voxel_nav import VoxelNavManager, world_to_serial    # noqa: E402

# reuse the strip-finder from the pose decoder live test
from scripts.test_pose_decoder_live import scan_and_decode    # noqa: E402


def _bbox(serials):
    if not serials:
        return None
    xs = [s[0] for s in serials]
    ys = [s[1] for s in serials]
    zs = [s[2] for s in serials]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="livetest",
                    help="world id label for the saved graph (default 'livetest')")
    ap.add_argument("--data-dir", default="data/voxel_nav")
    ap.add_argument("--plan-back", action="store_true",
                    help="on exit, plan a path from current voxel back to first observed")
    args = ap.parse_args()

    print("looking for pose strip on screen...")
    result = scan_and_decode(8)  # cell size hint; auto-detection picks the real one
    if not isinstance(result, tuple) or result[0] != 0:
        print("could not find the pose strip. is VRChat running with the shader on?")
        return 1
    _, mon_index, abs_x, abs_y, est_cell = result
    print(f"  strip at monitor {mon_index} ({abs_x},{abs_y}) cell={est_cell}")

    region = {
        "left":   abs_x,
        "top":    abs_y,
        "width":  GRID_W * est_cell,
        "height": GRID_H * est_cell,
    }
    reader = PoseExfilReader(region=region, cell_size=est_cell,
                             poll_hz=20.0, monitor_index=mon_index)
    reader.start()

    nav = VoxelNavManager(data_dir=args.data_dir, learning_mode=True)
    nav.load_world(args.world)
    print(f"loaded world '{args.world}' -- {len(nav.graph)} nodes already known")

    first_serial = None
    last_decode_time = 0.0
    last_print = 0.0
    last_flush = time.time()
    try:
        print("learning... walk around in VRChat. Ctrl+C to stop.")
        while True:
            pose = reader.get()
            if pose is not None and pose.timestamp != last_decode_time:
                last_decode_time = pose.timestamp
                node = nav.observe(pose.x, pose.y, pose.z, grounded=True)
                if first_serial is None:
                    first_serial = node.serial

            now = time.time()
            if now - last_print >= 0.25:
                last_print = now
                stats = reader.stats()
                bb = _bbox(list(nav.graph.nodes.keys()))
                cur = nav.current.serial if nav.current else "?"
                if bb:
                    (lo, hi) = bb
                    span_x = (hi[0] - lo[0]) * 0.25
                    span_z = (hi[2] - lo[2]) * 0.25
                    span_y = (hi[1] - lo[1]) * 0.25
                    line = (f"\rcells={len(nav.graph):4d}  current={cur}  "
                            f"span x={span_x:5.2f}m y={span_y:4.2f}m z={span_z:5.2f}m  "
                            f"decode={stats['decode_rate']*100:5.1f}%  ")
                else:
                    line = (f"\rcells={len(nav.graph):4d}  waiting for pose...  "
                            f"decode={stats['decode_rate']*100:5.1f}%  ")
                print(line, end="", flush=True)
            # periodic auto-save so a hard kill still keeps the trail
            if time.time() - last_flush >= 5.0:
                nav.flush()
                last_flush = time.time()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        reader.stop()
        nav.flush()
        print(f"saved {len(nav.graph)} nodes to {args.data_dir}/{args.world}.json")

    if args.plan_back and first_serial is not None and nav.current is not None:
        from src.voxel_nav import serial_to_center, find_path_astar
        cx, cy, cz = serial_to_center(first_serial)
        print(f"planning path back to start voxel {first_serial} ...")
        result = find_path_astar(nav.graph, nav.current.serial, first_serial)
        if not result.found:
            print("  NO PATH FOUND -- trail may have disconnected (try walking continuously)")
        else:
            print(f"  found path: {len(result.full_serials)} cells, "
                  f"{len(result.serials)} turn points, cost={result.cost:.2f}")
            for i, s in enumerate(result.serials):
                wx, wy, wz = serial_to_center(s)
                print(f"    {i+1:2d}. voxel {s} -> world ({wx:+.2f}, {wy:+.2f}, {wz:+.2f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
