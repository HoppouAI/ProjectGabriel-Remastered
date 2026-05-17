"""Quick sniffer for the VRCRaycast OSC params, prints them as they arrive.

Run while wearing the sensor-rigged avatar in VRChat with OSC enabled. If
the main gabriel process is running it'll have port 9001 bound and you
need to stop it first, or pass a different port.

Usage:
    .venv\\Scripts\\python.exe scripts\\test_raycast_osc.py
    .venv\\Scripts\\python.exe scripts\\test_raycast_osc.py --port 9001 --seconds 15
"""
from __future__ import annotations

import argparse
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from src.raycast import RaycastState, forward_blocked, drop_ahead


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--all", action="store_true", help="print every avatar param, not just ray ones")
    args = ap.parse_args()

    state = RaycastState()
    dispatcher = Dispatcher()
    state.register_handlers(dispatcher, prefix="/avatar/parameters/")

    seen = set()

    if args.all:
        def _everything(addr, *vals):
            print(f"{addr} = {vals}")
        dispatcher.set_default_handler(_everything)

    server = ThreadingOSCUDPServer(("127.0.0.1", args.port), dispatcher)
    print(f"listening on 127.0.0.1:{args.port} for {args.seconds}s...")
    print("expecting params like Fwd_Hit, Fwd_Distance, Fwd_Ratio, Left_*, etc.")

    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    end = time.monotonic() + args.seconds
    last_print = 0.0
    while time.monotonic() < end:
        time.sleep(0.25)
        readings = state.get_all()
        for name in readings:
            seen.add(name)
        now = time.monotonic()
        if now - last_print > 1.0 and readings:
            last_print = now
            fwd = state.get("Fwd")
            fwd_dist = f"{fwd.distance:.2f}m" if fwd else "n/a"
            print(
                f"  rays={len(readings):2d}  fwd={'HIT' if fwd and fwd.hit else 'open'}"
                f" dist={fwd_dist}"
                f"  drop_ahead={'YES' if drop_ahead(state) else 'no'}"
                f"  fwd_blocked@1m={'YES' if forward_blocked(state, 1.0) else 'no'}"
            )

    server.shutdown()
    print()
    print(f"saw {len(seen)} unique ray names: {sorted(seen)}")
    print()
    print("per-ray latest readings:")
    for name in sorted(seen):
        r = state.get(name)
        if r is None:
            continue
        print(f"  {name:10s}  hit={r.hit!s:5}  dist={r.distance:6.2f}m  ratio={r.ratio:.3f}")


if __name__ == "__main__":
    main()
