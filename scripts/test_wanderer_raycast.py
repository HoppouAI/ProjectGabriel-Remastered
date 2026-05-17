"""Test the wanderer's raycast safety override in isolation.

Sends a constant forward input through `_raycast_safety_override` every
tick and prints what the override decides. Lets you walk the avatar
toward a wall/ledge and confirm the override clamps forward and biases
turn correctly, WITHOUT loading the depth model.

This does NOT actually move the avatar (no /input/Vertical is sent),
it just shows the would-be output. Set --send to actually drive movement.

Usage:
    .venv\\Scripts\\python.exe scripts\\test_wanderer_raycast.py
    .venv\\Scripts\\python.exe scripts\\test_wanderer_raycast.py --seconds 30 --send
"""
from __future__ import annotations

import argparse
import time
import types

from src.vrchat import VRChatOSC
from src.wanderer import Wanderer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--send", action="store_true", help="actually send /input/Vertical to VRChat")
    ap.add_argument("--osc-ip", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=9000)
    ap.add_argument("--osc-receive-port", type=int, default=9001)
    args = ap.parse_args()

    cfg = types.SimpleNamespace(
        osc_ip=args.osc_ip,
        osc_port=args.osc_port,
        osc_receive_port=args.osc_receive_port,
    )
    osc = VRChatOSC(cfg)
    wanderer = Wanderer(config=cfg, osc=osc)

    print(f"listening on :{args.osc_receive_port}, sending to {args.osc_ip}:{args.osc_port}")
    print(f"send mode: {'ON (avatar will walk)' if args.send else 'OFF (decisions only)'}")
    print("baseline input: forward=0.6, turn=0.0")
    print()

    end = time.monotonic() + args.seconds
    last_print = 0.0
    while time.monotonic() < end:
        time.sleep(0.1)
        t_in, f_in = 0.0, 0.6
        t_out, f_out = wanderer._raycast_safety_override(t_in, f_in)
        now = time.monotonic()
        if now - last_print > 0.5:
            last_print = now
            state = osc.raycast_state
            fwd = state.get("Fwd")
            near = state.get("FwdNear")
            drop = state.get("DropFwd")
            fwd_s = f"{fwd.distance:.2f}m{'H' if fwd.hit else 'o'}" if fwd else "n/a"
            near_s = f"{near.distance:.2f}m{'H' if near.hit else 'o'}" if near else "n/a"
            drop_s = f"{drop.distance:.2f}m{'H' if drop.hit else 'o'}" if drop else "n/a"
            verdict = "GO  " if f_out > 0.0 else ("STOP" if f_out == 0.0 else "BACK")
            print(
                f"  fwd={fwd_s:>10}  near={near_s:>10}  drop={drop_s:>10}  ->  "
                f"f={f_out:+.2f} t={t_out:+.2f}  {verdict}"
            )
        if args.send:
            client = osc.client
            client.send_message("/input/Vertical", float(max(-1, min(1, f_out))))
            client.send_message("/input/LookHorizontal", float(max(-1, min(1, t_out))))

    if args.send:
        client = osc.client
        client.send_message("/input/Vertical", 0.0)
        client.send_message("/input/LookHorizontal", 0.0)
    print("done.")


if __name__ == "__main__":
    main()
