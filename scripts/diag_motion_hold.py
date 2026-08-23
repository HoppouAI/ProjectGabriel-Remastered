# drive the FBT puppet live from the motion server while logging what breaks.
#
# same OSC path as test_motion_stream.py so it looks identical in game, but
# it holds ONE prompt and reports per-bucket health. the two param groups have
# different origins, which is what makes this diagnostic:
#   muscles (26) come from the model's global_rot_mats
#   hips (3)     come from the retargeter's STATEFUL root tracking
# so if only hips drift the bug is root tracking, if muscles pin to their
# limits the model rollout itself has diverged.
#
# locomotion axes are deliberately not driven: a held pose should not move
# him, and leaving them out keeps the test about pose only.
#
# usage: python scripts/diag_motion_hold.py [prompt] [--seconds N] [--server IP]

import sys
import csv
import json
import time
import asyncio
import argparse
import statistics
from pathlib import Path

from pythonosc.udp_client import SimpleUDPClient

OSC_IP, OSC_PORT = '127.0.0.1', 9000
PREFIX = '/avatar/parameters/FBT/'
SEND_HZ = 60.0
SMOOTH_TAU = 0.07
BUCKET_S = 10.0

MUSCLES = [
    'SpineFB', 'SpineLR', 'SpineTW',
    'HeadNod', 'HeadTilt', 'HeadTurn',
    'LArmUp', 'LArmFB', 'LArmTW', 'LElbow', 'LWristUD', 'LWristIO',
    'RArmUp', 'RArmFB', 'RArmTW', 'RElbow', 'RWristUD', 'RWristIO',
    'LLegFB', 'LLegIO', 'LKnee', 'LFootUD',
    'RLegFB', 'RLegIO', 'RKnee', 'RFootUD',
]
HIPS = ['HipsY', 'HipsPitch', 'HipsRoll']
PARAMS = MUSCLES + HIPS


class State:
    def __init__(self):
        self.target = {p: 0.0 for p in PARAMS}
        self.current = {p: 0.0 for p in PARAMS}
        self.frames = 0
        self.log = []          # (t, {param: raw target})
        self.t0 = time.monotonic()


async def sender(osc, st):
    import math
    interval = 1.0 / SEND_HZ
    last = time.monotonic()
    while True:
        now = time.monotonic()
        dt = max(1e-3, now - last)
        last = now
        alpha = 1.0 - math.exp(-dt / SMOOTH_TAU)
        for p in PARAMS:
            st.current[p] += (st.target[p] - st.current[p]) * alpha
            osc.send_message(PREFIX + p, float(st.current[p]))
        await asyncio.sleep(interval)


def bucket_stats(rows, names):
    """jitter, saturation and range for one param group over a bucket."""
    jerks, sat, peak = [], 0, 0.0
    total = 0
    for p in names:
        series = [r[p] for r in rows]
        peak = max(peak, max(abs(v) for v in series))
        sat += sum(1 for v in series if abs(v) > 0.98)
        total += len(series)
        if len(series) > 2:
            d2 = [series[i] - 2 * series[i - 1] + series[i - 2]
                  for i in range(2, len(series))]
            jerks.append(statistics.fmean(abs(x) for x in d2))
    return {
        'jitter': statistics.fmean(jerks) if jerks else 0.0,
        'sat': sat / total if total else 0.0,
        'peak': peak,
    }


async def run(args):
    import websockets

    osc = SimpleUDPClient(OSC_IP, OSC_PORT)
    st = State()
    url = f'ws://{args.server}:8765'
    print(f'connecting to {url} ...')
    async with websockets.connect(url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        print(f"connected: {hello.get('backend')}/{hello.get('model')} "
              f"@{hello.get('fps')}fps")
        await ws.send(json.dumps({'type': 'prompt', 'text': args.prompt}))
        print(f'holding: {args.prompt!r} for {args.seconds:.0f}s\n')

        osc.send_message(PREFIX + 'Enable', True)
        asyncio.create_task(sender(osc, st))
        print(f"{'t':>6} | {'MUSCLES jit':>11} {'sat':>6} {'peak':>6} "
              f"| {'HIPS jit':>9} {'sat':>6} {'peak':>6} | fps")

        rows = []
        bucket_start = time.monotonic()
        deadline = bucket_start + args.seconds
        n_bucket = 0
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                print('  !! no frames for 5s, server stalled')
                continue
            msg = json.loads(raw)
            if msg.get('type') != 'frame':
                continue
            p = msg['params']
            for k in PARAMS:
                if k in p:
                    st.target[k] = float(p[k])
            rows.append({k: st.target[k] for k in PARAMS})
            st.frames += 1
            n_bucket += 1
            st.log.append((time.monotonic() - st.t0, dict(rows[-1])))

            now = time.monotonic()
            if now - bucket_start >= BUCKET_S:
                m = bucket_stats(rows, MUSCLES)
                h = bucket_stats(rows, HIPS)
                fps = n_bucket / (now - bucket_start)
                print(f'{now - st.t0:6.0f} | {m["jitter"]:11.4f} {m["sat"]:6.1%} '
                      f'{m["peak"]:6.2f} | {h["jitter"]:9.4f} {h["sat"]:6.1%} '
                      f'{h["peak"]:6.2f} | {fps:4.1f}')
                rows, bucket_start, n_bucket = [], now, 0

    osc.send_message(PREFIX + 'Enable', False)
    out = Path(__file__).parent.parent / 'data' / 'motion_hold_log.csv'
    out.parent.mkdir(exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t'] + PARAMS)
        for t, vals in st.log:
            w.writerow([f'{t:.3f}'] + [f'{vals[p]:.4f}' for p in PARAMS])
    print(f'\nwrote {len(st.log)} frames to {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prompt', nargs='*', default=['a', 'person', 'sits', 'on', 'the', 'ground'])
    ap.add_argument('--seconds', type=float, default=300.0)
    ap.add_argument('--server', default='127.0.0.1')
    args = ap.parse_args()
    args.prompt = ' '.join(args.prompt)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
