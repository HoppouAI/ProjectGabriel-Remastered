# stream frames from the motion server into VRChat's FBT puppet rig, with
# client side smoothing and a locomotion layer that drives the capsule.
#
# frames arrive at 30fps, muscles are exponentially smoothed and sent at
# 60hz so nothing steps visibly. body frame velocities from the server map
# to VRChat analog inputs (Vertical/Horizontal/LookHorizontal) so walking
# motions actually move him through the world, turning included.
#
# run: python scripts/test_motion_stream.py [server_ip] [prompt]
# type new prompts at the console to switch motion, ctrl+c to quit.

import sys
import json
import math
import time
import asyncio
import threading

from pythonosc.udp_client import SimpleUDPClient

OSC_IP, OSC_PORT = "127.0.0.1", 9000
PREFIX = "/avatar/parameters/FBT/"

SEND_HZ = 60.0
SMOOTH_TAU = 0.07          # seconds, muscle smoothing time constant
LOCO_TAU = 0.4             # axes smooth over most of a stride so speed doesnt pulse per step
WALK_FULL_MPS = 2.0        # vrchat desktop walk speed at full stick
TURN_FULL_RADS = 1.8       # yaw rate at full LookHorizontal, tune to look sensitivity
AXIS_DEADZONE = 0.3        # vrchat ignores axes below this, snap up to it
AXIS_CUTOFF = 0.15         # below this desired value just send 0

PARAMS = [
    "SpineFB", "SpineLR", "SpineTW",
    "HeadNod", "HeadTilt", "HeadTurn",
    "LArmUp", "LArmFB", "LArmTW", "LElbow", "LWristUD", "LWristIO",
    "RArmUp", "RArmFB", "RArmTW", "RElbow", "RWristUD", "RWristIO",
    "LLegFB", "LLegIO", "LKnee", "LFootUD",
    "RLegFB", "RLegIO", "RKnee", "RFootUD",
    "HipsY", "HipsPitch", "HipsRoll",
]


def snap_axis(v):
    """vrchat input axes have a dead zone, snap small-but-real values to it."""
    if abs(v) < AXIS_CUTOFF:
        return 0.0
    return math.copysign(min(1.0, max(AXIS_DEADZONE, abs(v))), v)


class StreamState:
    def __init__(self):
        self.target = {p: 0.0 for p in PARAMS}
        self.current = {p: 0.0 for p in PARAMS}
        self.vfwd = 0.0
        self.vside = 0.0
        self.vyaw = 0.0
        self.got_frame = False
        self.frames = 0


async def sender(osc, st):
    """60hz: smooth muscles toward targets, map velocities to input axes."""
    interval = 1.0 / SEND_HZ
    last = time.monotonic()
    sv = sh = sl = 0.0  # smoothed axes
    while True:
        now = time.monotonic()
        dt = max(1e-3, now - last)
        last = now
        alpha = 1.0 - math.exp(-dt / SMOOTH_TAU)
        beta = 1.0 - math.exp(-dt / LOCO_TAU)

        if st.got_frame:
            for p in PARAMS:
                cur = st.current[p]
                cur += (st.target[p] - cur) * alpha
                st.current[p] = cur
                osc.send_message(PREFIX + p, float(cur))

            sv += (st.vfwd / WALK_FULL_MPS - sv) * beta
            sh += (st.vside / WALK_FULL_MPS - sh) * beta
            sl += (st.vyaw / TURN_FULL_RADS - sl) * beta
            osc.send_message("/input/Vertical", float(snap_axis(sv)))
            osc.send_message("/input/Horizontal", float(snap_axis(sh)))
            osc.send_message("/input/LookHorizontal", float(max(-1.0, min(1.0, sl))))

        await asyncio.sleep(max(0.0, interval - (time.monotonic() - now)))


def zero_inputs(osc):
    osc.send_message("/input/Vertical", 0.0)
    osc.send_message("/input/Horizontal", 0.0)
    osc.send_message("/input/LookHorizontal", 0.0)


async def main():
    import websockets

    server = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "stand"
    uri = f"ws://{server}:8765"

    osc = SimpleUDPClient(OSC_IP, OSC_PORT)
    osc.send_message(PREFIX + "Enable", True)
    zero_inputs(osc)
    print(f"connecting to {uri}")

    st = StreamState()

    async with websockets.connect(uri, max_size=None) as ws:
        print(f"connected, prompt: {prompt!r}")
        await ws.send(json.dumps({"type": "prompt", "text": prompt}))

        loop = asyncio.get_running_loop()

        def stdin_reader():
            for line in sys.stdin:
                text = line.strip()
                if text:
                    asyncio.run_coroutine_threadsafe(
                        ws.send(json.dumps({"type": "prompt", "text": text})), loop)
                    print(f"prompt -> {text!r}")

        threading.Thread(target=stdin_reader, daemon=True).start()
        send_task = asyncio.create_task(sender(osc, st))

        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") != "frame":
                    continue
                params = msg.get("params")
                if not params:
                    if st.frames == 0:
                        print("server is streaming raw only (no muscle_ranges.json), nothing to send")
                    st.frames += 1
                    continue
                for p in PARAMS:
                    if p in params:
                        st.target[p] = float(params[p])
                st.vfwd = float(params.get("_vfwd", 0.0))
                st.vside = float(params.get("_vside", 0.0))
                st.vyaw = float(params.get("_vyaw", 0.0))
                if not st.got_frame:
                    # jump straight to the first frame instead of easing from zero
                    st.current.update(st.target)
                    st.got_frame = True
                st.frames += 1
                if st.frames % 150 == 0:
                    print(f"frame {msg['t']}  vfwd={st.vfwd:+.2f}m/s  vyaw={st.vyaw:+.2f}rad/s  "
                          f"HipsY={params.get('HipsY', 0):+.2f}")
        finally:
            send_task.cancel()
            zero_inputs(osc)
            osc.send_message(PREFIX + "Enable", False)
            print("puppet disabled, inputs zeroed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
