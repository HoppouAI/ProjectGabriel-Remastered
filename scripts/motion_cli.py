# interactive console for driving the FBT puppet from the motion server.
#
# type a prompt to play it, slash commands for everything else. same OSC path
# as test_motion_stream.py (60hz smoothed muscles + analog locomotion), but
# built to sit in and poke at rather than run once.
#
# run: python scripts/motion_cli.py [server_ip]

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
SMOOTH_TAU = 0.07        # muscle smoothing time constant
LOCO_TAU = 0.4           # smooth axes over most of a stride so speed doesnt pulse
WALK_FULL_MPS = 2.0      # vrchat desktop walk speed at full stick
TURN_FULL_RADS = 1.8     # yaw rate at full LookHorizontal
AXIS_DEADZONE = 0.3      # vrchat ignores axes below this, snap up to it
AXIS_CUTOFF = 0.15       # below this desired value just send 0

PARAMS = [
    "SpineFB", "SpineLR", "SpineTW",
    "HeadNod", "HeadTilt", "HeadTurn",
    "LArmUp", "LArmFB", "LArmTW", "LElbow", "LWristUD", "LWristIO",
    "RArmUp", "RArmFB", "RArmTW", "RElbow", "RWristUD", "RWristIO",
    "LLegFB", "LLegIO", "LKnee", "LFootUD",
    "RLegFB", "RLegIO", "RKnee", "RFootUD",
    "HipsY", "HipsPitch", "HipsRoll",
]

HELP = """
  <text>              play that prompt (looped until you change it)
  /once <text>        play once, then settle back to idle
  /seq a | b | c      play back to back, advancing as each one lands
  /seqloop a | b | c  same but hold the last step forever
  /stop               back to idle
  /reset              forget the rollout, respawn at the origin

  /on  /off  /toggle  the FBT puppet itself
  /loco               drive him through the world, or walk in place
  /status             what everything is currently doing

  /hist <s>           history fed back per step. ardy trains on 10s windows,
                      too short and he forgets the pose and drifts. try 7.2
  /steps <n>          denoising steps, 1-10. lower = faster, rougher
  /post               foot contact correction on/off (plants his feet)
  /margin <m>         how hard the correction pins the root down (0.04)
  /tune               show the current settings
  /help  /quit
"""


def snap_axis(v):
    """vrchat input axes have a dead zone, snap small-but-real values to it."""
    if abs(v) < AXIS_CUTOFF:
        return 0.0
    return math.copysign(min(1.0, max(AXIS_DEADZONE, abs(v))), v)


class State:
    def __init__(self):
        self.target = {p: 0.0 for p in PARAMS}
        self.current = {p: 0.0 for p in PARAMS}
        self.vfwd = self.vside = self.vyaw = 0.0
        self.got_frame = False
        self.frames = 0
        self.last_frame_at = 0.0
        self.enabled = True
        self.loco = True
        self.post = True
        self.prompt = "(none)"
        self.fps = 0.0


async def sender(osc, st):
    interval = 1.0 / SEND_HZ
    last = time.monotonic()
    sv = sh = sl = 0.0
    while True:
        now = time.monotonic()
        dt = max(1e-3, now - last)
        last = now
        alpha = 1.0 - math.exp(-dt / SMOOTH_TAU)
        beta = 1.0 - math.exp(-dt / LOCO_TAU)

        if st.got_frame and st.enabled:
            for p in PARAMS:
                st.current[p] += (st.target[p] - st.current[p]) * alpha
                osc.send_message(PREFIX + p, float(st.current[p]))

            if st.loco:
                sv += (st.vfwd / WALK_FULL_MPS - sv) * beta
                sh += (st.vside / WALK_FULL_MPS - sh) * beta
                sl += (st.vyaw / TURN_FULL_RADS - sl) * beta
                osc.send_message("/input/Vertical", float(snap_axis(sv)))
                osc.send_message("/input/Horizontal", float(snap_axis(sh)))
                osc.send_message("/input/LookHorizontal", float(max(-1.0, min(1.0, sl))))
            else:
                sv = sh = sl = 0.0

        await asyncio.sleep(max(0.0, interval - (time.monotonic() - now)))


def zero_inputs(osc):
    for a in ("Vertical", "Horizontal", "LookHorizontal"):
        osc.send_message("/input/" + a, 0.0)


def split_steps(rest):
    return [s.strip() for s in rest.split("|") if s.strip()]


async def handle(ws, osc, st, line):
    """returns False to quit."""
    if not line:
        return True
    if not line.startswith("/"):
        st.prompt = line
        await ws.send(json.dumps({"type": "prompt", "text": line}))
        print(f"  -> {line!r}")
        return True

    cmd, _, rest = line[1:].partition(" ")
    cmd, rest = cmd.lower(), rest.strip()

    if cmd in ("q", "quit", "exit"):
        return False
    if cmd in ("h", "help", "?"):
        print(HELP)
    elif cmd == "once":
        if not rest:
            print("  usage: /once <prompt>")
        else:
            st.prompt = rest + " (once)"
            await ws.send(json.dumps({"type": "prompt", "text": rest, "once": True}))
            print(f"  -> once {rest!r}")
    elif cmd in ("seq", "seqloop"):
        steps = split_steps(rest)
        if len(steps) < 2:
            print("  usage: /seq walks forward | sits down | waves")
        else:
            loop_last = cmd == "seqloop"
            st.prompt = " -> ".join(steps) + (" (hold last)" if loop_last else "")
            await ws.send(json.dumps({"type": "sequence", "steps": steps,
                                      "loop_last": loop_last}))
            print(f"  -> {len(steps)} steps{' holding the last' if loop_last else ''}")
    elif cmd == "stop":
        st.prompt = "(idle)"
        await ws.send(json.dumps({"type": "stop"}))
        print("  -> idle")
    elif cmd == "reset":
        st.prompt = "(idle)"
        st.got_frame = False
        await ws.send(json.dumps({"type": "reset"}))
        print("  -> rollout reset (paused, send a prompt to start again)")
    elif cmd in ("on", "off", "toggle"):
        st.enabled = (cmd == "on") if cmd != "toggle" else not st.enabled
        osc.send_message(PREFIX + "Enable", bool(st.enabled))
        if not st.enabled:
            zero_inputs(osc)
        print(f"  -> puppet {'ON' if st.enabled else 'OFF'}")
    elif cmd == "loco":
        st.loco = not st.loco
        if not st.loco:
            zero_inputs(osc)
        print(f"  -> locomotion {'ON (he walks around)' if st.loco else 'OFF (walks in place)'}")
    elif cmd in ("hist", "steps", "margin", "contact", "reanchor"):
        try:
            val = float(rest)
        except ValueError:
            print(f"  usage: /{cmd} <number>")
            return True
        key = {"hist": "history", "steps": "steps", "margin": "root_margin",
               "contact": "contact_threshold", "reanchor": "reanchor"}[cmd]
        await ws.send(json.dumps({"type": "tune", key: val}))
    elif cmd == "post":
        st.post = not st.post
        await ws.send(json.dumps({"type": "tune", "postprocess": st.post}))
    elif cmd == "tune":
        await ws.send(json.dumps({"type": "tune"}))
    elif cmd in ("s", "status"):
        age = time.monotonic() - st.last_frame_at if st.last_frame_at else -1
        stale = "  STREAM STALLED" if age > 1.0 else ""
        print(f"  puppet={'ON' if st.enabled else 'OFF'}  loco={'ON' if st.loco else 'OFF'}  "
              f"{st.fps:.1f}fps  frames={st.frames}{stale}\n"
              f"  prompt: {st.prompt}\n"
              f"  HipsY={st.target['HipsY']:+.2f}  vfwd={st.vfwd:+.2f}m/s  vyaw={st.vyaw:+.2f}rad/s")
    else:
        print(f"  unknown command {cmd!r}, /help for the list")
    return True


async def main():
    import websockets

    server = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    uri = f"ws://{server}:8765"
    osc = SimpleUDPClient(OSC_IP, OSC_PORT)
    st = State()

    print(f"connecting to {uri} ...")
    try:
        ws = await websockets.connect(uri, max_size=None)
    except OSError as e:
        print(f"could not reach the motion server ({e}).\n"
              f"start it with: motion_server\\.venv-ardy\\Scripts\\python.exe "
              f"motion_server\\server.py --model core8")
        return

    async with ws:
        hello = json.loads(await ws.recv())
        print(f"connected: {hello.get('backend')}/{hello.get('model')} @{hello.get('fps')}fps")
        osc.send_message(PREFIX + "Enable", True)
        zero_inputs(osc)
        print(HELP)

        loop = asyncio.get_running_loop()
        quit_ev = asyncio.Event()

        def stdin_reader():
            for raw in sys.stdin:
                fut = asyncio.run_coroutine_threadsafe(
                    handle(ws, osc, st, raw.strip()), loop)
                try:
                    if not fut.result(10):
                        loop.call_soon_threadsafe(quit_ev.set)
                        return
                except Exception as e:
                    print(f"  command failed: {e}")
                print("> ", end="", flush=True)
            loop.call_soon_threadsafe(quit_ev.set)

        threading.Thread(target=stdin_reader, daemon=True).start()
        send_task = asyncio.create_task(sender(osc, st))

        async def receiver():
            tick, last_tick = 0, time.monotonic()
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "tuning":
                    t = msg["tuning"]
                    st.post = bool(t.get("postprocess", st.post))
                    print(f"\n  history={t['history_s']}s ({t['history_frames']}f)  "
                          f"steps={t['steps']}  postprocess={t['postprocess']}  "
                          f"margin={t['root_margin']}  contact={t['contact_threshold']}  "
                          f"reanchor={t['reanchor_s']}s\n> ", end="", flush=True)
                    continue
                if msg.get("type") != "frame":
                    continue
                params = msg.get("params") or {}
                for p in PARAMS:
                    if p in params:
                        st.target[p] = float(params[p])
                st.vfwd = float(params.get("_vfwd", 0.0))
                st.vside = float(params.get("_vside", 0.0))
                st.vyaw = float(params.get("_vyaw", 0.0))
                if not st.got_frame and params:
                    st.current.update(st.target)  # snap to the first frame
                    st.got_frame = True
                st.frames += 1
                st.last_frame_at = time.monotonic()
                tick += 1
                if tick >= 40:
                    now = time.monotonic()
                    st.fps = tick / max(1e-3, now - last_tick)
                    tick, last_tick = 0, now

        recv_task = asyncio.create_task(receiver())
        print("> ", end="", flush=True)
        try:
            done, _ = await asyncio.wait(
                [asyncio.create_task(quit_ev.wait()), recv_task],
                return_when=asyncio.FIRST_COMPLETED)
        finally:
            send_task.cancel()
            recv_task.cancel()
            zero_inputs(osc)
            osc.send_message(PREFIX + "Enable", False)
            print("\npuppet released, inputs zeroed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
