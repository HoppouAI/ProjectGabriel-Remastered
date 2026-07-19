"""Client for the DART motion server.

Connects over websocket, receives 30fps retargeted FBT param frames, smooths
them at 60hz onto VRChat OSC, and maps the body frame velocities to analog
move/turn inputs so generated walking actually moves him through the world.
Owned by the motion tools, only connects when a motion is first requested.
"""

import json
import math
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

PREFIX = "/avatar/parameters/FBT/"

SEND_HZ = 60.0
SMOOTH_TAU = 0.07   # muscle smoothing time constant, seconds
LOCO_TAU = 0.4      # axes smooth over most of a stride so speed doesnt pulse per step
AXIS_DEADZONE = 0.3
AXIS_CUTOFF = 0.15

PARAMS = [
    "SpineFB", "SpineLR", "SpineTW",
    "HeadNod", "HeadTilt", "HeadTurn",
    "LArmUp", "LArmFB", "LArmTW", "LElbow", "LWristUD", "LWristIO",
    "RArmUp", "RArmFB", "RArmTW", "RElbow", "RWristUD", "RWristIO",
    "LLegFB", "LLegIO", "LKnee", "LFootUD",
    "RLegFB", "RLegIO", "RKnee", "RFootUD",
    "HipsY", "HipsPitch", "HipsRoll",
]


def _snap_axis(v):
    if abs(v) < AXIS_CUTOFF:
        return 0.0
    return math.copysign(min(1.0, max(AXIS_DEADZONE, abs(v))), v)


class MotionClient:
    def __init__(self, osc_client, host, port, walk_full=2.0, turn_full=1.8):
        self._osc = osc_client
        self._uri = f"ws://{host}:{port}"
        self._ws = None
        self._recv_task = None
        self._send_task = None
        self._timer_task = None
        self._connected = asyncio.Event()
        self._active = False  # puppet enabled and streaming to osc
        self._target = {p: 0.0 for p in PARAMS}
        self._current = {p: 0.0 for p in PARAMS}
        self._vfwd = self._vside = self._vyaw = 0.0
        self._got_frame = False
        self._walk_full = walk_full
        self._turn_full = turn_full
        self.current_prompt = None

    # -- lifecycle --

    async def ensure_connected(self, timeout=8.0):
        if self._recv_task is None or self._recv_task.done():
            self._connected.clear()
            self._recv_task = asyncio.create_task(self._receiver())
        if self._send_task is None or self._send_task.done():
            self._send_task = asyncio.create_task(self._sender())
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
        except asyncio.TimeoutError:
            raise ConnectionError(f"motion server not reachable at {self._uri}")

    async def shutdown(self):
        self._cancel_timer()
        for t in (self._recv_task, self._send_task):
            if t is not None:
                t.cancel()
        self._recv_task = self._send_task = None
        if self._active:
            self._set_active(False)

    # -- commands --

    async def play(self, prompt, seconds=None):
        await self.ensure_connected()
        self._cancel_timer()
        await self._send({"type": "prompt", "text": prompt})
        self.current_prompt = prompt
        if not self._active:
            self._set_active(True)
        if seconds is not None and seconds > 0:
            self._timer_task = asyncio.create_task(self._auto_stop(float(seconds)))

    async def stop_motion(self):
        """back to a generated standing idle, puppet stays up."""
        self._cancel_timer()
        if self._ws is None:
            return
        await self._send({"type": "prompt", "text": "stand"})
        self.current_prompt = "stand"

    async def reset(self):
        """wipe the motion models context and release the body back to vrchat."""
        self._cancel_timer()
        if self._ws is not None:
            await self._send({"type": "reset"})
        self.current_prompt = None
        self._got_frame = False
        if self._active:
            self._set_active(False)

    # -- internals --

    async def _auto_stop(self, seconds):
        try:
            await asyncio.sleep(seconds)
            logger.info(f"motion timer elapsed ({seconds:.0f}s), back to stand")
            await self.stop_motion()
        except asyncio.CancelledError:
            pass

    def _cancel_timer(self):
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

    async def _send(self, obj):
        if self._ws is None:
            raise ConnectionError("motion server not connected")
        await self._ws.send(json.dumps(obj))

    def _set_active(self, on):
        self._active = on
        self._osc.send_message(PREFIX + "Enable", bool(on))
        if not on:
            self._zero_inputs()

    def _zero_inputs(self):
        self._osc.send_message("/input/Vertical", 0.0)
        self._osc.send_message("/input/Horizontal", 0.0)
        self._osc.send_message("/input/LookHorizontal", 0.0)

    async def _receiver(self):
        import websockets
        while True:
            try:
                async with websockets.connect(self._uri, max_size=None) as ws:
                    self._ws = ws
                    self._connected.set()
                    logger.info(f"motion server connected: {self._uri}")
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if msg.get("type") != "frame":
                            continue
                        params = msg.get("params") or {}
                        for p in PARAMS:
                            if p in params:
                                self._target[p] = float(params[p])
                        self._vfwd = float(params.get("_vfwd", 0.0))
                        self._vside = float(params.get("_vside", 0.0))
                        self._vyaw = float(params.get("_vyaw", 0.0))
                        if not self._got_frame:
                            self._current.update(self._target)
                            self._got_frame = True
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"motion server connection lost ({e}), retrying in 3s")
            self._ws = None
            self._connected.clear()
            self._got_frame = False
            await asyncio.sleep(3.0)

    async def _sender(self):
        interval = 1.0 / SEND_HZ
        last = time.monotonic()
        sv = sh = sl = 0.0
        try:
            while True:
                now = time.monotonic()
                dt = max(1e-3, now - last)
                last = now
                alpha = 1.0 - math.exp(-dt / SMOOTH_TAU)
                beta = 1.0 - math.exp(-dt / LOCO_TAU)

                if self._active and self._got_frame:
                    for p in PARAMS:
                        cur = self._current[p]
                        cur += (self._target[p] - cur) * alpha
                        self._current[p] = cur
                        self._osc.send_message(PREFIX + p, float(cur))
                    sv += (self._vfwd / self._walk_full - sv) * beta
                    sh += (self._vside / self._walk_full - sh) * beta
                    sl += (self._vyaw / self._turn_full - sl) * beta
                    self._osc.send_message("/input/Vertical", float(_snap_axis(sv)))
                    self._osc.send_message("/input/Horizontal", float(_snap_axis(sh)))
                    self._osc.send_message("/input/LookHorizontal", float(max(-1.0, min(1.0, sl))))
                else:
                    sv = sh = sl = 0.0

                await asyncio.sleep(max(0.0, interval - (time.monotonic() - now)))
        except asyncio.CancelledError:
            pass
