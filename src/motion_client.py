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
LOCO_TAU = 0.4      # axes smooth over most of a stride so speed doesnt pulse per step
AXIS_DEADZONE = 0.3
AXIS_CUTOFF = 0.15

# one euro filter for the muscle params: cutoff rises with signal speed, so a
# still pose gets heavy smoothing (idle jitter would otherwise flicker across
# vrchats 8-bit sync steps for remote viewers) while fast motion stays snappy.
EURO_MIN_CUTOFF = 1.0   # hz at rest
EURO_BETA = 3.0         # cutoff gain per unit/s of speed
EURO_D_CUTOFF = 1.0     # hz, derivative low-pass

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


def _lp_alpha(cutoff, dt):
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuro:
    def __init__(self):
        self._x = None
        self._dx = 0.0

    def reset(self):
        self._x = None
        self._dx = 0.0

    def __call__(self, x, dt):
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) / dt
        self._dx += (dx - self._dx) * _lp_alpha(EURO_D_CUTOFF, dt)
        cutoff = EURO_MIN_CUTOFF + EURO_BETA * abs(self._dx)
        self._x += (x - self._x) * _lp_alpha(cutoff, dt)
        return self._x


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PoseTracker:
    """closed loop locomotion using the pose exfil shader strip.

    anchors darts world to vrchats world when tracking starts, then servos
    the position/heading error on top of the velocity feedforward so the
    avatar follows the generated trajectory 1:1 instead of dead reckoning.
    dart is z-up (ground = x/y), vrchat is y-up (ground = x/z), both yaws
    increase turning right, so the mapping is a plain 2d rotation.
    """

    KP_POS = 1.5        # m/s of correction per meter of error
    KP_YAW = 2.0        # rad/s per rad
    CORR_MAX = 1.2      # cap position correction so a desync cant send him sprinting
    YAW_CORR_MAX = 1.5
    STALE_S = 0.6       # ignore poses older than this (decode hiccup)
    DROP_ANCHOR_S = 3.0 # long blind stretch, re-anchor on the next good pose
    REANCHOR_DIST = 5.0 # error this big means a teleport/respawn, dont chase it

    def __init__(self, monitor_index=1):
        self._monitor = max(1, int(monitor_index))
        self._reader = None
        self._anchor = None  # (dart_x, dart_y, dart_yaw, vr_x, vr_z, vr_yaw_rad)

    def start(self):
        if self._reader is not None:
            return
        try:
            from src.pose_decoder import PoseExfilReader
            self._reader = PoseExfilReader(poll_hz=20.0, monitor_index=self._monitor)
            self._reader.start()
            logger.info("motion pose tracking: exfil reader started")
        except Exception as e:
            logger.warning(f"motion pose tracking unavailable ({e}), falling back to open loop")
            self._reader = None

    def stop(self):
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        self._anchor = None

    def reset_anchor(self):
        self._anchor = None

    def correction(self, dart_x, dart_y, dart_yaw, now):
        """returns (dv_fwd, dv_side, dv_yaw) to add to the feedforward, in
        m/s / m/s / rad/s body frame. zeros when blind or just anchored."""
        if self._reader is None:
            return 0.0, 0.0, 0.0
        pose = self._reader.get()
        if pose is None:
            return 0.0, 0.0, 0.0
        age = now - pose.timestamp
        if age > self.STALE_S:
            if age > self.DROP_ANCHOR_S:
                self._anchor = None
            return 0.0, 0.0, 0.0

        vr_yaw = math.radians(pose.yaw)
        if self._anchor is None:
            self._anchor = (dart_x, dart_y, dart_yaw, pose.x, pose.z, vr_yaw)
            return 0.0, 0.0, 0.0

        ax, ay, ayaw, vx0, vz0, vyaw0 = self._anchor
        # dart displacement since anchor, mapped onto vrchats ground plane
        # (dart x,y -> vr x,z) and rotated by the anchor heading offset
        dx, dz = dart_x - ax, dart_y - ay
        h = vyaw0 - ayaw
        ch, sh = math.cos(h), math.sin(h)
        tx = vx0 + dx * ch + dz * sh
        tz = vz0 + dz * ch - dx * sh
        tyaw = vyaw0 + _wrap_pi(dart_yaw - ayaw)

        ex, ez = tx - pose.x, tz - pose.z
        dist = math.hypot(ex, ez)
        if dist > self.REANCHOR_DIST:
            logger.info(f"motion pose tracking: {dist:.1f}m off target, re-anchoring")
            self._anchor = (dart_x, dart_y, dart_yaw, pose.x, pose.z, vr_yaw)
            return 0.0, 0.0, 0.0

        # world error into body frame of the actual avatar heading
        fwd_err = ex * math.sin(vr_yaw) + ez * math.cos(vr_yaw)
        side_err = ex * math.cos(vr_yaw) - ez * math.sin(vr_yaw)
        yaw_err = _wrap_pi(tyaw - vr_yaw)

        clamp = lambda v, m: max(-m, min(m, v))
        return (clamp(self.KP_POS * fwd_err, self.CORR_MAX),
                clamp(self.KP_POS * side_err, self.CORR_MAX),
                clamp(self.KP_YAW * yaw_err, self.YAW_CORR_MAX))


class MotionClient:
    def __init__(self, osc_client, host, port, walk_full=2.0, turn_full=1.8,
                 pose_tracking=False, pose_monitor=1):
        self._osc = osc_client
        self._uri = f"ws://{host}:{port}"
        self._ws = None
        self._recv_task = None
        self._send_task = None
        self._timer_task = None
        self._connected = asyncio.Event()
        self._active = False  # puppet enabled and streaming to osc
        self._target = {p: 0.0 for p in PARAMS}
        self._filters = {p: OneEuro() for p in PARAMS}
        self._vfwd = self._vside = self._vyaw = 0.0
        self._dart_x = self._dart_y = self._dart_yaw = 0.0
        self._got_frame = False
        self._walk_full = walk_full
        self._turn_full = turn_full
        self._tracker = PoseTracker(pose_monitor) if pose_tracking else None
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
        if self._tracker is not None:
            self._tracker.stop()

    # -- commands --

    async def play(self, prompt, seconds=None, once=False):
        await self.ensure_connected()
        self._cancel_timer()
        await self._send({"type": "prompt", "text": prompt, "once": bool(once)})
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
        # server side 'stop' switches to the models own idle prompt
        await self._send({"type": "stop"})
        self.current_prompt = "stand"

    async def reset(self):
        """wipe the motion models context and release the body back to vrchat."""
        self._cancel_timer()
        if self._ws is not None:
            await self._send({"type": "reset"})
        self.current_prompt = None
        self._got_frame = False
        if self._tracker is not None:
            self._tracker.reset_anchor()
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
        if self._tracker is not None:
            self._tracker.reset_anchor()
            if on:
                self._tracker.start()
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
                        self._dart_x = float(params.get("_x", 0.0))
                        self._dart_y = float(params.get("_y", 0.0))
                        self._dart_yaw = float(params.get("_yaw", 0.0))
                        if not self._got_frame:
                            # snap to the first frame, no glide from a stale pose
                            for f in self._filters.values():
                                f.reset()
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
                beta = 1.0 - math.exp(-dt / LOCO_TAU)

                if self._active and self._got_frame:
                    for p in PARAMS:
                        cur = self._filters[p](self._target[p], dt)
                        self._osc.send_message(PREFIX + p, float(cur))
                    cf = cs = cy = 0.0
                    if self._tracker is not None:
                        cf, cs, cy = self._tracker.correction(
                            self._dart_x, self._dart_y, self._dart_yaw, now)
                    sv += ((self._vfwd + cf) / self._walk_full - sv) * beta
                    sh += ((self._vside + cs) / self._walk_full - sh) * beta
                    sl += ((self._vyaw + cy) / self._turn_full - sl) * beta
                    self._osc.send_message("/input/Vertical", float(_snap_axis(sv)))
                    self._osc.send_message("/input/Horizontal", float(_snap_axis(sh)))
                    self._osc.send_message("/input/LookHorizontal", float(max(-1.0, min(1.0, sl))))
                else:
                    sv = sh = sl = 0.0

                await asyncio.sleep(max(0.0, interval - (time.monotonic() - now)))
        except asyncio.CancelledError:
            pass
