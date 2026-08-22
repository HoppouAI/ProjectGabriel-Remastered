"""Generated motion as Gabriel's expression layer.

Does what the canned emotion animations do, except the body is driven by the
motion server: gesture motions while he talks, a fidget once he's been quiet
a while, and the puppet handed back to VRChat when nothing is going on.

Only engages on the ARDY backend. DART's prompt styles are too terse to get
usable gestures out of, and its idle drifts.

Hooks are sync and only set flags. A background task at 2hz reconciles the
wanted state with what the server is actually playing, which keeps prompt
switches rate limited and keeps every caller off the event loop.
"""

import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)

TICK_S = 0.5
CONNECT_RETRY_S = 20.0

# used when config.yml has no motion.expression block, so turning motion on is
# enough to get the behaviour without editing prompts by hand
DEFAULT_TALKING = [
    "a person talks while gesturing with both hands",
    "a person speaks and shrugs lightly",
    "a person talks casually, one hand moving as they explain",
    "a person nods while talking, hands relaxed",
]
DEFAULT_THINKING = "a person stands still and looks up thoughtfully, hand near their chin"
DEFAULT_IDLE = [
    "a person shifts their weight and glances around",
    "a person stretches their arms and rolls their shoulders",
    "a person stands and looks around slowly, bored",
]


class MotionExpression:
    def __init__(self, config, osc):
        self.config = config
        self.osc = osc
        cfg = getattr(config, "motion_expression", {}) or {}

        self.enabled = bool(config.motion_enabled and cfg.get("enabled", True))
        self._talking = [str(p) for p in (cfg.get("talking") or DEFAULT_TALKING) if str(p).strip()]
        self._idle = [str(p) for p in (cfg.get("idle") or DEFAULT_IDLE) if str(p).strip()]
        self._thinking_prompt = str(cfg.get("thinking") or DEFAULT_THINKING).strip()
        self._switch_interval = float(cfg.get("switch_interval", 6.0))
        self._idle_after = float(cfg.get("idle_after", 25.0))
        self._release_after = float(cfg.get("release_after", 180.0))

        self._speaking = False
        self._thinking = False
        self._suppressed = False   # seated, crouching, wandering: leave the body alone
        self._last_activity = time.time()

        self._state = None         # 'talk' | 'think' | 'idle' | 'stand' | None (released)
        self._state_since = 0.0
        self._talk_index = random.randrange(len(self._talking)) if self._talking else 0
        self._client = None
        self._task = None
        self._next_connect = 0.0
        self._unsupported_logged = False

    @property
    def active(self):
        """True while this layer is actually driving the body, so the emotion
        system knows to stay out of the way."""
        return self._state is not None

    # -- hooks, all sync and cheap --

    def start(self):
        if not self.enabled:
            return
        self._ensure_task()
        if self._task is not None:
            logger.info("Motion expression started (waiting for the motion server)")

    def _ensure_task(self):
        if not self.enabled or (self._task is not None and not self._task.done()):
            return
        try:
            self._task = asyncio.get_running_loop().create_task(self._run())
        except RuntimeError:
            self._task = None  # no loop yet, a later hook will start it

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def set_speaking(self, speaking):
        self._speaking = bool(speaking)
        self._last_activity = time.time()
        self._ensure_task()

    def set_thinking(self, thinking):
        self._thinking = bool(thinking)
        self._last_activity = time.time()
        self._ensure_task()

    def mark_activity(self):
        self._last_activity = time.time()

    def set_suppressed(self, suppressed):
        self._suppressed = bool(suppressed)

    # -- reconciler --

    def _want(self):
        """What the body should be doing right now."""
        if self._suppressed:
            return None
        if self._speaking and self._talking:
            return "talk"
        if self._thinking and self._thinking_prompt:
            return "think"
        quiet = time.time() - self._last_activity
        if quiet >= self._release_after:
            return None
        if quiet >= self._idle_after and self._idle:
            return "idle"
        return "stand"

    async def _apply(self, want):
        now = time.monotonic()
        if want is None:
            if self._state is not None or self._client.active:
                await self._client.reset()
                self._state = None
            return

        if want == "talk":
            # rotate gestures so a long answer isn't one looping wave
            if self._state == "talk" and now - self._state_since < self._switch_interval:
                return
            self._talk_index = (self._talk_index + 1) % len(self._talking)
            await self._client.play(self._talking[self._talk_index], owner="expression")
        elif want == "think":
            if self._state == "think":
                return
            await self._client.play(self._thinking_prompt, owner="expression")
        elif want == "idle":
            if self._state == "idle" and now - self._state_since < self._switch_interval * 3:
                return
            await self._client.play(random.choice(self._idle), owner="expression")
        elif want == "stand":
            if self._state == "stand":
                return
            await self._client.stop_motion()
            self._client.set_locomotion(False)

        self._state = want
        self._state_since = now

    async def _run(self):
        if self._client is None:
            from src.motion_client import get_motion_client
            try:
                self._client = get_motion_client(self.config, self.osc)
            except Exception as e:
                self.enabled = False
                logger.warning(f"motion expression disabled, no client: {e}")
                return
        while True:
            try:
                await asyncio.sleep(TICK_S)
                want = self._want()
                if self._client.backend is None:
                    # nothing to drive until the server says hello, and there
                    # is no point dialing it up before he has something to do
                    if want in (None, "stand") or time.monotonic() < self._next_connect:
                        continue
                    self._next_connect = time.monotonic() + CONNECT_RETRY_S
                    await self._client.ensure_connected()
                    continue
                if not self._client.is_ardy:
                    if not self._unsupported_logged:
                        self._unsupported_logged = True
                        logger.info(
                            f"Motion expression off: needs the ardy backend, "
                            f"server is running {self._client.model}"
                        )
                    self._state = None
                    continue
                if self._client.owned_by_model:
                    # he asked for a motion himself, that wins until he stops it
                    self._state = None
                    continue
                await self._apply(want)
            except asyncio.CancelledError:
                break
            except ConnectionError:
                self._state = None
                self._next_connect = time.monotonic() + CONNECT_RETRY_S
            except Exception as e:
                logger.warning(f"motion expression tick failed: {e}")
                self._state = None


def build_instruction(config):
    """Prompt block telling him the generated body is how he expresses himself.
    Returns '' when motion is off, so the prompt stays clean."""
    if not config.motion_enabled:
        return ""
    expression = (getattr(config, "motion_expression", {}) or {}).get("enabled", True)
    lines = [
        "# Your body",
        "",
        "You have a real generated body in VRChat. playMotion describes an action in "
        "plain words and your whole body performs it live. This is your main way of "
        "expressing yourself physically, use it often and naturally, not just when "
        "someone asks. Laughing, sulking, getting excited, pointing something out, "
        "being smug, flopping down when you are bored: act it out with playMotion "
        "instead of only saying it.",
    ]
    if expression:
        lines.append(
            "Idle fidgeting and talking gestures happen on their own, so you never "
            "need to call playMotion just to look alive. Call it when you want a "
            "specific action, and it takes over immediately."
        )
    if config.emotion_enabled:
        lines.append(
            "You also still have the older `emotion` animation tool. Prefer playMotion "
            "for anything your body does, it is generated fresh and looks far better. "
            "Only reach for `emotion` when you want one of its specific canned "
            "animations that playMotion cannot produce."
        )
    return "\n".join(lines)
