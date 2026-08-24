import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class MotionTools(BaseTool):
    tool_key = "motion"

    def __init__(self, handler):
        super().__init__(handler)
        self._client = None

    def declarations(self, config=None):
        enabled = True
        if config is not None:
            enabled = getattr(config, "motion_enabled", False)
        if not enabled:
            return []
        return [
            types.FunctionDeclaration(
                name="playMotion",
                description=(
                    "Act out a full-body motion generated live from a text description. Your whole body "
                    "performs it, and walking or turning in the motion physically moves you through the "
                    "world. Describe ONE continuous action as a short third-person sentence, e.g. "
                    "'a person dances energetically', 'a person sits down cross legged', 'a person walks "
                    "forward slowly', 'a person waves with their right hand', 'a person jumps up and down'. "
                    "Style words strongly shape how it looks: slowly, excitedly, sneakily, 'like a zombie', "
                    "'while limping', 'like a gorilla', crawling, marching. Combos work too, e.g. 'a person "
                    "walks in a circle while flapping their arms'. With loop=true the motion keeps "
                    "going until you call playMotion again with a new description (switch directly, no "
                    "stop needed in between) or stopMotion. With loop=false the action plays exactly once "
                    "and you settle back to standing on your own, use it for flips, jumps, bows and other "
                    "tricks so they don't repeat forever. Give seconds for looped gestures that should "
                    "auto-return to standing, like a 3 second wave."
                    "\n**Invocation Condition:** Call when you want to physically perform an action with "
                    "your body, when someone asks you to dance, sit, jump, act something out, or when it "
                    "fits the moment emotionally. Also the PREFERRED way to move around when someone tells "
                    "you to walk, run, come closer, or back up ('a person walks forward', 'a person turns "
                    "around and walks away'), it looks far more natural than the basic vrchatMove tool. "
                    "Call again anytime to switch motions."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prompt": {"type": "STRING", "description": "Third-person action sentence, e.g. 'a person dances energetically'. One action per call."},
                        "loop": {"type": "BOOLEAN", "description": "true = continuous motion that keeps going (walking, dancing, holding a pose like sitting or lying down). false = one-shot trick that ends back on your feet (backflip, jump, bow, spin, kick); it plays exactly once, then you return to standing automatically. It waits for the landing, so a continuous double flip completes fine. For counted repetitions with pauses in between ('do 3 backflips'), use loop=true with seconds instead."},
                        "seconds": {"type": "NUMBER", "description": "Optional duration for looped motions; auto-returns to standing after this many seconds"},
                    },
                    "required": ["prompt", "loop"],
                },
            ),
            types.FunctionDeclaration(
                name="playMotionSequence",
                description=(
                    "Act out several motions back to back as one flowing routine. Each step is its "
                    "own short third-person action sentence, and they blend into each other, e.g. "
                    "['a person does a cartwheel', 'a person does a backflip', 'a person takes a bow']. "
                    "Use this INSTEAD of writing one sentence with 'then' in it, a chained sentence "
                    "makes the motion model freeze up partway through; separate steps are the only "
                    "reliable way to chain. Each step plays once and automatically moves on when it "
                    "lands, or after seconds_each if it never settles."
                    "\n**Invocation Condition:** Call when asked for a routine, combo or several "
                    "actions in a row ('do a flip then a spin then bow', 'dance then sit down'), or "
                    "whenever you want to perform a little sequence rather than one action."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "steps": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Ordered action sentences, ONE action each, e.g. ['a person does a front flip', 'a person does a backflip', 'a person waves']. 2 to 8 steps.",
                        },
                        "seconds_each": {"type": "NUMBER", "description": "Optional max seconds to spend on each step before moving on. Leave empty to advance as soon as each action lands."},
                        "loop_last": {"type": "BOOLEAN", "description": "true = the final step keeps going instead of returning to standing, e.g. finish a routine by settling into 'a person talks casually'. Default false."},
                    },
                    "required": ["steps"],
                },
            ),
            types.FunctionDeclaration(
                name="stopMotion",
                description=(
                    "Stop the current motion and return to a natural generated standing idle. The motion "
                    "system stays active and ready for the next playMotion."
                    "\n**Invocation Condition:** Call when told to stop moving or when a motion should end early."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="resetPose",
                description=(
                    "Full motion reset: clears the motion model's memory back to a clean standing pose and "
                    "releases body control back to normal VRChat idle animations."
                    "\n**Invocation Condition:** Call if a motion looks stuck or glitched, or when you are "
                    "completely done doing motions."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    def _get_client(self):
        if self._client is None:
            from src.motion_client import get_motion_client
            self._client = get_motion_client(self.config, self.osc)
        return self._client

    def _active_client(self):
        # the expression layer and the walk path share this client, so it can be
        # driving the body long before playMotion is ever called here
        if self._client is None:
            from src.motion_client import active_motion_client
            self._client = active_motion_client()
        return self._client

    async def handle(self, name, args):
        if name == "playMotion":
            prompt = str(args.get("prompt", "")).strip()
            if not prompt:
                return {"result": "error", "message": "prompt is required"}
            seconds = args.get("seconds")
            once = not bool(args.get("loop", True))
            try:
                await self._get_client().play(prompt, seconds=seconds, once=once)
            except ConnectionError as e:
                return {"result": "error", "message": str(e)}
            msg = f"performing motion: {prompt}"
            if once:
                msg += " (one-shot, will return to standing when it lands)"
            elif seconds:
                msg += f" for {seconds:.0f}s, then back to standing"
            return {"result": "ok", "message": msg}
        elif name == "playMotionSequence":
            steps = [str(s).strip() for s in (args.get("steps") or []) if str(s).strip()]
            if not steps:
                return {"result": "error", "message": "steps is required"}
            seconds_each = args.get("seconds_each")
            loop_last = bool(args.get("loop_last", False))
            try:
                await self._get_client().play_sequence(
                    steps, seconds_each=seconds_each, loop_last=loop_last)
            except ConnectionError as e:
                return {"result": "error", "message": str(e)}
            msg = f"performing {len(steps)} step routine: {' -> '.join(steps)}"
            msg += " (holding the last one)" if loop_last else " (then back to standing)"
            return {"result": "ok", "message": msg}
        elif name == "stopMotion":
            client = self._active_client()
            if client is None:
                return {"result": "ok", "message": "no motion playing"}
            await client.stop_motion()
            return {"result": "ok", "message": "motion stopped, standing idle"}
        elif name == "resetPose":
            client = self._active_client()
            if client is None:
                return {"result": "ok", "message": "motion system not active"}
            await client.reset()
            return {"result": "ok", "message": "motion context cleared, body control released"}
        return None
