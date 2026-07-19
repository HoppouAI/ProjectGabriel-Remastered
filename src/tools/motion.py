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
                    "Style words like slowly, excitedly, sneakily shape how it looks. The motion keeps "
                    "going until you call playMotion again with a new description (switch directly, no "
                    "stop needed in between) or stopMotion. Give seconds for one-off gestures that should "
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
                        "seconds": {"type": "NUMBER", "description": "Optional duration; auto-returns to standing after this many seconds"},
                    },
                    "required": ["prompt"],
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
            from src.motion_client import MotionClient
            self._client = MotionClient(
                self.osc.client,
                self.config.motion_server_host,
                self.config.motion_server_port,
                walk_full=self.config.motion_walk_full_speed,
                turn_full=self.config.motion_turn_full_rate,
            )
        return self._client

    async def handle(self, name, args):
        if name == "playMotion":
            prompt = str(args.get("prompt", "")).strip()
            if not prompt:
                return {"result": "error", "message": "prompt is required"}
            seconds = args.get("seconds")
            try:
                await self._get_client().play(prompt, seconds=seconds)
            except ConnectionError as e:
                return {"result": "error", "message": str(e)}
            msg = f"performing motion: {prompt}"
            if seconds:
                msg += f" for {seconds:.0f}s, then back to standing"
            return {"result": "ok", "message": msg}
        elif name == "stopMotion":
            if self._client is None:
                return {"result": "ok", "message": "no motion playing"}
            await self._client.stop_motion()
            return {"result": "ok", "message": "motion stopped, standing idle"}
        elif name == "resetPose":
            if self._client is None:
                return {"result": "ok", "message": "motion system not active"}
            await self._client.reset()
            return {"result": "ok", "message": "motion context cleared, body control released"}
        return None
