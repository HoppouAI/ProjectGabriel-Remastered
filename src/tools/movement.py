import asyncio
import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool
from src.emotions import get_emotion_system

logger = logging.getLogger(__name__)


@register_tool
class MovementTools(BaseTool):
    tool_key = "movement"

    def declarations(self, config=None):
        motion_on = getattr(config, "motion_enabled", False) if config is not None else False
        move_note = (
            " Prefer playMotion for walking/running/moving around, it animates your whole body "
            "and looks far better. Only use this for precise positioning nudges or if the motion "
            "system is down." if motion_on else ""
        )
        jump_note = (
            " Prefer playMotion ('a person jumps up and down') for expressive jumping; this is a "
            "plain engine hop for clearing obstacles." if motion_on else ""
        )
        return [
            types.FunctionDeclaration(
                name="vrchatCrouch",
                description="Toggle crouch in VRChat. Press once to crouch, press again to stand up.\n**Invocation Condition:** Call when asked to crouch or stand.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatCrawl",
                description="Toggle crawl/prone position in VRChat. Press once to go prone, press again to stand up.\n**Invocation Condition:** Call when asked to crawl or go prone.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatMove",
                description="Walk/run in VRChat until duration expires or vrchatStop is called. Supports strafe and sprint.\n**Invocation Condition:** Asked to walk, run, or move somewhere." + move_note,
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "'forward', 'backward', 'left' (strafe), 'right' (strafe)."},
                        "duration": {"type": "NUMBER", "description": "Seconds, 0.1-600."},
                        "speed": {"type": "STRING", "description": "'slow', 'normal', 'fast', 'sprint'."},
                    },
                    "required": ["direction", "duration"],
                },
            ),
            types.FunctionDeclaration(
                name="vrchatStop",
                description="Stop all movement in VRChat immediately.\n**Invocation Condition:** Call immediately when asked to stop moving.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatJump",
                description="Make the avatar jump in VRChat.\n**Invocation Condition:** Call when asked to jump." + jump_note,
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatGrab",
                description="Grab/pickup the item directly in front of you (center of your view) in VRChat. You must be looking straight at the item.\n**Invocation Condition:** Call when someone says 'grab this', 'pick that up', or similar. The item must be centered in your vision.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatDrop",
                description="Drop the item you are currently holding in VRChat.\n**Invocation Condition:** Call when someone says 'drop it', 'let go', 'put it down', or similar.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatUse",
                description="Use/interact with the item directly in front of you (center of your view) in VRChat. This activates interactable objects like buttons, doors, or pickups.\n**Invocation Condition:** Call when someone says 'use that', 'press that', 'interact with that', or similar. The item must be centered in your vision.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatLook",
                description="Smoothly turn left/right in VRChat. Same EMA turning as follow system.\n**Invocation Condition:** Asked to look or turn, or to aim view before grabbing/using something.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "'left' or 'right'."},
                        "duration": {"type": "NUMBER", "description": "Seconds, 0.1-10."},
                        "speed": {"type": "STRING", "description": "'slow', 'normal', 'fast'."},
                    },
                    "required": ["direction", "duration"],
                },
            ),
            types.FunctionDeclaration(
                name="vrchatLookVertical",
                description="Smoothly tilt view up/down in VRChat.\n**Invocation Condition:** Asked to look up/down or check something above/below.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "'up' or 'down'."},
                        "duration": {"type": "NUMBER", "description": "Seconds, 0.1-10."},
                        "speed": {"type": "STRING", "description": "'slow', 'normal', 'fast'."},
                    },
                    "required": ["direction", "duration"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "vrchatCrouch":
            self.osc.toggle_crouch()
            emo = get_emotion_system()
            if emo:
                emo.set_crouching(not emo._crouching)
            return {"result": "ok"}
        elif name == "vrchatCrawl":
            self.osc.toggle_crawl()
            emo = get_emotion_system()
            if emo:
                emo.set_crouching(not emo._crouching)
            return {"result": "ok"}
        elif name == "vrchatMove":
            return await self._vrchat_move(args["direction"], args["duration"], args.get("speed", "normal"))
        elif name == "vrchatStop":
            self.osc.stop_all_movement()
            return {"result": "ok"}
        elif name == "vrchatJump":
            self.osc.jump()
            return {"result": "ok"}
        elif name == "vrchatGrab":
            await asyncio.to_thread(self.osc.grab)
            return {"result": "ok"}
        elif name == "vrchatDrop":
            await asyncio.to_thread(self.osc.drop)
            return {"result": "ok"}
        elif name == "vrchatUse":
            await asyncio.to_thread(self.osc.use)
            return {"result": "ok"}
        elif name == "vrchatLook":
            direction = args.get("direction", "right")
            duration = min(max(float(args.get("duration", 0.5)), 0.1), 10.0)
            speed = args.get("speed", "normal")
            asyncio.get_event_loop().run_in_executor(None, self.osc.look, direction, duration, speed)
            return {"result": "ok"}
        elif name == "vrchatLookVertical":
            direction = args.get("direction", "up")
            duration = min(max(float(args.get("duration", 0.5)), 0.1), 10.0)
            speed = args.get("speed", "normal")
            asyncio.get_event_loop().run_in_executor(None, self.osc.look_vertical, direction, duration, speed)
            return {"result": "ok"}
        return None

    async def _vrchat_move(self, direction: str, duration: float, speed: str = "normal"):
        direction = direction.lower()
        if direction not in ("forward", "backward", "left", "right"):
            return {"result": "error", "message": f"Invalid direction: {direction}. Use forward, backward, left, or right."}
        duration = max(0.1, min(600.0, duration))
        self.osc.start_move(direction, speed)

        async def _stop_after():
            await asyncio.sleep(duration)
            self.osc.stop_all_movement()
        asyncio.create_task(_stop_after())

        return {"result": "ok", "direction": direction, "duration": duration, "speed": speed}
