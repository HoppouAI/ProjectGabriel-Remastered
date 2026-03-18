import asyncio
import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool
from src.emotions import get_emotion_system

logger = logging.getLogger(__name__)


@register_tool
class MovementTools(BaseTool):

    def declarations(self, config=None):
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
                description="Start walking in a direction in VRChat. Supports strafing (left/right) and sprinting. The avatar will keep moving until duration expires or vrchatStop is called.\n**Invocation Condition:** Call when asked to walk, run, or move somewhere.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "Direction to move: 'forward', 'backward', 'left' (strafe left), or 'right' (strafe right)"},
                        "duration": {"type": "NUMBER", "description": "How long to move in seconds (0.1 to 600). After this time, movement stops automatically."},
                        "speed": {"type": "STRING", "description": "Movement speed: 'slow' (careful walk), 'normal' (default walk), 'fast' (brisk walk), 'sprint' (full run)"},
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
                description="Make the avatar jump in VRChat.\n**Invocation Condition:** Call when asked to jump.",
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
                description="Smoothly turn/rotate the avatar left or right in VRChat. Uses the same smooth EMA turning as the follow system with gradual ramp up and ramp down.\n**Invocation Condition:** Call when asked to look left or right, turn around, or face a direction. Also useful for aiming your view at objects before grabbing/using them.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "Direction to turn: 'left' or 'right'"},
                        "duration": {"type": "NUMBER", "description": "How long to turn in seconds (0.1 to 10). Small values for slight adjustments, larger for big turns."},
                        "speed": {"type": "STRING", "description": "Turn speed: 'slow' (gentle glance), 'normal' (default), 'fast' (quick snap)"},
                    },
                    "required": ["direction", "duration"],
                },
            ),
            types.FunctionDeclaration(
                name="vrchatLookVertical",
                description="Smoothly tilt the avatar's view up or down in VRChat. Same smooth EMA turning as horizontal look.\n**Invocation Condition:** Call when asked to look up, look down, tilt head up/down, or check something above/below.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "Direction to look: 'up' or 'down'"},
                        "duration": {"type": "NUMBER", "description": "How long to look in seconds (0.1 to 10). Small values for slight tilts, larger for full up/down."},
                        "speed": {"type": "STRING", "description": "Tilt speed: 'slow' (gentle), 'normal' (default), 'fast' (quick snap)"},
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
                emo._crouching = not emo._crouching
                if emo._crouching and emo._is_speaking:
                    emo.stop_speaking()
            return {"result": "ok"}
        elif name == "vrchatCrawl":
            self.osc.toggle_crawl()
            emo = get_emotion_system()
            if emo:
                emo._crouching = not emo._crouching
                if emo._crouching and emo._is_speaking:
                    emo.stop_speaking()
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
