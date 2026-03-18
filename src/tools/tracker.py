import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class TrackerTools(BaseTool):

    def declarations(self, config=None):
        tracker_enabled = True
        if config is not None:
            tracker_enabled = getattr(config, "tracker_enabled", True)
        if not tracker_enabled:
            return []
        return [
            types.FunctionDeclaration(
                name="startFollow",
                description="Start following a player visible on screen using YOLO vision tracking. Automatically detects and locks onto the nearest person.\n**Invocation Condition:** Call when asked to follow someone.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "mode": {"type": "STRING", "description": "Follow mode (default 'auto')"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="stopFollow",
                description="Stop following a player and halt all tracking movement.\n**Invocation Condition:** Call immediately when told to stop following.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="setFollowDistance",
                description="Set how close to follow the target player. Value is a fraction from 0.01 (very close) to 0.5 (far away). Default is 0.08.\n**Invocation Condition:** Call when asked to get closer or farther while following.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "value": {"type": "NUMBER", "description": "Target follow distance as bounding-box area fraction (0.01 to 0.5)"},
                    },
                    "required": ["value"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "startFollow":
            if self.tracker is None:
                return {"result": "error", "message": "Tracker is disabled in config"}
            return self.tracker.startfollow(args.get("mode", "auto"))
        elif name == "stopFollow":
            if self.tracker is None:
                return {"result": "error", "message": "Tracker is disabled in config"}
            return self.tracker.stopfollow()
        elif name == "setFollowDistance":
            if self.tracker is None:
                return {"result": "error", "message": "Tracker is disabled in config"}
            return self.tracker.setfollowdistance(args["value"])
        return None
