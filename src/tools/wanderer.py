import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class WandererTools(BaseTool):

    def declarations(self, config=None):
        wanderer_enabled = True
        if config is not None:
            wanderer_enabled = getattr(config, "wanderer_enabled", False)
        if not wanderer_enabled:
            return []
        return [
            types.FunctionDeclaration(
                name="startWander",
                description="Start autonomously wandering around the VRChat map. Uses depth perception to avoid walls and obstacles. Will walk, turn, look around, and jump on its own.\n**Invocation Condition:** Call when asked to wander, explore, walk around, or roam freely.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="stopWander",
                description="Stop autonomous wandering and halt all movement.\n**Invocation Condition:** Call when asked to stop wandering or exploring.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "startWander":
            if self.wanderer is None:
                return {"result": "error", "message": "Wanderer is disabled in config"}
            if self.tracker and self.tracker.active:
                self.tracker.stopfollow()
            return self.wanderer.start()
        elif name == "stopWander":
            if self.wanderer is None:
                return {"result": "error", "message": "Wanderer is disabled in config"}
            return self.wanderer.stop()
        return None
