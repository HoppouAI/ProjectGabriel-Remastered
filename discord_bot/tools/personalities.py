import logging
from google.genai import types

logger = logging.getLogger(__name__)


class PersonalityTool:
    """Personality switching tools for the Discord bot."""

    def __init__(self, handler):
        self.handler = handler

    def declarations(self):
        return [
            types.FunctionDeclaration(
                name="list_personalities",
                description="List all available personalities with their descriptions.\n**Invocation Condition:** Call when asked about available personalities or modes.",
                parameters={"type": "OBJECT", "properties": {}, "required": []},
            ),
            types.FunctionDeclaration(
                name="switch_personality",
                description="Switch to a different personality mode. This changes your behavior and speaking style.\n**Invocation Condition:** Call when asked to change personality, mode, or character.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "personality_id": {"type": "STRING", "description": "The ID of the personality to switch to"},
                    },
                    "required": ["personality_id"],
                },
            ),
            types.FunctionDeclaration(
                name="get_current_personality",
                description="Get the currently active personality.\n**Invocation Condition:** Call when asked what personality or mode is active.",
                parameters={"type": "OBJECT", "properties": {}, "required": []},
            ),
        ]

    async def handle(self, name, args):
        if name == "list_personalities":
            mgr = self.handler._personality_mgr
            if not mgr:
                return {"result": "error", "message": "Personality system not available"}
            return mgr.list_personalities()

        elif name == "switch_personality":
            mgr = self.handler._personality_mgr
            if not mgr:
                return {"result": "error", "message": "Personality system not available"}
            pid = args.get("personality_id", "")
            result = mgr.switch(pid)
            if "error" not in result and "personality_prompt" in result:
                # Inject the personality prompt into the session
                self.handler._personality_prompt = result["personality_prompt"]
            return result

        elif name == "get_current_personality":
            mgr = self.handler._personality_mgr
            if not mgr:
                return {"result": "error", "message": "Personality system not available"}
            return mgr.get_current()

        return None
