import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class PersonalityTools(BaseTool):

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="listPersonalities",
                description="List all available personality modes that can be switched to.\n**Invocation Condition:** Call when asked what personalities are available.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="switchPersonality",
                description="Switch to a different personality mode. This changes behavior and response style. Once switched, stay fully committed to that personality until told to switch back. Do not randomly switch without reason.\n**Invocation Condition:** Call only when explicitly asked to switch, or when context unmistakably calls for it. If a personality is disabled, say 'Uh oh seems like I cant remember that one.'",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "personalityId": {"type": "STRING", "description": "The ID of the personality to switch to (e.g., 'chill', 'scammer', 'tsundere')"},
                    },
                    "required": ["personalityId"],
                },
            ),
            types.FunctionDeclaration(
                name="getCurrentPersonality",
                description="Get information about the currently active personality mode.\n**Invocation Condition:** Call when asked what mode you are in.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "listPersonalities":
            result = self.personality.list_personalities()
            result["result"] = "ok"
            return result
        elif name == "switchPersonality":
            result = self.personality.switch(args["personalityId"])
            avatar_id = result.get("avatar_id")
            if avatar_id and avatar_id != self.handler._current_avatar_id:
                try:
                    api = self.handler._get_vrchat_api()
                    av_result = await api.select_avatar(avatar_id)
                    if av_result.get("result") == "ok" or av_result.get("avatar_id"):
                        self.handler._current_avatar_id = avatar_id
                        logger.info(f"Auto-switched avatar to {avatar_id} for personality '{args['personalityId']}'")
                    else:
                        logger.warning(f"Failed to auto-switch avatar: {av_result.get('error', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Avatar auto-switch failed: {e}")
            response = {k: v for k, v in result.items() if k != "avatar_id"}
            return response
        elif name == "getCurrentPersonality":
            result = self.personality.get_current()
            result["result"] = "ok"
            return result
        return None
