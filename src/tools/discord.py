import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class DiscordTools(BaseTool):
    """Tools for the main VRChat AI to interact with Discord via the selfbot."""

    def declarations(self, config=None):
        if not config or not config.get("discord_bot", "enabled", default=False):
            return []
        return [
            types.FunctionDeclaration(
                name="sendDiscordMessage",
                description="Send a message to a Discord user. The Discord selfbot will deliver the message and handle any replies.\n**Invocation Condition:** Call when someone asks you to message someone on Discord, or when you want to relay info to a Discord user.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Discord username or user ID to message"},
                        "message": {"type": "STRING", "description": "The message content to send"},
                    },
                    "required": ["username", "message"],
                },
            ),
            types.FunctionDeclaration(
                name="getDiscordStatus",
                description="Check if the Discord bot is connected and get its status.\n**Invocation Condition:** Call when asked about Discord bot status or connectivity.",
                parameters={"type": "OBJECT", "properties": {}, "required": []},
            ),
        ]

    async def handle(self, name, args):
        if name == "sendDiscordMessage":
            return await self._send_message(args)
        elif name == "getDiscordStatus":
            return await self._get_status(args)
        return None

    async def _send_message(self, args):
        username = args.get("username", "")
        message = args.get("message", "")
        if not username or not message:
            return {"result": "error", "message": "username and message required"}

        # Access the Discord bot instance from the tool handler
        discord_bot = getattr(self.handler, "discord_bot", None)
        if not discord_bot:
            return {"result": "error", "message": "Discord bot not running"}

        try:
            result = await discord_bot.send_message_to_user(username, message)
            return result
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return {"result": "error", "message": str(e)}

    async def _get_status(self, args):
        discord_bot = getattr(self.handler, "discord_bot", None)
        if not discord_bot:
            return {"result": "ok", "connected": False, "reason": "Discord bot not configured"}

        client = discord_bot._client
        if not client or not client.is_ready():
            return {"result": "ok", "connected": False, "reason": "Discord client not ready"}

        return {
            "result": "ok",
            "connected": True,
            "username": str(client.user),
            "guilds": len(client.guilds),
        }
