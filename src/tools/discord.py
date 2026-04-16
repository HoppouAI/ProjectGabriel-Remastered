import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class DiscordTools(BaseTool):
    tool_key = "discord"
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
                name="relayToDiscord",
                description=(
                    "Send a message to your Discord self (your other instance running on Discord). "
                    "Use this to communicate with your Discord self, ask it to do something, or share context. "
                    "Be specific and actionable so your Discord self can act immediately. "
                    "The Discord user CANNOT hear you, they are on Discord not in VRChat. "
                    "When handling a Discord relay request, talk to VRChat people naturally in third person "
                    "about it (e.g. 'Oh, someone from Discord wants me to play this song'). "
                    "After completing the action, use this tool to confirm back to your Discord self.\n"
                    "**Invocation Condition:** Call when you want to tell your Discord self something, "
                    "ask it to relay info to Discord users, or coordinate between VRChat and Discord."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "content": {"type": "STRING", "description": "The message to send to your Discord self"},
                    },
                    "required": ["content"],
                },
            ),
            types.FunctionDeclaration(
                name="getDiscordStatus",
                description="Check if the Discord bot is connected and get its status.\n**Invocation Condition:** Call when asked about Discord bot status or connectivity.",
                parameters={"type": "OBJECT", "properties": {}, "required": []},
            ),
            types.FunctionDeclaration(
                name="discord_hangUp",
                description="Hang up / leave the current Discord voice call. Disconnects and stops ringing.\n**Invocation Condition:** Call when you want to end a Discord call, hang up, or leave Discord voice.",
                parameters={"type": "OBJECT", "properties": {}, "required": []},
            ),
        ]

    async def handle(self, name, args):
        if name == "sendDiscordMessage":
            return await self._send_message(args)
        elif name == "relayToDiscord":
            return await self._relay_to_discord(args)
        elif name == "getDiscordStatus":
            return await self._get_status(args)
        elif name == "discord_hangUp":
            return await self._hang_up()
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

    async def _relay_to_discord(self, args):
        content = args.get("content", "")
        if not content:
            return {"result": "error", "message": "content required"}

        discord_bot = getattr(self.handler, "discord_bot", None)
        if not discord_bot:
            return {"result": "error", "message": "Discord bot not running"}

        try:
            relay_text = f"[From your VRChat self] {content}"
            await discord_bot.receive_relay(relay_text)
            logger.info(f"Relayed to Discord: {content[:80]}")
            return {"result": "ok", "relayed": True}
        except Exception as e:
            logger.error(f"Discord relay failed: {e}")
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

    async def _hang_up(self):
        try:
            from discord_bot.tools.voice_control import _send_command
            res = await _send_command("leave_voice")
            if res.get("success"):
                return {"result": "ok", "message": "Disconnected from Discord voice call"}
            return {"result": "error", "message": res.get("error", "Failed to hang up")}
        except Exception as e:
            logger.error(f"Discord hang up failed: {e}")
            return {"result": "error", "message": str(e)}
