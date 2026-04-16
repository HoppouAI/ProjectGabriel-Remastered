"""Discord voice control tools for the main VRChat AI session.

Connects to the GabrielVoiceControl Vencord plugin's WebSocket server
to join/leave voice channels, call users, and manage calls from the main AI.
"""
from google.genai import types
from src.tools._base import BaseTool, register_tool


@register_tool
class DiscordVoiceControlTool(BaseTool):
    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="discord_joinVoice",
                description="Join a Discord voice channel via the Vencord plugin. Requires GabrielVoiceControl plugin running in Discord.\n**Invocation Condition:** Call when asked to join a Discord voice channel or VC.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The voice channel ID to join"},
                    },
                    "required": ["channel_id"],
                },
            ),
            types.FunctionDeclaration(
                name="discord_callUser",
                description="Start a Discord DM voice call via the Vencord plugin.\n**Invocation Condition:** Call when asked to call someone on Discord.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The DM channel ID to call"},
                    },
                    "required": ["channel_id"],
                },
            ),
            types.FunctionDeclaration(
                name="discord_leaveVoice",
                description="Leave the current Discord voice channel or hang up.\n**Invocation Condition:** Call when asked to leave Discord voice or hang up a call.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="discord_getVoiceState",
                description="Get current Discord voice state (connected channel, users, mute/deaf).\n**Invocation Condition:** Call when asked about Discord voice status.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if not name.startswith("discord_"):
            return None

        from discord_bot.tools.voice_control import _send_command

        op_map = {
            "discord_joinVoice": "join_voice",
            "discord_callUser": "call_user",
            "discord_leaveVoice": "leave_voice",
            "discord_getVoiceState": "get_voice_state",
        }

        op = op_map.get(name)
        if op is None:
            return None

        kwargs = {}
        if "channel_id" in args:
            kwargs["channel_id"] = args["channel_id"]

        res = await _send_command(op, **kwargs)
        if res.get("success"):
            return {"result": "ok", **res.get("data", {})}
        return {"result": "error", "message": res.get("error", "Unknown error")}
