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
                description="Join a Discord voice channel. Requires the GabrielVoiceControl Vencord plugin.\n**Invocation Condition:** Call when asked to join a Discord voice channel or VC.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "Voice channel ID to join"},
                    },
                    "required": ["channel_id"],
                },
            ),
            types.FunctionDeclaration(
                name="discord_callUser",
                description="Call someone on Discord. Provide ONE of: channel_id (DM channel), user_id (their Discord ID), or name (username/display name). Creates a DM if needed, joins voice, and rings them.\n**Invocation Condition:** Call when asked to call someone on Discord, start a voice call, or ring a user.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "DM/group DM channel ID"},
                        "user_id": {"type": "STRING", "description": "Discord user ID"},
                        "name": {"type": "STRING", "description": "Username or display name to search and call"},
                    },
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="discord_findUser",
                description="Search for a Discord user by name. Returns matching users with IDs, DM channels, and friend status.\n**Invocation Condition:** Call when you need to look up a Discord user by name without calling them.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Username or display name to search for"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="discord_leaveVoice",
                description="Leave Discord voice or hang up a call.\n**Invocation Condition:** Call when asked to leave voice, hang up, disconnect, or end a Discord call.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="discord_getVoiceState",
                description="Get current Discord voice state (channel, users, mute/deaf status).\n**Invocation Condition:** Call when asked about Discord voice status or who's in a call.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if not name.startswith("discord_"):
            return None

        from discord_bot.tools.voice_control import _send_command

        if name == "discord_callUser":
            return await self._handle_call(args, _send_command)
        if name == "discord_findUser":
            res = await _send_command("find_user", query=args.get("query", ""))
            if res.get("success"):
                return {"result": "ok", **res.get("data", {})}
            return {"result": "error", "message": res.get("error", "Unknown error")}

        op_map = {
            "discord_joinVoice": "join_voice",
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

    async def _handle_call(self, args, _send_command):
        """Unified call handler: channel_id > user_id > name."""
        channel_id = args.get("channel_id")
        user_id = args.get("user_id")
        name = (args.get("name") or "").strip()

        if channel_id:
            res = await _send_command("call_user", channel_id=channel_id)
        elif user_id:
            res = await _send_command("call_user_by_id", user_id=user_id)
        elif name:
            # Resolve name to user_id first
            find_res = await _send_command("find_user", query=name)
            if not find_res.get("success"):
                return {"result": "error", "message": find_res.get("error", "Find user failed")}
            users = find_res.get("data", {}).get("users", [])
            if not users:
                return {"result": "error", "message": f"No Discord user found matching '{name}'"}
            target = users[0]
            res = await _send_command("call_user_by_id", user_id=target["id"])
            if res.get("success"):
                return {"result": "ok", "called_user": target["username"], "user_id": target["id"], **res.get("data", {})}
            return {"result": "error", "message": res.get("error", "Call failed")}
        else:
            return {"result": "error", "message": "Provide channel_id, user_id, or name"}

        if res.get("success"):
            return {"result": "ok", **res.get("data", {})}
        return {"result": "error", "message": res.get("error", "Unknown error")}
