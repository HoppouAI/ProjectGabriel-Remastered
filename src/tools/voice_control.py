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
                description="Start a Discord DM voice call via the Vencord plugin.\n**Invocation Condition:** Call when asked to call someone on Discord by channel ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The DM channel ID to call"},
                    },
                    "required": ["channel_id"],
                },
            ),
            types.FunctionDeclaration(
                name="discord_callUserById",
                description="Call a Discord user by their user ID. Creates a DM if needed, then rings them.\n**Invocation Condition:** Call when asked to call a specific Discord user and you have their user ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_id": {"type": "STRING", "description": "The Discord user ID to call"},
                    },
                    "required": ["user_id"],
                },
            ),
            types.FunctionDeclaration(
                name="discord_findUser",
                description="Search for a Discord user by username or display name. Returns matching users with their IDs and DM channel IDs.\n**Invocation Condition:** Call when you need to find a Discord user's ID to call them, or look up someone by name.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Username or display name to search for"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="discord_callUserByName",
                description="Find a Discord user by name and call them. Searches friends/contacts, creates DM if needed, then rings.\n**Invocation Condition:** Call when asked to call a Discord user and you only have their name, not their ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING", "description": "Username or display name of the person to call"},
                    },
                    "required": ["name"],
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

        # callUserByName is handled specially (find_user + call_user_by_id chain)
        if name == "discord_callUserByName":
            return await self._call_by_name(args)
        if name == "discord_findUser":
            res = await _send_command("find_user", query=args.get("query", ""))
            if res.get("success"):
                return {"result": "ok", **res.get("data", {})}
            return {"result": "error", "message": res.get("error", "Unknown error")}

        op_map = {
            "discord_joinVoice": "join_voice",
            "discord_callUser": "call_user",
            "discord_callUserById": "call_user_by_id",
            "discord_leaveVoice": "leave_voice",
            "discord_getVoiceState": "get_voice_state",
        }

        op = op_map.get(name)
        if op is None:
            return None

        kwargs = {}
        if "channel_id" in args:
            kwargs["channel_id"] = args["channel_id"]
        if "user_id" in args:
            kwargs["user_id"] = args["user_id"]

        res = await _send_command(op, **kwargs)
        if res.get("success"):
            return {"result": "ok", **res.get("data", {})}
        return {"result": "error", "message": res.get("error", "Unknown error")}

    async def _call_by_name(self, args):
        """Find user by name then call them."""
        from discord_bot.tools.voice_control import _send_command
        query = args.get("name", "").strip()
        if not query:
            return {"result": "error", "message": "name required"}

        find_res = await _send_command("find_user", query=query)
        if not find_res.get("success"):
            return {"result": "error", "message": find_res.get("error", "Find user failed")}

        users = find_res.get("data", {}).get("users", [])
        if not users:
            return {"result": "error", "message": f"No Discord user found matching '{query}'"}

        # Pick the best match (first result - sorted by friend status + exact match)
        target = users[0]
        user_id = target["id"]

        call_res = await _send_command("call_user_by_id", user_id=user_id)
        if call_res.get("success"):
            return {"result": "ok", "called_user": target["username"], "user_id": user_id, **call_res.get("data", {})}
        return {"result": "error", "message": call_res.get("error", "Call failed")}
