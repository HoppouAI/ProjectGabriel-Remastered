"""Voice control tools for the Discord bot.

Connects to the GabrielVoiceControl Vencord plugin's WebSocket server
to join/leave voice channels, call users, and manage calls.
"""
import asyncio
import json
import logging
import uuid

from google.genai import types

logger = logging.getLogger(__name__)

# We lazy-import websockets to avoid issues if not installed
_ws_connection = None
_ws_lock = asyncio.Lock()


async def _get_ws(port: int = 9473):
    """Get or create a WebSocket connection to the Vencord plugin."""
    global _ws_connection
    async with _ws_lock:
        if _ws_connection is not None:
            try:
                await _ws_connection.ping()
                return _ws_connection
            except Exception:
                _ws_connection = None

        try:
            import websockets
            _ws_connection = await websockets.connect(
                f"ws://127.0.0.1:{port}",
                open_timeout=5,
                close_timeout=5,
            )
            logger.info(f"Connected to Vencord voice control on port {port}")
            return _ws_connection
        except Exception as e:
            logger.warning(f"Could not connect to Vencord plugin: {e}")
            _ws_connection = None
            return None


async def _send_command(op: str, port: int = 9473, **kwargs) -> dict:
    """Send a command to the Vencord plugin and wait for the response."""
    ws = await _get_ws(port)
    if ws is None:
        return {"success": False, "error": "Vencord plugin not connected. Is Discord open with the GabrielVoiceControl plugin enabled?"}

    nonce = str(uuid.uuid4())
    cmd = {"op": op, "nonce": nonce, **kwargs}

    try:
        await ws.send(json.dumps(cmd))
        # Wait for matching response (up to 10s)
        async with asyncio.timeout(10):
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("nonce") == nonce:
                    return msg
                # Events from the plugin (voice_state_update, etc.) - ignore for now
    except asyncio.TimeoutError:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        global _ws_connection
        _ws_connection = None
        return {"success": False, "error": f"Connection error: {e}"}


class VoiceControlTool:
    """Voice channel and call control via Vencord plugin."""

    def __init__(self, handler):
        self.handler = handler
        self._port = 9473

    def declarations(self):
        return [
            types.FunctionDeclaration(
                name="joinVoiceChannel",
                description="Join a voice channel in a Discord server or DM. Requires the GabrielVoiceControl Vencord plugin.\n**Invocation Condition:** Call when someone asks you to join a voice channel, VC, or voice chat in a server. You need the voice channel ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The voice channel ID to join"},
                    },
                    "required": ["channel_id"],
                },
            ),
            types.FunctionDeclaration(
                name="callUser",
                description="Start a voice call in a DM or group DM by channel ID. Joins voice and rings the recipients. Requires the GabrielVoiceControl Vencord plugin.\n**Invocation Condition:** Call when someone asks you to call them or start a voice call. Leave channel_id empty to use the current channel.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The DM/group DM channel ID to call. If omitted, uses the current channel."},
                    },
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="callUserById",
                description="Start a voice call with a specific user by their user ID. Creates a DM if needed, then joins voice and rings them.\n**Invocation Condition:** Call when you want to call a specific Discord user and you have their user ID but not a channel ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_id": {"type": "STRING", "description": "The Discord user ID to call"},
                    },
                    "required": ["user_id"],
                },
            ),
            types.FunctionDeclaration(
                name="leaveVoiceChannel",
                description="Leave the current voice channel or hang up a call. Also stops ringing.\n**Invocation Condition:** Call when asked to leave voice, hang up, disconnect from VC, or end the call.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="findUser",
                description="Search for a Discord user by username or display name. Returns matching users with IDs and DM channel IDs.\n**Invocation Condition:** Call when you need to find a Discord user by name to get their ID or DM channel.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Username or display name to search for"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="callUserByName",
                description="Find a Discord user by name and call them. Searches friends/contacts, creates DM if needed, then rings.\n**Invocation Condition:** Call when asked to call someone by their name and you don't have their ID or channel ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING", "description": "Username or display name of the person to call"},
                    },
                    "required": ["name"],
                },
            ),
            types.FunctionDeclaration(
                name="getVoiceState",
                description="Get your current voice connection status, who is in the channel, and mute/deaf state.\n**Invocation Condition:** Call when you need to check if you are in a voice channel or who is in the call.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "joinVoiceChannel":
            channel_id = args.get("channel_id")
            if not channel_id:
                return {"result": "error", "message": "channel_id required"}
            res = await _send_command("join_voice", self._port, channel_id=channel_id)
            return {"result": "ok", **res.get("data", {})} if res.get("success") else {"result": "error", "message": res.get("error")}

        elif name == "callUser":
            channel_id = args.get("channel_id")
            if not channel_id:
                ch = getattr(self.handler, "_current_channel", None)
                if ch:
                    channel_id = str(ch.id)
                else:
                    return {"result": "error", "message": "No channel_id and no current channel"}
            res = await _send_command("call_user", self._port, channel_id=channel_id)
            return {"result": "ok", **res.get("data", {})} if res.get("success") else {"result": "error", "message": res.get("error")}

        elif name == "callUserById":
            user_id = args.get("user_id")
            if not user_id:
                return {"result": "error", "message": "user_id required"}
            res = await _send_command("call_user_by_id", self._port, user_id=user_id)
            return {"result": "ok", **res.get("data", {})} if res.get("success") else {"result": "error", "message": res.get("error")}

        elif name == "leaveVoiceChannel":
            res = await _send_command("leave_voice", self._port)
            return {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("error")}

        elif name == "findUser":
            query = args.get("query", "").strip()
            if not query:
                return {"result": "error", "message": "query required"}
            res = await _send_command("find_user", self._port, query=query)
            return {"result": "ok", **res.get("data", {})} if res.get("success") else {"result": "error", "message": res.get("error")}

        elif name == "callUserByName":
            query = args.get("name", "").strip()
            if not query:
                return {"result": "error", "message": "name required"}
            # Find user first
            find_res = await _send_command("find_user", self._port, query=query)
            if not find_res.get("success"):
                return {"result": "error", "message": find_res.get("error", "Find user failed")}
            users = find_res.get("data", {}).get("users", [])
            if not users:
                return {"result": "error", "message": f"No Discord user found matching '{query}'"}
            target = users[0]
            user_id = target["id"]
            # Call the best match
            call_res = await _send_command("call_user_by_id", self._port, user_id=user_id)
            if call_res.get("success"):
                return {"result": "ok", "called_user": target["username"], "user_id": user_id, **call_res.get("data", {})}
            return {"result": "error", "message": call_res.get("error", "Call failed")}

        elif name == "getVoiceState":
            res = await _send_command("get_voice_state", self._port)
            return {"result": "ok", **res.get("data", {})} if res.get("success") else {"result": "error", "message": res.get("error")}

        return None
