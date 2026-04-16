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
                description="Join a Discord voice channel. Requires the GabrielVoiceControl Vencord plugin.\n**Invocation Condition:** Call when asked to join a voice channel, VC, or voice chat.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "Voice channel ID to join"},
                    },
                    "required": ["channel_id"],
                },
            ),
            types.FunctionDeclaration(
                name="callUser",
                description="Call someone on Discord. Provide ONE of: channel_id (DM channel), user_id (their Discord ID), or name (username/display name). Creates a DM if needed, joins voice, and rings them.\n**Invocation Condition:** Call when asked to call someone, start a voice call, or ring a user.",
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
                name="leaveVoiceChannel",
                description="Leave voice or hang up a call.\n**Invocation Condition:** Call when asked to leave voice, hang up, disconnect, or end the call.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="findUser",
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
                name="getVoiceState",
                description="Get current voice state (channel, users, mute/deaf status).\n**Invocation Condition:** Call when you need to check voice status or who's in a call.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "joinVoiceChannel":
            channel_id = args.get("channel_id")
            if not channel_id:
                return {"result": "error", "message": "channel_id required"}
            res = await _send_command("join_voice", self._port, channel_id=channel_id)
            if res.get("success"):
                await self._relay_voice_state("joined voice channel")
                return {"result": "ok", **res.get("data", {})}
            return {"result": "error", "message": res.get("error")}

        elif name == "callUser":
            return await self._handle_call(args)

        elif name == "leaveVoiceChannel":
            res = await _send_command("leave_voice", self._port)
            if res.get("success"):
                await self._relay_callback_safe("[Discord Call] You have left the Discord voice call / hung up.")
                return {"result": "ok"}
            return {"result": "error", "message": res.get("error")}

        elif name == "findUser":
            query = args.get("query", "").strip()
            if not query:
                return {"result": "error", "message": "query required"}
            res = await _send_command("find_user", self._port, query=query)
            return {"result": "ok", **res.get("data", {})} if res.get("success") else {"result": "error", "message": res.get("error")}

        elif name == "getVoiceState":
            res = await _send_command("get_voice_state", self._port)
            return {"result": "ok", **res.get("data", {})} if res.get("success") else {"result": "error", "message": res.get("error")}

        return None

    async def _handle_call(self, args):
        """Unified call handler: channel_id > user_id > name > current channel."""
        channel_id = args.get("channel_id")
        user_id = args.get("user_id")
        name = (args.get("name") or "").strip()
        called_user = None

        if channel_id:
            res = await _send_command("call_user", self._port, channel_id=channel_id)
        elif user_id:
            res = await _send_command("call_user_by_id", self._port, user_id=user_id)
        elif name:
            find_res = await _send_command("find_user", self._port, query=name)
            if not find_res.get("success"):
                return {"result": "error", "message": find_res.get("error", "Find user failed")}
            users = find_res.get("data", {}).get("users", [])
            if not users:
                return {"result": "error", "message": f"No Discord user found matching '{name}'"}
            target = users[0]
            called_user = target.get("username", name)
            res = await _send_command("call_user_by_id", self._port, user_id=target["id"])
            if res.get("success"):
                await self._relay_voice_state(f"started a call with {called_user}", called_user=called_user)
                return {"result": "ok", "called_user": called_user, "user_id": target["id"], **res.get("data", {})}
            return {"result": "error", "message": res.get("error", "Call failed")}
        else:
            ch = getattr(self.handler, "_current_channel", None)
            if ch:
                res = await _send_command("call_user", self._port, channel_id=str(ch.id))
            else:
                return {"result": "error", "message": "Provide channel_id, user_id, or name"}

        if res.get("success"):
            await self._relay_voice_state("started a call", called_user=called_user)
            return {"result": "ok", **res.get("data", {})}
        return {"result": "error", "message": res.get("error", "Unknown error")}

    async def _relay_callback_safe(self, text: str):
        """Relay a message to the main VRChat session if relay is available."""
        cb = getattr(self.handler, "_relay_callback", None)
        if cb:
            try:
                await cb(text)
                logger.info(f"Relayed to VRChat: {text[:80]}")
            except Exception as e:
                logger.debug(f"Relay failed (non-critical): {e}")

    async def _relay_voice_state(self, action: str, called_user: str = None):
        """Fetch current voice state and relay it to the main session."""
        state_res = await _send_command("get_voice_state", self._port)
        if not state_res.get("success"):
            msg = f"[Discord Call] You {action} on Discord."
            if called_user:
                msg = f"[Discord Call] You {action} (calling {called_user}) on Discord."
            await self._relay_callback_safe(msg)
            return

        data = state_res.get("data", {})
        channel_name = data.get("channel_name")
        guild_id = data.get("guild_id")
        users = data.get("users", [])
        # Filter out self from the user list
        other_users = [u.get("name", "Unknown") for u in users if u.get("name") != "Unknown"]

        if guild_id and channel_name:
            location = f"server voice channel '{channel_name}'"
        elif channel_name:
            location = f"'{channel_name}'"
        elif called_user:
            location = f"a DM call with {called_user}"
        elif other_users:
            location = f"a DM call with {', '.join(other_users)}"
        else:
            location = "a Discord call"

        parts = [f"[Discord Call] You {action} in {location}."]
        if other_users:
            parts.append(f"Users in call: {', '.join(other_users)}.")
        elif called_user:
            parts.append(f"Calling {called_user} (ringing, waiting for them to pick up).")
        parts.append("You are now in BOTH VRChat and a Discord call -- keep that in mind.")
        await self._relay_callback_safe(" ".join(parts))
