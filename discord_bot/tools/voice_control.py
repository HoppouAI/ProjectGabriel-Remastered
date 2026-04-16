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


async def _get_ws(port: int = 6463):
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


async def _send_command(op: str, port: int = 6463, **kwargs) -> dict:
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
        self._port = 6463

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
                description="Start a voice call in a DM or group DM. Rings the user like a real Discord call. Requires the GabrielVoiceControl Vencord plugin.\n**Invocation Condition:** Call when someone asks you to call them, start a voice call, or hop on a call. Leave target empty to call in the current DM/group DM, or specify a channel ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The DM/group DM channel ID to call. If omitted, uses the current channel."},
                    },
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="leaveVoiceChannel",
                description="Leave the current voice channel or hang up a call.\n**Invocation Condition:** Call when asked to leave voice, hang up, disconnect from VC, or end the call.",
                parameters={"type": "OBJECT", "properties": {}},
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

        elif name == "leaveVoiceChannel":
            res = await _send_command("leave_voice", self._port)
            return {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("error")}

        elif name == "getVoiceState":
            res = await _send_command("get_voice_state", self._port)
            return {"result": "ok", **res.get("data", {})} if res.get("success") else {"result": "error", "message": res.get("error")}

        return None
