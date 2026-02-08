import asyncio
import logging
from google.genai import types
from src.memory import get_memory_tools, handle_memory_function_call

logger = logging.getLogger(__name__)


def get_tool_declarations():
    return [
        types.Tool(google_search=types.GoogleSearch()),
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="play_sound_effect",
                description="Search and play a sound effect from MyInstants website",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Sound effect name to search for"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="list_music",
                description="List all available local music files that can be played",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="play_music",
                description="Play a local music file by filename",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "filename": {"type": "STRING", "description": "Music filename to play"},
                        "volume": {"type": "INTEGER", "description": "Volume 0-100, can boost up to 300 for louder playback"},
                    },
                    "required": ["filename"],
                },
            ),
            types.FunctionDeclaration(
                name="stop_music",
                description="Stop currently playing music",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="pause_music",
                description="Pause the currently playing music. Can be resumed later.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="resume_music",
                description="Resume paused music playback.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="set_music_volume",
                description="Adjust the music volume while playing. Only works for volumes 0-100 during playback.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "volume": {"type": "INTEGER", "description": "Volume level 0-100"},
                    },
                    "required": ["volume"],
                },
            ),
            types.FunctionDeclaration(
                name="set_voice_boost",
                description="Set voice boost level for loud distorted bass-boosted yelling effect on your microphone output",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "level": {"type": "INTEGER", "description": "0=normal voice, 1-10=increasingly loud/distorted/bass-boosted"},
                    },
                    "required": ["level"],
                },
            ),
            types.FunctionDeclaration(
                name="toggle_vrchat_mic",
                description="Mute or unmute the VRChat microphone",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "muted": {"type": "BOOLEAN", "description": "True to mute, False to unmute"},
                    },
                    "required": ["muted"],
                },
            ),
            types.FunctionDeclaration(
                name="follow_person",
                description="Enable or disable following a person using YOLO vision tracking in VRChat",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "enabled": {"type": "BOOLEAN", "description": "True to start following, False to stop"},
                    },
                    "required": ["enabled"],
                },
            ),
            types.FunctionDeclaration(
                name="list_personalities",
                description="List all available personality modes that can be switched to",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="switch_personality",
                description="Switch to a different personality mode. This changes behavior and response style.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "personality_id": {"type": "STRING", "description": "The ID of the personality to switch to (e.g., 'chill', 'scammer', 'tsundere')"},
                    },
                    "required": ["personality_id"],
                },
            ),
            types.FunctionDeclaration(
                name="get_current_personality",
                description="Get information about the currently active personality mode",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchat_crouch",
                description="Toggle crouch in VRChat. Press once to crouch, press again to stand up.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchat_crawl",
                description="Toggle crawl/prone position in VRChat. Press once to go prone, press again to stand up.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchat_move",
                description="Start walking in a direction in VRChat. The avatar will keep moving until stop_movement is called.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "Direction to move: 'forward', 'backward', 'left', or 'right'"},
                        "duration": {"type": "NUMBER", "description": "How long to move in seconds (0.1 to 600). After this time, movement stops automatically."},
                    },
                    "required": ["direction", "duration"],
                },
            ),
            types.FunctionDeclaration(
                name="vrchat_stop",
                description="Stop all movement in VRChat immediately.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchat_jump",
                description="Make the avatar jump in VRChat.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            # Memory system (unified action-based)
            types.FunctionDeclaration(
                name="memory",
                description="Persistent memory system. Actions: save, read, update, delete, list, search, stats, pin, promote. Memory types: 'long_term' (permanent), 'short_term' (7 days), 'quick_note' (6 hours).",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING", "description": "Action: save, read, update, delete, list, search, stats, pin, promote"},
                        "key": {"type": "STRING", "description": "Memory identifier (required for most actions)"},
                        "content": {"type": "STRING", "description": "Content to store (required for save)"},
                        "category": {"type": "STRING", "description": "Category (e.g., 'personal', 'facts')"},
                        "memory_type": {"type": "STRING", "description": "Type: long_term, short_term, quick_note"},
                        "tags": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Tags for organization"},
                        "search_term": {"type": "STRING", "description": "Search query (for search action)"},
                        "limit": {"type": "INTEGER", "description": "Max results (default 20)"},
                        "new_type": {"type": "STRING", "description": "Target type for promote action"},
                        "pin": {"type": "BOOLEAN", "description": "Pin status for pin action"},
                    },
                    "required": ["action"],
                },
            ),
        ]),
    ]


class ToolHandler:
    def __init__(self, audio_mgr, osc, tracker, personality_mgr):
        self.audio = audio_mgr
        self.osc = osc
        self.tracker = tracker
        self.personality = personality_mgr
        self._tracking_task = None
        self.session = None

    async def handle(self, function_call) -> types.FunctionResponse:
        name = function_call.name
        args = dict(function_call.args) if function_call.args else {}
        
        # Memory function returns FunctionResponse directly
        if name == "memory":
            try:
                return await handle_memory_function_call(function_call)
            except Exception as e:
                logger.error(f"Memory tool failed: {e}")
                return types.FunctionResponse(
                    id=function_call.id,
                    name=name,
                    response={"error": str(e)},
                )
        
        try:
            result = await self._dispatch(name, args)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            result = {"error": str(e)}
        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response=result,
        )

    async def _dispatch(self, name, args):
        if name == "play_sound_effect":
            return await self._play_sfx(args["query"])
        elif name == "list_music":
            return {"files": self.audio.list_music()}
        elif name == "play_music":
            ok = self.audio.play_music(args["filename"], args.get("volume", 50))
            return {"result": "ok" if ok else "file not found"}
        elif name == "stop_music":
            self.audio.stop_music()
            return {"result": "ok"}
        elif name == "pause_music":
            ok = self.audio.pause_music()
            return {"result": "ok" if ok else "nothing playing"}
        elif name == "resume_music":
            ok = self.audio.resume_music()
            return {"result": "ok" if ok else "nothing paused"}
        elif name == "set_music_volume":
            ok = self.audio.set_music_volume(args["volume"])
            return {"result": "ok" if ok else "nothing playing"}
        elif name == "set_voice_boost":
            self.audio.set_boost(args["level"])
            return {"result": "ok"}
        elif name == "toggle_vrchat_mic":
            self.osc.toggle_voice()
            return {"result": "ok"}
        elif name == "follow_person":
            return await self._toggle_follow(args["enabled"])
        elif name == "list_personalities":
            return self.personality.list_personalities()
        elif name == "switch_personality":
            result = self.personality.switch(args["personality_id"])
            if "personality_prompt" in result and self.session:
                await self.session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=result["personality_prompt"])],
                    ),
                    turn_complete=True,
                )
            return {k: v for k, v in result.items() if k != "personality_prompt"}
        elif name == "get_current_personality":
            return self.personality.get_current()
        elif name == "vrchat_crouch":
            self.osc.toggle_crouch()
            return {"result": "ok"}
        elif name == "vrchat_crawl":
            self.osc.toggle_crawl()
            return {"result": "ok"}
        elif name == "vrchat_move":
            return await self._vrchat_move(args["direction"], args["duration"])
        elif name == "vrchat_stop":
            self.osc.stop_all_movement()
            return {"result": "ok"}
        elif name == "vrchat_jump":
            self.osc.jump()
            return {"result": "ok"}
        return {"error": "unknown function"}

    async def _play_sfx(self, query):
        from src.myinstants import search_sound, download_sound
        result = await search_sound(query)
        if not result:
            return {"error": "no sound found"}
        filepath = await download_sound(result["url"])
        if not filepath:
            return {"error": "download failed"}
        self.audio.play_sfx_file(filepath)
        return {"result": "ok", "name": result["name"]}

    async def _toggle_follow(self, enabled):
        if enabled:
            if self._tracking_task and not self._tracking_task.done():
                return {"result": "already following"}
            self.tracker.active = True
            self._tracking_task = asyncio.create_task(
                self.tracker.tracking_loop(self.osc)
            )
            return {"result": "ok"}
        else:
            self.tracker.active = False
            if self._tracking_task:
                await self._tracking_task
                self._tracking_task = None
            return {"result": "ok"}

    async def _vrchat_move(self, direction: str, duration: float):
        """Move in a direction for a specified duration."""
        direction = direction.lower()
        if direction not in ("forward", "backward", "left", "right"):
            return {"error": f"Invalid direction: {direction}. Use forward, backward, left, or right."}
        
        # Clamp duration between 0.1 and 600 seconds
        duration = max(0.1, min(600.0, duration))
        
        # Start movement
        self.osc.start_move(direction)
        
        # Wait for duration then stop
        await asyncio.sleep(duration)
        self.osc.stop_all_movement()
        
        return {"result": "ok", "direction": direction, "duration": duration}
