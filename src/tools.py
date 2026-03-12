import asyncio
import logging
from google.genai import types
from src.memory import get_memory_tools, handle_memory_function_call, recall_memories
from src.emotions import generate_emotion_function_declarations, handle_emotion_function_call, get_emotion_system

logger = logging.getLogger(__name__)


def get_tool_declarations(config=None):
    tracker_enabled = True
    if config is not None:
        tracker_enabled = getattr(config, "tracker_enabled", True)

    # Base function declarations - NO UNDERSCORES to avoid 1011 errors with Gemini Live
    function_decls = [
        types.FunctionDeclaration(
                name="searchSoundboard",
                description="Search the MyInstants soundboard for short audio clips. Returns results with IDs and titles. Use playSoundboard with the ID to play one. This is NOT for music - use playMusic/listMusic for songs.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Soundboard clip name to search for on MyInstants"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="playSoundboard",
                description="Play a MyInstants soundboard clip by ID or name. Automatically searches MyInstants if not cached locally. This is NOT for music - use playMusic for songs.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "soundId": {"type": "STRING", "description": "The ID of the soundboard clip to play from search results"},
                        "boost": {"type": "INTEGER", "description": "Bass boost/distortion level 0-10. 0=normal, higher=louder and more distorted like a blown-out mic. Great for funny earrape moments."},
                    },
                    "required": ["soundId"],
                },
            ),
            types.FunctionDeclaration(
                name="stopSoundboard",
                description="Stop all currently playing MyInstants soundboard clips immediately.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="listMusic",
                description="List all available local music files that can be played",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="playMusic",
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
                name="stopMusic",
                description="Stop currently playing music",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="pauseMusic",
                description="Pause the currently playing music. Can be resumed later.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="resumeMusic",
                description="Resume paused music playback.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="setMusicVolume",
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
                name="setVoiceBoost",
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
                name="toggleVrchatMic",
                description="Mute or unmute the VRChat microphone",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "muted": {"type": "STRING", "description": "Set to 'true' to mute, 'false' to unmute"},
                    },
                    "required": ["muted"],
                },
            ),
            types.FunctionDeclaration(
                name="listPersonalities",
                description="List all available personality modes that can be switched to",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="switchPersonality",
                description="Switch to a different personality mode. This changes behavior and response style.",
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
                description="Get information about the currently active personality mode",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatCrouch",
                description="Toggle crouch in VRChat. Press once to crouch, press again to stand up.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatCrawl",
                description="Toggle crawl/prone position in VRChat. Press once to go prone, press again to stand up.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatMove",
                description="Start walking in a direction in VRChat. The avatar will keep moving until stopMovement is called.",
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
                name="vrchatStop",
                description="Stop all movement in VRChat immediately.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatJump",
                description="Make the avatar jump in VRChat.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            # Memory system (unified action-based) - NO BOOLEAN/ARRAY types to avoid 1008
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
                        "memoryType": {"type": "STRING", "description": "Type: long_term, short_term, quick_note"},
                        "tags": {"type": "STRING", "description": "Comma-separated tags for organization (e.g., 'important,friend,vrc')"},
                        "searchTerm": {"type": "STRING", "description": "Search query (for search action)"},
                        "limit": {"type": "INTEGER", "description": "Max results (default 20)"},
                        "newType": {"type": "STRING", "description": "Target type for promote action"},
                        "pin": {"type": "STRING", "description": "Set to 'true' to pin, 'false' to unpin (pinned memories won't auto-delete)"},
                    },
                    "required": ["action"],
                },
            ),
            types.FunctionDeclaration(
                name="recallMemories",
                description="Deep memory recall agent. Searches through ALL stored memories using AI to find and summarize relevant information. Use this when you need to remember something specific about a person, event, or topic. Much more thorough than the basic memory search.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "What to recall — a person's name, topic, event, or question about past interactions"},
                        "context": {"type": "STRING", "description": "Why you need this info — helps the recall agent find the most relevant memories"},
                    },
                    "required": ["query"],
                },
            ),
        ]
    
    # Add emotion function declarations if enabled
    if config:
        emotion_decls = generate_emotion_function_declarations(config)
        for decl in emotion_decls:
            function_decls.append(types.FunctionDeclaration(
                name=decl["name"],
                description=decl["description"],
                parameters=decl["parameters"],
            ))

    # Player following tools (only if tracker is enabled in config)
    if tracker_enabled:
        function_decls.extend([
            types.FunctionDeclaration(
                name="startFollow",
                description="Start following a player visible on screen using YOLO vision tracking. Automatically detects and locks onto the nearest person.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "mode": {"type": "STRING", "description": "Follow mode (default 'auto')"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="stopFollow",
                description="Stop following a player and halt all tracking movement.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="setFollowDistance",
                description="Set how close to follow the target player. Value is a fraction from 0.01 (very close) to 0.5 (far away). Default is 0.08.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "value": {"type": "NUMBER", "description": "Target follow distance as bounding-box area fraction (0.01 to 0.5)"},
                    },
                    "required": ["value"],
                },
            ),
        ])

    return [
        types.Tool(google_search=types.GoogleSearch()),
        types.Tool(function_declarations=function_decls),
    ]


class ToolHandler:
    def __init__(self, audio_mgr, osc, tracker, personality_mgr, config=None):
        self.audio = audio_mgr
        self.osc = osc
        self.tracker = tracker
        self.personality = personality_mgr
        self.config = config
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
                    response={"result": "error", "message": str(e)},
                )
        
        # Memory recall sub-agent
        if name == "recallMemories":
            try:
                if self.osc:
                    self.osc.send_chatbox("Thinking about the past...")
                api_key = self.config.api_key if self.config else ""
                personality_prompt = ""
                if self.personality:
                    current = self.personality.get_current()
                    personality_prompt = current.get("prompt", "")
                result = await recall_memories(
                    query=args.get("query", ""),
                    context=args.get("context", ""),
                    api_key=api_key,
                    personality_prompt=personality_prompt,
                )
                return types.FunctionResponse(
                    id=function_call.id,
                    name=name,
                    response=result,
                )
            except Exception as e:
                logger.error(f"Recall agent failed: {e}")
                return types.FunctionResponse(
                    id=function_call.id,
                    name=name,
                    response={"result": "error", "message": str(e)},
                )
        
        # Emotion functions return FunctionResponse directly
        if name in ("emotion", "stopAnimation"):
            try:
                return await handle_emotion_function_call(function_call)
            except Exception as e:
                logger.error(f"Emotion tool failed: {e}")
                return types.FunctionResponse(
                    id=function_call.id,
                    name=name,
                    response={"result": "error", "message": str(e)},
                )
        
        try:
            result = await self._dispatch(name, args)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            result = {"result": "error", "message": str(e)}
        
        # CRITICAL: Always return a valid FunctionResponse to prevent 1008 errors
        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response=result if result else {"result": "ok"},
        )

    async def _dispatch(self, name, args):
        if name == "searchSoundboard":
            return await self._search_sfx(args["query"])
        elif name == "playSoundboard":
            return await self._play_sfx(args["soundId"], boost=int(args.get("boost", 0)))
        elif name == "stopSoundboard":
            self.audio.stop_sfx()
            return {"result": "ok"}
        elif name == "listMusic":
            return {"result": "ok", "files": self.audio.list_music()}
        elif name == "playMusic":
            ok = self.audio.play_music(args["filename"], args.get("volume", 50))
            if ok:
                return {"result": "ok", "message": "playing"}
            return {"result": "error", "message": "file not found. Use listMusic to see available files and try again with the exact filename"}
        elif name == "stopMusic":
            self.audio.stop_music()
            return {"result": "ok"}
        elif name == "pauseMusic":
            ok = self.audio.pause_music()
            return {"result": "ok" if ok else "error", "message": "paused" if ok else "nothing playing"}
        elif name == "resumeMusic":
            ok = self.audio.resume_music()
            return {"result": "ok" if ok else "error", "message": "resumed" if ok else "nothing paused"}
        elif name == "setMusicVolume":
            ok = self.audio.set_music_volume(args["volume"])
            return {"result": "ok" if ok else "error", "message": "volume set" if ok else "nothing playing"}
        elif name == "setVoiceBoost":
            self.audio.set_boost(args["level"])
            return {"result": "ok", "level": args["level"]}
        elif name == "toggleVrchatMic":
            self.osc.toggle_voice()
            return {"result": "ok"}
        elif name == "startFollow":
            if self.tracker is None:
                return {"result": "error", "message": "Tracker is disabled in config"}
            return self.tracker.startfollow(args.get("mode", "auto"))
        elif name == "stopFollow":
            if self.tracker is None:
                return {"result": "error", "message": "Tracker is disabled in config"}
            return self.tracker.stopfollow()
        elif name == "setFollowDistance":
            if self.tracker is None:
                return {"result": "error", "message": "Tracker is disabled in config"}
            return self.tracker.setfollowdistance(args["value"])
        elif name == "listPersonalities":
            result = self.personality.list_personalities()
            result["result"] = "ok"
            return result
        elif name == "switchPersonality":
            result = self.personality.switch(args["personalityId"])
            if "personality_prompt" in result and self.session:
                await self.session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=result["personality_prompt"])],
                    ),
                    turn_complete=True,
                )
            return {k: v for k, v in result.items() if k != "personality_prompt"}
        elif name == "getCurrentPersonality":
            result = self.personality.get_current()
            result["result"] = "ok"
            return result
        elif name == "vrchatCrouch":
            self.osc.toggle_crouch()
            return {"result": "ok"}
        elif name == "vrchatCrawl":
            self.osc.toggle_crawl()
            return {"result": "ok"}
        elif name == "vrchatMove":
            return await self._vrchat_move(args["direction"], args["duration"])
        elif name == "vrchatStop":
            self.osc.stop_all_movement()
            return {"result": "ok"}
        elif name == "vrchatJump":
            self.osc.jump()
            return {"result": "ok"}
        # Default fallback - always return something
        return {"result": "error", "message": f"unknown function: {name}"}

    async def _search_sfx(self, query):
        from src.myinstants import search_sounds
        logger.info(f"searchSoundboard: searching MyInstants for '{query}'")
        results = await search_sounds(query)
        if not results:
            logger.warning(f"searchSoundboard: no sounds found for '{query}'")
            return {"result": "error", "message": "no sounds found on MyInstants"}
        logger.info(f"searchSoundboard: found {len(results)} results")
        return {"result": "ok", "sounds": results}

    async def _play_sfx(self, sound_id, boost=0):
        from src.myinstants import get_sound_url, download_sound, search_sounds
        logger.info(f"playSoundboard: playing ID '{sound_id}' with boost={boost}")
        entry = get_sound_url(sound_id)
        if not entry:
            # Auto-search MyInstants if not found locally
            logger.info(f"playSoundboard: '{sound_id}' not cached, auto-searching MyInstants...")
            results = await search_sounds(sound_id)
            if results:
                entry = get_sound_url(results[0]["id"])
            if not entry:
                return {"result": "error", "message": f"Sound '{sound_id}' not found on MyInstants either. Try a different search query with searchSoundboard."}
        # If it's already a local cached file, use it directly
        if entry.get("_local"):
            filepath = entry["mp3"]
        else:
            filepath = await download_sound(entry["mp3"])
        if not filepath:
            logger.warning(f"playSoundboard: download failed for '{entry['title']}'")
            return {"result": "error", "message": "download failed"}
        logger.info(f"playSoundboard: playing '{entry['title']}' from {filepath} (boost={boost})")
        self.audio.play_sfx_file(filepath, boost=boost)
        return {"result": "ok", "name": entry["title"], "boost": boost}

    async def _vrchat_move(self, direction: str, duration: float):
        """Move in a direction for a specified duration."""
        direction = direction.lower()
        if direction not in ("forward", "backward", "left", "right"):
            return {"result": "error", "message": f"Invalid direction: {direction}. Use forward, backward, left, or right."}
        
        # Clamp duration between 0.1 and 600 seconds
        duration = max(0.1, min(600.0, duration))
        
        # Start movement
        self.osc.start_move(direction)
        
        # Wait for duration then stop
        await asyncio.sleep(duration)
        self.osc.stop_all_movement()
        
        return {"result": "ok", "direction": direction, "duration": duration}
