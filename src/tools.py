import asyncio
import logging
from google.genai import types
from src.memory import get_memory_tools, handle_memory_function_call, recall_memories
from src.emotions import generate_emotion_function_declarations, handle_emotion_function_call, get_emotion_system

logger = logging.getLogger(__name__)


def get_tool_declarations(config=None):
    tracker_enabled = True
    wanderer_enabled = True
    if config is not None:
        tracker_enabled = getattr(config, "tracker_enabled", True)
        wanderer_enabled = getattr(config, "wanderer_enabled", False)

    # Base function declarations - NO UNDERSCORES to avoid 1011 errors with Gemini Live
    function_decls = [
        types.FunctionDeclaration(
                name="searchSoundboard",
                description="Search the MyInstants soundboard for short audio clips. Returns results with IDs and titles. Use playSoundboard with the ID to play one.\n**Invocation Condition:** Call only when you want to browse results before playing, or when the user asks to search. For direct playback, use playSoundboard instead (it auto-searches).",
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
                description="Play a MyInstants soundboard clip by ID or name. Automatically searches MyInstants if not cached locally.\n**Invocation Condition:** Call directly when a sound clip would enhance the conversation. Do not search first. Do not ask for confirmation. Use repeat+delay to play it multiple times in a single call instead of calling this function repeatedly.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "soundId": {"type": "STRING", "description": "The ID of the soundboard clip to play from search results"},
                        "boost": {"type": "INTEGER", "description": "Bass boost/distortion level 0-10. 0=normal, higher=louder and more distorted like a blown-out mic. Great for funny earrape moments."},
                        "repeat": {"type": "INTEGER", "description": "Number of times to play the sound. Default 1. Max 25."},
                        "delay": {"type": "NUMBER", "description": "Delay in seconds between each repeat. Default 1.0. Min 0.1, max 10."},
                    },
                    "required": ["soundId"],
                },
            ),
            types.FunctionDeclaration(
                name="stopSoundboard",
                description="Stop all currently playing MyInstants soundboard clips immediately.\n**Invocation Condition:** Call when asked to stop a clip or when it is disruptive.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="playRandomSoundboard",
                description="Play random sound effects from previously searched MyInstants clips. Each repeat plays a different random sound.\n**Invocation Condition:** Call when asked for random sounds, surprise sounds, or sound chaos/spam.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "boost": {"type": "INTEGER", "description": "Bass boost/distortion level 0-10. Default 0."},
                        "repeat": {"type": "INTEGER", "description": "Number of random sounds to play. Default 1. Max 25."},
                        "delay": {"type": "NUMBER", "description": "Delay in seconds between each sound. Default 1.0. Min 0.1, max 10."},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="listMusic",
                description="List all available local music files that can be played.\n**Invocation Condition:** Call when asked what songs are available, or after a playMusic failure to get the correct filename.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="playMusic",
                description="Play a local music file by filename.\n**Invocation Condition:** Call when asked to play a song. Use exact filenames from listMusic results.",
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
                description="Stop currently playing music.\n**Invocation Condition:** Call when asked to stop the current song.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="pauseMusic",
                description="Pause the currently playing music. Can be resumed later.\n**Invocation Condition:** Call when asked to pause.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="resumeMusic",
                description="Resume paused music playback.\n**Invocation Condition:** Call when asked to resume or unpause.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="setMusicVolume",
                description="Adjust the music volume while playing. Only works during playback.\n**Invocation Condition:** Call when asked to change music volume.",
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
                description="Set voice boost level for loud distorted bass-boosted yelling effect on your microphone output.\n**Invocation Condition:** Call when asked to get loud or distorted, or for comedic yelling effects.",
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
                description="Mute or unmute the VRChat microphone.\n**Invocation Condition:** Call when asked to mute or unmute.",
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
                description="List all available personality modes that can be switched to.\n**Invocation Condition:** Call when asked what personalities are available.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="switchPersonality",
                description="Switch to a different personality mode. This changes behavior and response style.\n**Invocation Condition:** Call only when explicitly asked to switch, or when context unmistakably calls for it.",
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
                description="Get information about the currently active personality mode.\n**Invocation Condition:** Call when asked what mode you are in.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatCrouch",
                description="Toggle crouch in VRChat. Press once to crouch, press again to stand up.\n**Invocation Condition:** Call when asked to crouch or stand.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatCrawl",
                description="Toggle crawl/prone position in VRChat. Press once to go prone, press again to stand up.\n**Invocation Condition:** Call when asked to crawl or go prone.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatMove",
                description="Start walking in a direction in VRChat. Supports strafing (left/right) and sprinting. The avatar will keep moving until duration expires or vrchatStop is called.\n**Invocation Condition:** Call when asked to walk, run, or move somewhere.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "Direction to move: 'forward', 'backward', 'left' (strafe left), or 'right' (strafe right)"},
                        "duration": {"type": "NUMBER", "description": "How long to move in seconds (0.1 to 600). After this time, movement stops automatically."},
                        "speed": {"type": "STRING", "description": "Movement speed: 'slow' (careful walk), 'normal' (default walk), 'fast' (brisk walk), 'sprint' (full run)"},
                    },
                    "required": ["direction", "duration"],
                },
            ),
            types.FunctionDeclaration(
                name="vrchatStop",
                description="Stop all movement in VRChat immediately.\n**Invocation Condition:** Call immediately when asked to stop moving.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatJump",
                description="Make the avatar jump in VRChat.\n**Invocation Condition:** Call when asked to jump.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatGrab",
                description="Grab/pickup the item directly in front of you (center of your view) in VRChat. You must be looking straight at the item.\n**Invocation Condition:** Call when someone says 'grab this', 'pick that up', or similar. The item must be centered in your vision.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatDrop",
                description="Drop the item you are currently holding in VRChat.\n**Invocation Condition:** Call when someone says 'drop it', 'let go', 'put it down', or similar.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatUse",
                description="Use/interact with the item directly in front of you (center of your view) in VRChat. This activates interactable objects like buttons, doors, or pickups.\n**Invocation Condition:** Call when someone says 'use that', 'press that', 'interact with that', or similar. The item must be centered in your vision.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="vrchatLook",
                description="Smoothly turn/rotate the avatar left or right in VRChat. Uses the same smooth EMA turning as the follow system with gradual ramp up and ramp down.\n**Invocation Condition:** Call when asked to look left or right, turn around, or face a direction. Also useful for aiming your view at objects before grabbing/using them.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "Direction to turn: 'left' or 'right'"},
                        "duration": {"type": "NUMBER", "description": "How long to turn in seconds (0.1 to 10). Small values for slight adjustments, larger for big turns."},
                        "speed": {"type": "STRING", "description": "Turn speed: 'slow' (gentle glance), 'normal' (default), 'fast' (quick snap)"},
                    },
                    "required": ["direction", "duration"],
                },
            ),
            types.FunctionDeclaration(
                name="vrchatLookVertical",
                description="Smoothly tilt the avatar's view up or down in VRChat. Same smooth EMA turning as horizontal look.\n**Invocation Condition:** Call when asked to look up, look down, tilt head up/down, or check something above/below.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "Direction to look: 'up' or 'down'"},
                        "duration": {"type": "NUMBER", "description": "How long to look in seconds (0.1 to 10). Small values for slight tilts, larger for full up/down."},
                        "speed": {"type": "STRING", "description": "Tilt speed: 'slow' (gentle), 'normal' (default), 'fast' (quick snap)"},
                    },
                    "required": ["direction", "duration"],
                },
            ),
            # Memory system (unified action-based) - NO BOOLEAN/ARRAY types to avoid 1008
            types.FunctionDeclaration(
                name="memory",
                description="Persistent memory system. Actions: save, read, update, delete, list, search, stats, pin, promote. Memory types: long_term (permanent), short_term (7 days), quick_note (6 hours).\n**Invocation Condition:** Call with action=save when you learn something worth remembering. Call with action=search before asking someone a question you might already know. Always include actual usernames, never generic terms like 'User'.",
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
                description="Deep memory recall agent. Searches through ALL stored memories using AI to find and summarize relevant information.\n**Invocation Condition:** Call when you need to remember something specific about a person, event, or topic. More thorough than basic memory search. Use when someone references past events or asks about people you have met.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "What to recall — a person's name, topic, event, or question about past interactions"},
                        "context": {"type": "STRING", "description": "Why you need this info — helps the recall agent find the most relevant memories"},
                    },
                    "required": ["query"],
                },
            ),            types.FunctionDeclaration(
                name="switchTTSProvider",
                description="Switch the text-to-speech voice mid-session. Changes take effect immediately on the next spoken response. NEVER mention provider names, technology names, or internal voice IDs to the user. Just say you switched your voice.\n**Invocation Condition:** Call when asked to change voice, switch TTS, or use a different voice. Use listTTSProviders first to see what is available.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "provider": {"type": "STRING", "description": "Provider ID to switch to (e.g. 'gemini', 'chirp3_hd', 'hoppou', 'qwen3')"},
                        "voice": {"type": "STRING", "description": "A custom voice name OR a built-in voice name for the provider. Optional -- uses config default if omitted. NOT supported for 'gemini' provider."},
                    },
                    "required": ["provider"],
                },
            ),
            types.FunctionDeclaration(
                name="listTTSProviders",
                description="List available voice providers and the currently active one. Internal use only -- NEVER reveal provider names or IDs to the user.\n**Invocation Condition:** Call when you need to know what providers are available before switching.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="listVoices",
                description="List custom voices available for switching. Each voice has a display name and description. When telling the user about voices, use ONLY the display_name, never the internal ID or provider name.\n**Invocation Condition:** Call when asked what voices are available, or before switching to a custom voice.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="searchAvatars",
                description="Search for VRChat avatars by name. Returns up to 25 avatar names. Use switchAvatar with the exact name to switch.\n**Invocation Condition:** Call when asked to find, search, or look for avatars.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Avatar name or keyword to search for"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="switchAvatar",
                description="Switch to a VRChat avatar by name or ID. Checks the local cache first, then searches online if needed. Use the exact name from searchAvatars results.\n**Invocation Condition:** Call when asked to change avatar, switch avatar, or put on a specific avatar.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "nameOrId": {"type": "STRING", "description": "Avatar name (from search results or cache) or avatar ID (avtr_xxx)"},
                    },
                    "required": ["nameOrId"],
                },
            ),
            types.FunctionDeclaration(
                name="getInstancePlayers",
                description="Get a list of all players currently in the same VRChat instance. Returns each player's display name. Useful for knowing who is around you.\n**Invocation Condition:** Call when asked who is in the instance, who is here, who is around, or to list the people in the room/world.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "includeIds": {"type": "BOOLEAN", "description": "If true, also return user IDs alongside names. Default false."},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="invitePlayer",
                description="Invite a player to your current VRChat instance. Use the exact display name from getInstancePlayers or a user ID.\n**Invocation Condition:** Call when asked to invite someone to the instance or world.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "player": {"type": "STRING", "description": "Player display name or user ID (usr_xxx) to invite"},
                    },
                    "required": ["player"],
                },
            ),
            types.FunctionDeclaration(
                name="requestInvite",
                description="Request an invite from a player to join their instance.\n**Invocation Condition:** Call when asked to request an invite from someone.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "player": {"type": "STRING", "description": "Player display name or user ID (usr_xxx) to request invite from"},
                    },
                    "required": ["player"],
                },
            ),
            types.FunctionDeclaration(
                name="getOwnAvatar",
                description="Get information about your currently equipped VRChat avatar. Returns name, description, author, and performance ratings.\n**Invocation Condition:** Call when asked what avatar you are wearing, what your current avatar is, or for details about your avatar.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="getAvatarInfo",
                description="Get information about a VRChat avatar by its ID. Returns name, description, author, and performance ratings.\n**Invocation Condition:** Call when asked about a specific avatar's details using its ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "avatarId": {"type": "STRING", "description": "Avatar ID (avtr_xxx) to look up"},
                    },
                    "required": ["avatarId"],
                },
            ),
            types.FunctionDeclaration(
                name="searchWorlds",
                description="Search for VRChat worlds by name or keyword. Returns world names, IDs, author, capacity, player count, and favorites.\n**Invocation Condition:** Call when asked to find, search, or look for VRChat worlds or maps.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "World name or keyword to search for"},
                        "count": {"type": "INTEGER", "description": "Max results to return (1-25, default 10)"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="updateStatus",
                description="Update your VRChat profile status description, online status, and/or bio. At least one field must be provided.\n**Invocation Condition:** Call when asked to change your status, status description, bio, or profile text.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "statusDescription": {"type": "STRING", "description": "The short status description shown on your profile (the tagline under your name)"},
                        "status": {"type": "STRING", "description": "Online status: 'active', 'join me', 'ask me', 'busy', or 'offline'"},
                        "bio": {"type": "STRING", "description": "Your profile bio text"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="getFriendInfo",
                description="Look up a friend by name and get their current profile info including online status, status description, bio, pronouns, and platform. Searches your friends list by display name.\n**Invocation Condition:** Call when asked about a friend's status, whether they are online, or for info about a specific friend.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING", "description": "Friend's display name to look up"},
                    },
                    "required": ["name"],
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
                description="Start following a player visible on screen using YOLO vision tracking. Automatically detects and locks onto the nearest person.\n**Invocation Condition:** Call when asked to follow someone.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "mode": {"type": "STRING", "description": "Follow mode (default 'auto')"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="stopFollow",
                description="Stop following a player and halt all tracking movement.\n**Invocation Condition:** Call immediately when told to stop following.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="setFollowDistance",
                description="Set how close to follow the target player. Value is a fraction from 0.01 (very close) to 0.5 (far away). Default is 0.08.\n**Invocation Condition:** Call when asked to get closer or farther while following.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "value": {"type": "NUMBER", "description": "Target follow distance as bounding-box area fraction (0.01 to 0.5)"},
                    },
                    "required": ["value"],
                },
            ),
        ])

    if wanderer_enabled:
        function_decls.extend([
            types.FunctionDeclaration(
                name="startWander",
                description="Start autonomously wandering around the VRChat map. Uses depth perception to avoid walls and obstacles. Will walk, turn, look around, and jump on its own.\n**Invocation Condition:** Call when asked to wander, explore, walk around, or roam freely.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="stopWander",
                description="Stop autonomous wandering and halt all movement.\n**Invocation Condition:** Call when asked to stop wandering or exploring.",
                parameters={"type": "OBJECT", "properties": {}},
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
        self.wanderer = None
        self.personality = personality_mgr
        self.config = config
        self.session = None
        self.live_session = None
        self.vrchat_api = None
        self._current_avatar_id = None
        self.instance_monitor = None

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
            return await self._play_sfx(
                args["soundId"],
                boost=int(args.get("boost", 0)),
                repeat=int(args.get("repeat", 1)),
                delay=float(args.get("delay", 1.0)),
            )
        elif name == "stopSoundboard":
            self.audio.stop_sfx()
            return {"result": "ok"}
        elif name == "playRandomSoundboard":
            return await self._play_random_sfx(
                boost=int(args.get("boost", 0)),
                repeat=int(args.get("repeat", 1)),
                delay=float(args.get("delay", 1.0)),
            )
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
        elif name == "startWander":
            if self.wanderer is None:
                return {"result": "error", "message": "Wanderer is disabled in config"}
            # Stop following if active
            if self.tracker and self.tracker.active:
                self.tracker.stopfollow()
            return self.wanderer.start()
        elif name == "stopWander":
            if self.wanderer is None:
                return {"result": "error", "message": "Wanderer is disabled in config"}
            return self.wanderer.stop()
        elif name == "listPersonalities":
            result = self.personality.list_personalities()
            result["result"] = "ok"
            return result
        elif name == "switchPersonality":
            result = self.personality.switch(args["personalityId"])
            # Auto-switch avatar if personality has one configured
            avatar_id = result.get("avatar_id")
            if avatar_id and avatar_id != self._current_avatar_id:
                try:
                    api = self._get_vrchat_api()
                    av_result = await api.select_avatar(avatar_id)
                    if av_result.get("result") == "ok" or av_result.get("avatar_id"):
                        self._current_avatar_id = avatar_id
                        logger.info(f"Auto-switched avatar to {avatar_id} for personality '{args['personalityId']}'")
                    else:
                        logger.warning(f"Failed to auto-switch avatar: {av_result.get('error', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Avatar auto-switch failed: {e}")
            response = {k: v for k, v in result.items() if k != "avatar_id"}
            return response
        elif name == "getCurrentPersonality":
            result = self.personality.get_current()
            result["result"] = "ok"
            return result
        elif name == "vrchatCrouch":
            self.osc.toggle_crouch()
            emo = get_emotion_system()
            if emo:
                emo._crouching = not emo._crouching
                if emo._crouching and emo._is_speaking:
                    emo.stop_speaking()
            return {"result": "ok"}
        elif name == "vrchatCrawl":
            self.osc.toggle_crawl()
            emo = get_emotion_system()
            if emo:
                emo._crouching = not emo._crouching
                if emo._crouching and emo._is_speaking:
                    emo.stop_speaking()
            return {"result": "ok"}
        elif name == "vrchatMove":
            return await self._vrchat_move(args["direction"], args["duration"], args.get("speed", "normal"))
        elif name == "vrchatStop":
            self.osc.stop_all_movement()
            return {"result": "ok"}
        elif name == "vrchatJump":
            self.osc.jump()
            return {"result": "ok"}
        elif name == "vrchatGrab":
            await asyncio.to_thread(self.osc.grab)
            return {"result": "ok"}
        elif name == "vrchatDrop":
            await asyncio.to_thread(self.osc.drop)
            return {"result": "ok"}
        elif name == "vrchatUse":
            await asyncio.to_thread(self.osc.use)
            return {"result": "ok"}
        elif name == "vrchatLook":
            direction = args.get("direction", "right")
            duration = min(max(float(args.get("duration", 0.5)), 0.1), 10.0)
            speed = args.get("speed", "normal")
            asyncio.get_event_loop().run_in_executor(None, self.osc.look, direction, duration, speed)
            return {"result": "ok"}
        elif name == "vrchatLookVertical":
            direction = args.get("direction", "up")
            duration = min(max(float(args.get("duration", 0.5)), 0.1), 10.0)
            speed = args.get("speed", "normal")
            asyncio.get_event_loop().run_in_executor(None, self.osc.look_vertical, direction, duration, speed)
            return {"result": "ok"}
        elif name == "switchTTSProvider":
            return await self._switch_tts(args.get("provider", ""), args.get("voice"))
        elif name == "listTTSProviders":
            return self._list_tts_providers()
        elif name == "listVoices":
            return self._list_voices()
        elif name == "searchAvatars":
            return await self._search_avatars(args["query"])
        elif name == "switchAvatar":
            return await self._switch_avatar(args["nameOrId"])
        elif name == "getInstancePlayers":
            return self._get_instance_players(args.get("includeIds", False))
        elif name == "invitePlayer":
            return await self._invite_player(args["player"])
        elif name == "requestInvite":
            return await self._request_invite(args["player"])
        elif name == "getOwnAvatar":
            return await self._get_own_avatar()
        elif name == "getAvatarInfo":
            return await self._get_avatar_info(args["avatarId"])
        elif name == "searchWorlds":
            return await self._search_worlds(args["query"], int(args.get("count", 10)))
        elif name == "updateStatus":
            return await self._update_status(args.get("statusDescription"), args.get("status"), args.get("bio"))
        elif name == "getFriendInfo":
            return await self._get_friend_info(args["name"])
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

    async def _play_sfx(self, sound_id, boost=0, repeat=1, delay=1.0):
        from src.myinstants import get_sound_url, download_sound, search_sounds
        repeat = min(max(int(repeat), 1), 25)
        delay = min(max(float(delay), 0.1), 10.0)
        logger.info(f"playSoundboard: playing ID '{sound_id}' with boost={boost}, repeat={repeat}, delay={delay}")
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
        logger.info(f"playSoundboard: playing '{entry['title']}' from {filepath} (boost={boost}, repeat={repeat})")
        self.audio.play_sfx_file(filepath, boost=boost)
        if repeat > 1:
            async def _repeat_sfx():
                for i in range(1, repeat):
                    await asyncio.sleep(delay)
                    self.audio.play_sfx_file(filepath, boost=boost)
            asyncio.create_task(_repeat_sfx())
        return {"result": "ok", "name": entry["title"], "boost": boost, "repeat": repeat}

    async def _play_random_sfx(self, boost=0, repeat=1, delay=1.0):
        from src.myinstants import get_random_sounds, download_sound
        repeat = min(max(int(repeat), 1), 25)
        delay = min(max(float(delay), 0.1), 10.0)
        picks = get_random_sounds(repeat)
        if not picks:
            return {"result": "error", "message": "No sounds cached yet. Use searchSoundboard or playSoundboard first to build up the sound library."}
        # Play first sound immediately, schedule the rest in background
        first = picks[0]
        if first.get("_local"):
            filepath = first["mp3"]
        else:
            filepath = await download_sound(first["mp3"])
        if not filepath:
            return {"result": "error", "message": f"Download failed for '{first['title']}'"}
        logger.info(f"playRandomSoundboard: [1/{repeat}] playing '{first['title']}' (boost={boost})")
        self.audio.play_sfx_file(filepath, boost=boost)
        played_names = [first["title"]]
        if len(picks) > 1:
            remaining = picks[1:]
            async def _play_remaining():
                for i, pick in enumerate(remaining):
                    await asyncio.sleep(delay)
                    if pick.get("_local"):
                        fp = pick["mp3"]
                    else:
                        fp = await download_sound(pick["mp3"])
                    if not fp:
                        logger.warning(f"playRandomSoundboard: download failed for '{pick['title']}'")
                        continue
                    logger.info(f"playRandomSoundboard: [{i+2}/{repeat}] playing '{pick['title']}' (boost={boost})")
                    self.audio.play_sfx_file(fp, boost=boost)
            asyncio.create_task(_play_remaining())
            played_names.extend(p["title"] for p in remaining)
        return {"result": "ok", "played": played_names, "count": len(played_names), "boost": boost}

    async def _vrchat_move(self, direction: str, duration: float, speed: str = "normal"):
        """Move in a direction for a specified duration with speed control."""
        direction = direction.lower()
        if direction not in ("forward", "backward", "left", "right"):
            return {"result": "error", "message": f"Invalid direction: {direction}. Use forward, backward, left, or right."}
        
        # Clamp duration between 0.1 and 600 seconds
        duration = max(0.1, min(600.0, duration))
        
        # Start movement with speed
        self.osc.start_move(direction, speed)
        
        # Schedule stop in background so tool response returns immediately
        async def _stop_after():
            await asyncio.sleep(duration)
            self.osc.stop_all_movement()
        asyncio.create_task(_stop_after())
        
        return {"result": "ok", "direction": direction, "duration": duration, "speed": speed}

    async def _switch_tts(self, provider_name, voice=None):
        provider_name = provider_name.strip().lower()
        allowed = [p.strip().lower() for p in (self.config.tts_switchable_providers if self.config else ["gemini"])]
        if provider_name not in allowed:
            return {"result": "error", "message": f"Provider '{provider_name}' not allowed. Allowed: {allowed}"}
        if not self.live_session:
            return {"result": "error", "message": "No active session"}

        # Resolve custom voice from voices.yml if provided
        voice_override = None
        if voice and self.config:
            voice_def = self.config.get_voice(voice)
            if voice_def and provider_name in voice_def:
                voice_override = voice_def[provider_name]
                logger.info(f"switchTTSProvider: using custom voice '{voice}' for {provider_name}")

        new_provider = None
        if provider_name == "gemini":
            if voice:
                return {"result": "error", "message": "Cannot change Gemini voice mid-session. Gemini voice requires a full session restart."}
        elif provider_name == "qwen3":
            from src.tts import QwenTTSProvider
            new_provider = QwenTTSProvider(self.config, voice_override=voice_override)
        elif provider_name == "hoppou":
            from src.tts import HoppouTTSProvider
            if not voice_override and voice:
                voice_override = {"voice": voice}
            new_provider = HoppouTTSProvider(self.config, voice_override=voice_override)
        elif provider_name == "chirp3_hd":
            from src.tts import Chirp3HDTTSProvider
            if not voice_override and voice:
                voice_override = {"voice": voice}
            new_provider = Chirp3HDTTSProvider(self.config, voice_override=voice_override)
        else:
            return {"result": "error", "message": f"Unknown provider: {provider_name}"}

        self.live_session.switch_tts_provider(new_provider)
        result = {"result": "ok", "provider": provider_name}
        if voice:
            result["voice"] = voice
        logger.info(f"switchTTSProvider: switched to '{provider_name}'" + (f" voice='{voice}'" if voice else ""))
        return result

    def _list_tts_providers(self):
        allowed = self.config.tts_switchable_providers if self.config else ["gemini"]
        current_tts = self.live_session._tts if self.live_session else None
        if current_tts is None:
            current = "gemini"
        else:
            name = type(current_tts).__name__
            mapping = {"QwenTTSProvider": "qwen3", "HoppouTTSProvider": "hoppou", "Chirp3HDTTSProvider": "chirp3_hd"}
            current = mapping.get(name, name)
        return {"result": "ok", "providers": allowed, "current": current}

    def _list_voices(self):
        voices = {}
        if self.config:
            for vname, vdef in self.config.list_voices().items():
                voices[vname] = {
                    "display_name": vdef.get("display_name", vname),
                    "description": vdef.get("description", ""),
                    "providers": [p for p in ("qwen3", "hoppou", "chirp3_hd") if p in vdef],
                }
        return {"result": "ok", "voices": voices}

    def _get_vrchat_api(self):
        if self.vrchat_api is None:
            from src.vrchatapi import VRChatAPI
            self.vrchat_api = VRChatAPI(self.config)
        return self.vrchat_api

    async def _search_avatars(self, query):
        from src.avatars import search_avatars
        results = await search_avatars(query, max_results=25)
        if not results:
            return {"result": "error", "message": f"No avatars found for '{query}'"}
        names = [av["name"] for av in results]
        return {"result": "ok", "count": len(results), "avatars": names}

    async def _switch_avatar(self, name_or_id):
        from src.avatars import switch_avatar
        api = self._get_vrchat_api()
        result = await switch_avatar(api, name_or_id)
        if result.get("result") == "ok":
            self._current_avatar_id = result.get("avatar_id")
        return result

    def _get_instance_players(self, include_ids=False):
        if not self.instance_monitor:
            return {"result": "error", "message": "Instance monitor not available"}
        players = self.instance_monitor.get_players()
        location = self.instance_monitor.current_location
        if not players:
            if not location:
                return {"result": "ok", "message": "Not currently in a VRChat instance", "players": [], "count": 0}
            return {"result": "ok", "message": "No players detected yet", "location": location, "players": [], "count": 0}
        if include_ids:
            player_list = [{"name": p["name"], "id": p["id"]} for p in players]
        else:
            player_list = [p["name"] for p in players]
        return {"result": "ok", "location": location, "count": len(player_list), "players": player_list}

    def _resolve_player_id(self, player):
        """Resolve player name to user ID using instance monitor then friends cache."""
        if player.startswith("usr_"):
            return player
        player_lower = player.lower()
        if self.instance_monitor:
            for p in self.instance_monitor.get_players():
                if p["name"].lower() == player_lower:
                    return p["id"]
        from src.vrchatapi import VRChatAPI
        for f in VRChatAPI.load_cached_friends():
            if f.get("displayName", "").lower() == player_lower:
                return f["id"]
        return None

    async def _invite_player(self, player):
        api = self._get_vrchat_api()
        user_id = self._resolve_player_id(player)
        if not user_id:
            return {"result": "error", "message": f"Could not find player '{player}' -- use getInstancePlayers first or provide a user ID (usr_xxx)"}
        location = self.instance_monitor.current_location if self.instance_monitor else ""
        if not location:
            user_data = await api.get_current_user()
            if isinstance(user_data, dict):
                location = user_data.get("location", "") or ""
        if not location or location in ("", "offline", "private"):
            return {"result": "error", "message": "Not currently in a VRChat instance"}
        result = await api.invite_user(user_id, location)
        return result

    async def _request_invite(self, player):
        api = self._get_vrchat_api()
        user_id = self._resolve_player_id(player)
        if not user_id:
            return {"result": "error", "message": f"Could not find player '{player}' -- use getInstancePlayers first or provide a user ID (usr_xxx)"}
        result = await api.request_invite(user_id)
        return result

    async def _get_own_avatar(self):
        api = self._get_vrchat_api()
        data = await api.get_own_avatar()
        if "error" in data:
            return data
        return {
            "result": "ok",
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "author": data.get("authorName", ""),
            "id": data.get("id", ""),
            "performance": data.get("performance", {}),
        }

    async def _get_avatar_info(self, avatar_id):
        api = self._get_vrchat_api()
        data = await api.get_avatar(avatar_id)
        if "error" in data:
            return data
        return {
            "result": "ok",
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "author": data.get("authorName", ""),
            "id": data.get("id", ""),
            "performance": data.get("performance", {}),
        }

    async def _search_worlds(self, query, count=10):
        api = self._get_vrchat_api()
        n = max(1, min(count, 25))
        data = await api.search_worlds(query, n=n)
        if isinstance(data, dict) and "error" in data:
            return data
        if not data:
            return {"result": "error", "message": f"No worlds found for '{query}'"}
        worlds = []
        for w in data:
            worlds.append({
                "name": w.get("name", ""),
                "id": w.get("id", ""),
                "author": w.get("authorName", ""),
                "players": w.get("occupants", 0),
                "capacity": w.get("capacity", 0),
                "favorites": w.get("favorites", 0),
            })
        return {"result": "ok", "count": len(worlds), "worlds": worlds}

    async def _update_status(self, status_description=None, status=None, bio=None):
        if bio is not None and self.config and not self.config.vrchat_api_allow_bio_edit:
            return {"result": "error", "message": "Bio editing is disabled."}
        api = self._get_vrchat_api()
        result = await api.update_status(
            status_description=status_description,
            status=status,
            bio=bio,
        )
        return result

    async def _get_friend_info(self, name):
        from src.vrchatapi import VRChatAPI
        name_lower = name.lower()
        # Search friends cache for matching name
        friends = VRChatAPI.load_cached_friends()
        match = None
        for f in friends:
            if f.get("displayName", "").lower() == name_lower:
                match = f
                break
        if not match:
            # Partial match
            matches = [f for f in friends if name_lower in f.get("displayName", "").lower()]
            if len(matches) == 1:
                match = matches[0]
            elif len(matches) > 1:
                names = [f["displayName"] for f in matches[:10]]
                return {"result": "error", "message": f"Multiple friends match '{name}': {', '.join(names)}"}
            else:
                return {"result": "error", "message": f"No friend named '{name}' found in friends list"}
        # Fetch live profile from API
        api = self._get_vrchat_api()
        data = await api.get_user(match["id"])
        if isinstance(data, dict) and "error" in data:
            return data
        return {
            "result": "ok",
            "displayName": data.get("displayName", ""),
            "status": data.get("status", ""),
            "statusDescription": data.get("statusDescription", ""),
            "state": data.get("state", "offline"),
            "bio": data.get("bio", ""),
            "pronouns": data.get("pronouns", ""),
            "last_platform": data.get("last_platform", ""),
            "last_login": data.get("last_login", ""),
            "isFriend": data.get("isFriend", False),
        }
