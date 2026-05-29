import asyncio
import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class SoundboardTools(BaseTool):
    tool_key = "soundboard"

    def __init__(self, handler):
        super().__init__(handler)
        # background tasks spawned by play* with repeat>1. stopSoundboard
        # needs to cancel these or the remaining repeats keep firing after
        # stop_sfx() killed the currently audible one.
        self._repeat_tasks: list[asyncio.Task] = []

    def _track_repeat_task(self, coro):
        task = asyncio.create_task(coro)
        self._repeat_tasks.append(task)
        task.add_done_callback(lambda t: self._repeat_tasks.remove(t) if t in self._repeat_tasks else None)
        return task

    def _cancel_repeat_tasks(self):
        count = 0
        for task in list(self._repeat_tasks):
            if not task.done():
                task.cancel()
                count += 1
        self._repeat_tasks.clear()
        return count

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="searchSoundboard",
                description="Search the MyInstants soundboard for short audio clips. Returns results with IDs and titles. Use playSoundboard with the ID to play one.\n**Invocation Condition:** Call only when you want to browse results before playing, or when the user asks to search. For direct playback, use playSoundboard instead (it auto-searches).\n**After calling:** When telling the user what was found, ALWAYS read out the actual clip titles from the results so they know what to pick. Never say something vague like 'I found a few' or 'which one would you like' without listing the titles, the user has not seen the search output.",
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
                description="Play a MyInstants soundboard clip by ID or name. Automatically searches MyInstants if not cached locally. ONLY for short sound effects and meme clips (sad violin, vine boom, laugh track, etc.), NOT for actual songs or music. ANY request to 'play [song name]' should use playMusic instead.\n**Invocation Condition:** Call directly when a sound clip would enhance the conversation. Do not search first. Do not ask for confirmation. Use repeat+delay to play it multiple times in a single call instead of calling this function repeatedly.",
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
                description="Stop ALL currently playing MyInstants soundboard clips immediately, including any pending repeats from playSoundboard or playRandomSoundboard. Works even if a batch of 25 sounds is in flight.\n**Invocation Condition:** Call when asked to stop a clip, stop the spam, or when sounds are disruptive.",
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
        ]

    async def handle(self, name, args):
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
            cancelled = self._cancel_repeat_tasks()
            self.audio.stop_sfx()
            if cancelled:
                logger.info(f"stopSoundboard: cancelled {cancelled} pending repeat task(s)")
            return {"result": "ok", "cancelled_pending": cancelled}
        elif name == "playRandomSoundboard":
            return await self._play_random_sfx(
                boost=int(args.get("boost", 0)),
                repeat=int(args.get("repeat", 1)),
                delay=float(args.get("delay", 1.0)),
            )
        return None

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
            logger.info(f"playSoundboard: '{sound_id}' not cached, auto-searching MyInstants...")
            results = await search_sounds(sound_id)
            if results:
                entry = get_sound_url(results[0]["id"])
            if not entry:
                return {"result": "error", "message": f"Sound '{sound_id}' not found on MyInstants either. Try a different search query with searchSoundboard."}
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
                try:
                    for i in range(1, repeat):
                        await asyncio.sleep(delay)
                        self.audio.play_sfx_file(filepath, boost=boost)
                except asyncio.CancelledError:
                    logger.info(f"playSoundboard: repeat loop cancelled for '{entry['title']}'")
                    raise
            self._track_repeat_task(_repeat_sfx())
        return {"result": "ok", "name": entry["title"], "boost": boost, "repeat": repeat}

    async def _play_random_sfx(self, boost=0, repeat=1, delay=1.0):
        from src.myinstants import get_random_sounds, download_sound
        repeat = min(max(int(repeat), 1), 25)
        delay = min(max(float(delay), 0.1), 10.0)
        picks = get_random_sounds(repeat)
        if not picks:
            return {"result": "error", "message": "No sounds cached yet. Use searchSoundboard or playSoundboard first to build up the sound library."}
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
                try:
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
                except asyncio.CancelledError:
                    logger.info("playRandomSoundboard: remaining loop cancelled")
                    raise
            self._track_repeat_task(_play_remaining())
            played_names.extend(p["title"] for p in remaining)
        return {"result": "ok", "played": played_names, "count": len(played_names), "boost": boost}
