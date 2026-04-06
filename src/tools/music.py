import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class MusicTools(BaseTool):

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="listMusic",
                description="List all available local music files that can be played. You MUST read the song names out loud to the user after calling this -- they cannot see the tool response.\n**Invocation Condition:** Call when asked what songs are available, or after a playMusic failure to get the correct filename.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="playMusic",
                description="Play a local music file by filename. ALWAYS call listMusic FIRST to get exact filenames before calling this. Do not guess filenames. ANY request to 'play [song name]' should use this tool, NOT playSoundboard.\n**Invocation Condition:** Call when asked to play a song, music, or track from your local library.",
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
        ]

    async def handle(self, name, args):
        if name == "listMusic":
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
        return None
