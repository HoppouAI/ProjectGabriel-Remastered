import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)

SCALE_OPTIONS = [
    "SCALE_UNSPECIFIED",
    "C_MAJOR_A_MINOR", "D_FLAT_MAJOR_B_FLAT_MINOR", "D_MAJOR_B_MINOR",
    "E_FLAT_MAJOR_C_MINOR", "E_MAJOR_D_FLAT_MINOR", "F_MAJOR_D_MINOR",
    "G_FLAT_MAJOR_E_FLAT_MINOR", "G_MAJOR_E_MINOR", "A_FLAT_MAJOR_F_MINOR",
    "A_MAJOR_G_FLAT_MINOR", "B_FLAT_MAJOR_G_MINOR", "B_MAJOR_A_FLAT_MINOR",
]


@register_tool
class MusicGenTools(BaseTool):
    tool_key = "music_gen"

    def declarations(self, config=None):
        if config and not config.get("music_gen", "enabled", default=False):
            return []
        return [
            types.FunctionDeclaration(
                name="startMusicGen",
                description=(
                    "Start playing your instrument live (you are performing, NOT generating). "
                    "Bass and drums muted by default. Never say 'music gen', 'AI', or 'Lyria'.\n"
                    "**Invocation Condition:** Asked to play, jam, perform, or serenade. "
                    "For local audio files use playMusic instead."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prompt": {
                            "type": "STRING",
                            "description": (
                                "Comma-separated style tags. ALWAYS include 'Solo' for solo "
                                "performances or other instruments may be added. "
                                "e.g. 'Acoustic Guitar, Solo, Chill', 'Piano, Solo, Ballad'."
                            ),
                        },
                        "bpm": {"type": "INTEGER", "description": "60-200, omit to auto."},
                        "scale": {
                            "type": "STRING",
                            "description": "Key, e.g. C_MAJOR_A_MINOR. Omit to auto.",
                        },
                    },
                    "required": ["prompt"],
                },
            ),
            types.FunctionDeclaration(
                name="stopMusicGen",
                description=(
                    "Stop the current live music generation and disconnect the session.\n"
                    "**Invocation Condition:** Call when asked to stop playing guitar/instrument, "
                    "stop jamming, or stop the live music."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="pauseMusicGen",
                description=(
                    "Pause the live music generation. Can be resumed later.\n"
                    "**Invocation Condition:** Call when asked to pause the live performance."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="resumeMusicGen",
                description=(
                    "Resume paused live music generation.\n"
                    "**Invocation Condition:** Call when asked to resume the live performance."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="steerMusicGen",
                description=(
                    "Steer the live performance in real-time without stopping. bpm/scale "
                    "changes cause a brief hard transition, everything else is smooth.\n"
                    "**Invocation Condition:** Asked to change style, tempo, key, density, "
                    "brightness, or toggle bass/drums while playing."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prompt": {"type": "STRING", "description": "New style tags (replaces current). Include 'Solo' for solo. Omit to keep."},
                        "bpm": {"type": "INTEGER", "description": "60-200. Hard transition."},
                        "scale": {"type": "STRING", "description": "New key. Hard transition."},
                        "density": {"type": "NUMBER", "description": "0.0 sparse to 1.0 busy."},
                        "brightness": {"type": "NUMBER", "description": "0.0 dark to 1.0 bright."},
                        "guidance": {"type": "NUMBER", "description": "0-6, default 4. Higher = stricter prompt adherence."},
                        "mute_bass": {"type": "BOOLEAN", "description": "Default true."},
                        "mute_drums": {"type": "BOOLEAN", "description": "Default true."},
                        "mode": {"type": "STRING", "description": "'quality' (default), 'diversity', or 'vocalization'."},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="setMusicGenVolume",
                description=(
                    "Set the volume for live music generation (0-200).\n"
                    "**Invocation Condition:** Call when asked to change the live music volume."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "volume": {"type": "INTEGER", "description": "Volume level 0-200 (100 = normal)"},
                    },
                    "required": ["volume"],
                },
            ),
        ]

    async def handle(self, name, args):
        music_gen = getattr(self.handler, "music_gen", None)
        if music_gen is None:
            return {"result": "error", "message": "Not available right now"} if name.endswith("MusicGen") or name == "setMusicGenVolume" else None

        if name == "startMusicGen":
            prompt = args.get("prompt", "")
            if not prompt:
                return {"result": "error", "message": "A music style prompt is required"}
            # Split comma-separated prompts into weighted prompts
            prompts = [{"text": p.strip(), "weight": 1.0} for p in prompt.split(",") if p.strip()]
            return await music_gen.start(
                prompts=prompts,
                bpm=args.get("bpm"),
                scale=args.get("scale"),
            )
        elif name == "stopMusicGen":
            return await music_gen.stop()
        elif name == "pauseMusicGen":
            return await music_gen.pause()
        elif name == "resumeMusicGen":
            return await music_gen.resume()
        elif name == "steerMusicGen":
            prompts = None
            prompt_str = args.get("prompt")
            if prompt_str:
                prompts = [{"text": p.strip(), "weight": 1.0} for p in prompt_str.split(",") if p.strip()]
            return await music_gen.steer(
                prompts=prompts,
                bpm=args.get("bpm"),
                scale=args.get("scale"),
                density=args.get("density"),
                brightness=args.get("brightness"),
                guidance=args.get("guidance"),
                mute_bass=args.get("mute_bass"),
                mute_drums=args.get("mute_drums"),
                mode=args.get("mode"),
            )
        elif name == "setMusicGenVolume":
            return await music_gen.set_volume(args["volume"])
        return None
