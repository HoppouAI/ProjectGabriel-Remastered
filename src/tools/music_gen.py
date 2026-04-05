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

    def declarations(self, config=None):
        if config and not config.get("music_gen", "enabled", default=False):
            return []
        return [
            types.FunctionDeclaration(
                name="startMusicGen",
                description=(
                    "Start generating real-time instrumental music using Lyria RealTime AI. "
                    "You are playing your guitar/instrument live - describe the style/mood/genre "
                    "you want to play. Bass and drums are muted by default so it sounds like "
                    "a solo performance.\n"
                    "**Invocation Condition:** Call when someone asks you to play guitar, jam, "
                    "perform music, play an instrument, or generate live music. NOT for playing "
                    "local music files (use playMusic for that)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prompt": {
                            "type": "STRING",
                            "description": (
                                "Music style description. Can combine multiple styles separated by commas. "
                                "IMPORTANT: For solo instrument performances (no backing band), ALWAYS include "
                                "'Solo' as one of the tags - e.g. 'Acoustic Guitar, Solo, Chill' not just "
                                "'Acoustic Guitar, Chill'. The 'Solo' tag tells the AI to generate a solo "
                                "performance without other instruments. "
                                "Examples: 'Acoustic Guitar, Solo', 'Flamenco Guitar, Solo, Live Performance', "
                                "'Blues Rock Guitar, Solo', 'Classical Guitar, Solo, Dreamy', "
                                "'Indian Classical Sitar, Solo', 'Jazz Fusion Guitar, Solo, Smooth', "
                                "'Piano, Solo, Ballad', 'Violin, Solo, Emotional'. "
                                "Without 'Solo', other instruments may be added automatically."
                            ),
                        },
                        "bpm": {
                            "type": "INTEGER",
                            "description": "Beats per minute (60-200). Leave empty to let the AI decide based on the style.",
                        },
                        "scale": {
                            "type": "STRING",
                            "description": (
                                "Musical scale/key. Options: " + ", ".join(SCALE_OPTIONS)
                                + ". Leave empty to let the AI decide."
                            ),
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
                    "Steer the live music generation in real-time. Change the style, mood, "
                    "instruments, tempo, key, density, brightness, or toggle bass/drums. "
                    "Changes apply smoothly without stopping playback. For bpm/scale changes, "
                    "a brief hard transition occurs as the model resets context.\n"
                    "**Invocation Condition:** Call when asked to change the music style, "
                    "speed up/slow down, change key, add/remove bass or drums, make it "
                    "brighter/darker, busier/sparser, or shift the vibe while playing."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prompt": {"type": "STRING", "description": "New music style description (replaces current). Include 'Solo' for solo performances. Omit to keep current style."},
                        "bpm": {"type": "INTEGER", "description": "New BPM (60-200). Causes a hard transition."},
                        "scale": {"type": "STRING", "description": "New musical scale. Causes a hard transition."},
                        "density": {"type": "NUMBER", "description": "Note density 0.0 (sparse) to 1.0 (busy)"},
                        "brightness": {"type": "NUMBER", "description": "Tonal brightness 0.0 (dark) to 1.0 (bright)"},
                        "guidance": {"type": "NUMBER", "description": "Prompt adherence 0.0-6.0 (default 4.0). Higher = follows prompts more strictly."},
                        "mute_bass": {"type": "BOOLEAN", "description": "Mute bass line (default true)"},
                        "mute_drums": {"type": "BOOLEAN", "description": "Mute drums (default true)"},
                        "mode": {"type": "STRING", "description": "Generation mode: 'quality' (default), 'diversity', or 'vocalization'"},
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
