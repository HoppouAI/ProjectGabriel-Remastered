import logging

from google.genai import types

from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


LYRICS_FORMAT_DOC = (
    "Lyrics use Suno's section tag format. Each section starts with a tag in "
    "square brackets on its own line, followed by the lines for that section. "
    "Common tags: [Intro], [Verse], [Verse 1], [Pre-Chorus], [Chorus], [Bridge], "
    "[Hook], [Drop], [Outro], [Instrumental], [Solo]. You can use [End] to mark "
    "the song's end. Keep lines short, song-like, and rhyming when natural. "
    "Example:\n"
    "[Verse]\n"
    "walking through the neon rain\n"
    "city lights inside my brain\n"
    "[Chorus]\n"
    "we are the ones who never sleep\n"
    "promises we cant keep\n"
    "[Outro]\n"
    "fading out into the dawn\n"
    "Max length is roughly 3000 characters. Avoid stage directions, narration, "
    "or anything that isn't actually sung. Style/genre/voice are set in the suno "
    "you only supply lyrics."
)


@register_tool
class SunoTools(BaseTool):
    tool_key = "suno"

    def declarations(self, config=None):
        if not config or not config.get("suno", "enabled", default=False):
            return []
        return [
            types.FunctionDeclaration(
                name="generateSong",
                description=(
                    "Generate a new full song with Suno using your own original lyrics, "
                    "then stream it back live as it's being made. The song starts playing "
                    "within a few seconds and Suno fills it in as you listen.\n\n"
                    "While a Suno song is playing your spoken voice is muted (the music "
                    "takes over the audio output) but you can still hear the people around "
                    "you. Announce the song before calling this -- something casual like "
                    "'okay let me cook one up real quick'. Don't say 'Suno' or 'AI generated' "
                    "out loud, just treat it like you're performing.\n\n"
                    f"**Lyrics format:**\n{LYRICS_FORMAT_DOC}\n\n"
                    "**Rate limit:** Only one song every 30 seconds. If it fails with "
                    "rate_limited, wait it out and tell the user. Other errors usually "
                    "mean the operator needs to check their setup -- pass that on.\n\n"
                    "**Invocation Condition:** Call when someone asks you to write, make, "
                    "generate, or sing an original song with custom lyrics. Do NOT call "
                    "this for playing existing local music (use playMusic) or for live "
                    "instrument jamming (use startMusicGen)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "lyrics": {
                            "type": "STRING",
                            "description": (
                                "Full song lyrics with [Section] tags. See description for "
                                "exact format. Up to ~3000 chars."
                            ),
                        },
                    },
                    "required": ["lyrics"],
                },
            ),
            types.FunctionDeclaration(
                name="stopSong",
                description=(
                    "Immediately stop the currently playing Suno song. The chatbox music UI "
                    "clears and your voice un-mutes.\n"
                    "**Invocation Condition:** Call when asked to stop the song, kill the "
                    "music, shut up the music, etc. Only stops Suno songs -- use stopMusic "
                    "for local files and stopMusicGen for live instrument."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        suno = getattr(self.handler, "suno", None)
        if name == "generateSong":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.generate(args.get("lyrics", ""))
        if name == "stopSong":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.stop()
        return None
