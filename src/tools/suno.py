import logging

from google.genai import types

from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


LYRICS_FORMAT_DOC = (
    "Lyrics use Suno's section tag format. Each section starts with a tag in "
    "square brackets on its own line, followed by the lines for that section. "
    "Common tags: [Intro], [Verse], [Verse 1], [Pre-Chorus], [Chorus], [Bridge], "
    "[Hook], [Drop], [Outro], [Instrumental], [Solo]. You can use [End] to mark "
    "the song's end. Keep lines short, song-like, and rhyming when natural.\n\n"
    "**Length matters.** Each generation costs credits, so DO NOT submit short "
    "songs. Aim for at least 2 minutes of sung material -- in practice that "
    "means a real song structure: intro, two or three verses, a chorus that "
    "repeats two or three times, a bridge, and an outro. Around 1500 to 2800 "
    "characters of lyrics is the sweet spot. Anything under ~800 chars is too "
    "short and wastes a credit. Hard cap is 3000 chars.\n\n"
    "Example skeleton (expand each section, do NOT just copy this):\n"
    "[Intro]\n"
    "soft opening line\n"
    "another opening line\n"
    "[Verse 1]\n"
    "four to eight lines telling the first part of the story\n"
    "...\n"
    "[Pre-Chorus]\n"
    "two lines that build tension\n"
    "[Chorus]\n"
    "four catchy lines, the hook of the song\n"
    "...\n"
    "[Verse 2]\n"
    "four to eight more lines that develop the story\n"
    "...\n"
    "[Chorus]\n"
    "(repeat the chorus block)\n"
    "[Bridge]\n"
    "two to four lines that shift the mood\n"
    "[Chorus]\n"
    "(repeat the chorus one more time)\n"
    "[Outro]\n"
    "two to four closing lines\n\n"
    "Avoid stage directions, narration, or anything that isn't actually sung. "
    "Style/genre/voice are set by the operator -- you only supply lyrics."
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
                    "then stream it back live as it's being made. The tool returns "
                    "instantly with status 'submitted' -- audio actually starts "
                    "playing about 5-10 seconds later as Suno warms up. The chatbox "
                    "will show 'Generating song...' until then.\n\n"
                    "While a Suno song is playing your spoken voice is muted (the music "
                    "takes over the audio output) but you can still hear the people around "
                    "you. Announce the song before calling this -- something casual like "
                    "'okay let me cook one up real quick' -- then SHUT UP and let the "
                    "music play. Don't keep talking over it. Don't say 'Suno' or 'AI "
                    "generated' out loud, just treat it like you're performing.\n\n"
                    "**Make the song long.** Each call costs a generation credit, so "
                    "submit a full-length song (target 2+ minutes / ~1500-2800 chars of "
                    "lyrics, with proper Verse/Chorus/Bridge structure). Short stub songs "
                    "waste credits.\n\n"
                    f"**Lyrics format:**\n{LYRICS_FORMAT_DOC}\n\n"
                    "**Rate limit:** Only one song every 30 seconds. If it fails with "
                    "rate_limited, wait it out and tell the user. If it fails with "
                    "already_generating, a song is still being made -- be patient. "
                    "Other errors usually mean the operator needs to check their setup -- "
                    "pass that on.\n\n"
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
