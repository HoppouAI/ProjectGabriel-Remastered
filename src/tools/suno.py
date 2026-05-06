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


STYLE_FORMAT_DOC = (
    "Style is a free-form prose description of how the song should sound. It "
    "drives Suno's instrument choice, mix, vibe, and vocal delivery. You can "
    "use any instruments, any genre fusion, anything. Be SEMI-DETAILED -- one "
    "well-packed paragraph (roughly 3 to 7 sentences, ~300-900 chars). Cover:\n"
    "  * Genre / fusion (e.g. 'Indian Trap and Desi Hip-Hop fusion')\n"
    "  * Specific instruments and their character (e.g. 'menacing distorted "
    "808 sub-bass', 'thundering Punjabi Dhol', 'sampled Harmonium looped')\n"
    "  * Rhythm and percussion details (e.g. 'rapid-fire rattling trap hi-hats')\n"
    "  * Melody / harmony feel (key, mode, mood)\n"
    "  * Vocal style, accent, delivery (e.g. 'aggressive rhythmic rap with a "
    "distinct Indian accent and punchy flow')\n"
    "  * Production polish, energy, atmosphere (e.g. 'polished, loud, bass-heavy, "
    "high energy, underground, intimidating')\n\n"
    "Example:\n"
    "\"Hard-hitting Indian Trap and Desi Hip-Hop fusion. The track is driven by "
    "a menacing, distorted 808 sub-bass that shakes the floor, layered over "
    "thundering traditional Punjabi Dhol percussion for a syncopated bounce. "
    "The rhythm features rapid-fire, rattling trap hi-hats and a sharp, snapping "
    "snare. A dark, minor-key melody is played by a sampled Harmonium or Sitar, "
    "looped to create a hypnotic and tense atmosphere. The vocals are delivered "
    "in an aggressive, rhythmic rap style with a distinct Indian accent and "
    "punchy flow. The production is polished, loud, and bass-heavy, blending "
    "raw folk roots with modern street grit. High energy, underground, and "
    "intimidating.\"\n\n"
    "Hard cap is 1000 characters. If you omit style, whatever was last set in "
    "the operator's Suno tab is reused."
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
                    "Generate a new full song with Suno using your own original lyrics "
                    "and style description, then stream it back live as it's being made. "
                    "The tool returns instantly with status 'submitted' -- audio actually "
                    "starts playing about 5-10 seconds later as Suno warms up. The chatbox "
                    "will show 'Generating song...' until then. Finished songs are "
                    "automatically saved to the local music library and can be replayed "
                    "later with playMusic.\n\n"
                    "**CRITICAL: DO NOT speak, sing, or recite the lyrics out loud.** "
                    "The lyrics go ONLY into the `lyrics` parameter of this function call. "
                    "Suno's voice will sing them. If you say the lyrics yourself you ruin "
                    "the surprise and waste your turn. Only briefly announce that you're "
                    "making a song (one short sentence max), then call the function and "
                    "STOP TALKING. Don't read back what you wrote, don't preview a verse, "
                    "don't hum it -- just call the tool.\n\n"
                    "While a Suno song is playing your spoken voice is muted (the music "
                    "takes over the audio output) but you can still hear the people around "
                    "you. Announce the song in one casual line like 'okay let me cook one "
                    "up real quick' and then SHUT UP and let the music play. Don't keep "
                    "talking over it. Don't say 'Suno' or 'AI generated' out loud, just "
                    "treat it like you're performing.\n\n"
                    "**Make the song long.** Each call costs a generation credit, so "
                    "submit a full-length song (target 2+ minutes / ~1500-2800 chars of "
                    "lyrics, with proper Verse/Chorus/Bridge structure). Short stub songs "
                    "waste credits.\n\n"
                    f"**Style format:**\n{STYLE_FORMAT_DOC}\n\n"
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
                        "style": {
                            "type": "STRING",
                            "description": (
                                "Semi-detailed prose description of the song's sound: "
                                "genre/fusion, specific instruments and their character, "
                                "rhythm, melody/key/mood, vocal style and accent, and "
                                "overall production feel. Up to 1000 chars. See description "
                                "for the exact format and an example."
                            ),
                        },
                    },
                    "required": ["lyrics", "style"],
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
            types.FunctionDeclaration(
                name="replayLastSong",
                description=(
                    "Replay the last Suno song you generated this session, without spending "
                    "another generation credit. Streams from the same temporary URL the "
                    "bridge gave us (these stay warm for around 5 to 10 minutes after "
                    "generation). If the URL has expired you'll get an error and need to "
                    "generate a fresh song instead. Stops anything currently playing first.\n"
                    "**Invocation Condition:** Call when asked to play that song again, "
                    "replay it, do it one more time, etc. Do NOT call this for songs from "
                    "the local music library, use playMusic for those."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="playOtherSong",
                description=(
                    "Suno actually generates 2 versions of every song; only the first one "
                    "auto-plays and gets saved. This tool plays the OTHER version (the "
                    "alternate take) of your most recent generation. No generation credit is "
                    "spent. The alternate take is NOT saved to the music library, so this is "
                    "the only way to hear it. Stops anything currently playing first.\n\n"
                    "If the URL has expired (after roughly 5-10 minutes) or suno only gave "
                    "us one version this time, you'll get an error.\n"
                    "**Invocation Condition:** Call when asked to hear the other version, "
                    "the alternate take, the second one, the B-side, etc."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        suno = getattr(self.handler, "suno", None)
        if name == "generateSong":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.generate(args.get("lyrics", ""), args.get("style"))
        if name == "stopSong":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.stop()
        if name == "replayLastSong":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.replay("last")
        if name == "playOtherSong":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.replay("other")
        return None
