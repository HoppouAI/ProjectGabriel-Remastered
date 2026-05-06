import logging

from google.genai import types

from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


LYRICS_FORMAT_DOC = (
    "Lyrics use Suno's section tag format. Each section starts with a tag in "
    "square brackets on its own line, followed by the actual sung lines for "
    "that section. Tags Suno understands well: [Intro], [Verse], [Verse 1], "
    "[Verse 2], [Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Hook], "
    "[Refrain], [Breakdown], [Drop], [Solo], [Instrumental], [Outro]. Do NOT "
    "use [End], [Fade], [Stop], [Silence], or other meta tags -- Suno doesn't "
    "need them and they tend to confuse it. The song just ends when the last "
    "[Outro] block ends.\n\n"
    "Other tags to AVOID: anything describing instruments or production "
    "(`[guitar solo]`, `[heavy drums]`, `[808 drop]`), stage directions "
    "(`[whispered]`, `[shouting]`, `[laughs]`), structural meta (`[End]`, "
    "`[Fade out]`, `[Repeat 2x]`), and parenthetical ad-libs longer than a "
    "couple words. All of that belongs in the `style` parameter, not in the "
    "lyrics. Inside the lines themselves you CAN use short ad-libs in "
    "parentheses for backing vocals (like `(yeah)`, `(uh)`, `(one more time)`) "
    "but use them sparingly.\n\n"
    "**Length matters A LOT.** Each generation costs real credits, so DO NOT "
    "submit short songs unless the user explicitly asks for a short one. "
    "Default target: a FULL-LENGTH song, 3 to 5 minutes of sung material. In "
    "practice that means a real, fleshed-out song structure: intro, three or "
    "four verses, a chorus that repeats three or four times, a pre-chorus, a "
    "bridge, maybe a second bridge or breakdown, and an outro. Each verse "
    "should be 6 to 12 lines, not just 4. Around 3000 to 5000 characters of "
    "lyrics is the sweet spot for a proper full track. Anything under ~1500 "
    "chars is a stub that wastes a credit -- only do that if the user "
    "specifically asked for a short song, jingle, or skit. Hard cap is 6000 "
    "chars.\n\n"
    "**Structure rules of thumb:**\n"
    "  * Intro is optional and short (2-4 lines) or skippable entirely.\n"
    "  * Verses tell/develop the story. Each verse should advance, not "
    "restate. 6-12 lines each.\n"
    "  * Pre-Chorus (optional) is 2-4 tension-building lines that lead into "
    "the chorus. Lyrically it sets up the hook.\n"
    "  * Chorus is the hook -- the most memorable, most repeated section. "
    "4-6 lines, sing-alongable. Usually identical every time it returns "
    "(small variations are fine on the last repeat).\n"
    "  * Bridge changes perspective, key feel, or energy. Use it once, late "
    "in the song. 4-8 lines.\n"
    "  * Outro winds it down, 2-6 lines. Don't tag [End] -- just stop "
    "writing.\n"
    "  * Do NOT pad with `la la la`, `yeah yeah yeah` filler lines or by "
    "repeating the same line over and over. Length comes from real lyrical "
    "content.\n\n"
    "Example skeleton (expand each section heavily, do NOT just copy this):\n"
    "[Intro]\n"
    "two to four soft opening lines that set the mood\n"
    "[Verse 1]\n"
    "eight to twelve lines telling the first part of the story\n"
    "...\n"
    "[Pre-Chorus]\n"
    "two to four lines that build tension into the hook\n"
    "[Chorus]\n"
    "four to six catchy lines, the hook of the song\n"
    "...\n"
    "[Verse 2]\n"
    "eight to twelve more lines that develop the story\n"
    "...\n"
    "[Pre-Chorus]\n"
    "(repeat or vary the pre-chorus)\n"
    "[Chorus]\n"
    "(repeat the chorus block)\n"
    "[Verse 3]\n"
    "eight to twelve lines that take the story further or twist it\n"
    "[Bridge]\n"
    "four to eight lines that shift the mood, often a key change feel\n"
    "[Chorus]\n"
    "(repeat the chorus, sometimes with an ad-lib variation)\n"
    "[Outro]\n"
    "four to eight closing lines that wind the song down\n\n"
    "Avoid stage directions, narration, or anything that isn't actually sung. "
    "You ALSO supply the music/genre/instruments/vocal direction separately via "
    "the `style` parameter -- match your lyrics to it. If the style says rap, "
    "write dense rhymed bars; if it says ballad, write emotional sparser lines; "
    "if it says metal, write aggressive hooks; etc. The two parameters work "
    "together, write them as one cohesive song."
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
    "Hard cap is 1000 characters. The `style` parameter is REQUIRED -- you "
    "must supply both lyrics and style on every call so the song is fully "
    "yours."
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
                    "submit a FULL-LENGTH song by default (target 3 to 5 minutes / "
                    "~3000-5000 chars of lyrics, with a real Verse/Pre-Chorus/Chorus/"
                    "Bridge/Verse/Chorus/Bridge/Outro structure and 8-12 line verses). "
                    "Only make a short song (under ~1500 chars) if the user explicitly "
                    "asks for something short like a jingle, skit, intro stinger, or "
                    "30-second snippet. Stub songs waste credits.\n\n"
                    "**Songcraft -- write like an actual artist, not a chatbot.** The "
                    "lyrics need to make sense, follow a coherent theme from start to "
                    "finish, and have real bars. Aim for fire lines: punchlines, vivid "
                    "imagery, internal rhymes, multisyllabic rhymes, wordplay, double "
                    "meanings. Each verse should advance a story or angle, not restate "
                    "the same idea. The chorus should be the catchiest, most repeatable "
                    "part -- a clear hook the listener can sing along to. Pre-chorus "
                    "builds tension into the chorus. Bridge changes perspective, key "
                    "feel, or energy. Outro lands the song. NO random word salad, NO "
                    "filler lines like 'yeah yeah', NO repeating the same line over and "
                    "over to pad length. If the user wants funny, lean into clever "
                    "comedy, situational humor, callbacks and absurdism, NOT just dumb "
                    "lazy jokes -- think Weird Al, MF DOOM punchlines, Bo Burnham, "
                    "Lonely Island. Funny songs still need craft. Match rhyme density "
                    "and flow to the genre in `style` (rap = dense rhymes and flow, "
                    "ballad = sparser more emotional, pop = simple memorable hook).\n\n"
                    "**Pull context BEFORE writing.** If the song is about a specific "
                    "person, place, event, or inside joke, you need to actually know "
                    "stuff about them, otherwise the song is generic trash. Steps:\n"
                    "  1. If a name comes up that you might have notes on (a regular, a "
                    "friend, someone mentioned before), call `searchMemories` or "
                    "`recallMemories` FIRST and weave real details in. Don't fabricate.\n"
                    "  2. If the song is about people currently in the VRChat instance, "
                    "look at who's actually visible. Yellow nameplate = your VRChat "
                    "friend (use `getFriendInfo` if you want their bio/status). Known "
                    "regulars = check memory. Use their actual usernames and known "
                    "traits in the bars.\n"
                    "  3. If you have NOTHING -- no memories, person isn't a friend, "
                    "you don't know them -- DO NOT pad the song with vague flattery or "
                    "made-up facts. Either ask one quick question first, or just write "
                    "a song that doesn't rely on personal details. Don't drag empty "
                    "shoutouts on for whole verses.\n"
                    "Skip these steps for songs that aren't about specific people "
                    "(generic vibe songs, joke songs about objects/concepts, etc).\n\n"
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
                                "Full song lyrics with [Section] tags. Default to a full "
                                "3-5 minute song (~3000-5000 chars) unless user asks for "
                                "something short. See description for exact format. Hard "
                                "cap ~5000 chars."
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
            types.FunctionDeclaration(
                name="searchSongLibrary",
                description=(
                    "Search the operator's existing Suno song library (their saved playlist) "
                    "for songs you can cover/parody. Returns a list of `{id, title, styles, "
                    "has_lyrics}` entries. Use this to discover what's available BEFORE "
                    "calling coverSong. The library is the operator's own catalog of past "
                    "Suno generations and is the ONLY pool of songs that can be covered -- "
                    "you can't cover arbitrary commercial tracks.\n\n"
                    "Pass an optional `query` substring to filter by title (case-insensitive). "
                    "Omit `query` to list everything (capped at 25 entries). The full lyrics "
                    "are NOT included in this response to keep it small -- use "
                    "`getSongLyrics` on the specific id once you've picked one.\n"
                    "**Invocation Condition:** Call before `coverSong` to find a source "
                    "song. Also call when someone asks 'what songs can you parody', "
                    "'what's in your library', 'do you have a cover of X', etc."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "Optional case-insensitive substring to filter by title.",
                        },
                    },
                },
            ),
            types.FunctionDeclaration(
                name="getSongLyrics",
                description=(
                    "Fetch the full lyrics of one specific song from the operator's library "
                    "by its id. Use this AFTER `searchSongLibrary` once you've picked which "
                    "song to cover/parody, so you can rewrite the lyrics line-for-line and "
                    "keep the same structure/meter/rhyme scheme. Pass the id verbatim from "
                    "`searchSongLibrary`'s response.\n"
                    "**Invocation Condition:** Call right before `coverSong` whenever you "
                    "intend to supply your own parody lyrics. Skip if you're doing a "
                    "no-override cover (just remixing the same song)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "id": {
                            "type": "STRING",
                            "description": "Source song uuid from searchSongLibrary.",
                        },
                    },
                    "required": ["id"],
                },
            ),
            types.FunctionDeclaration(
                name="coverSong",
                description=(
                    "Cover or parody an existing Suno song from the operator's library. "
                    "The bridge tells Suno to remix the source clip, optionally swapping "
                    "in your own lyrics and/or style. The melody/voice/instrumentation are "
                    "conditioned on the original clip's audio, so the new version will "
                    "sound musically similar to the source -- this is the right tool for "
                    "PARODIES (same melody, new words) and REMIXES (same words, new vibe). "
                    "Costs a generation credit, same as generateSong.\n\n"
                    "**Workflow:** First call `searchSongLibrary` to find a source song. "
                    "Then call `getSongLyrics` to grab the original lyrics. Then write "
                    "parody lyrics that match the original's structure (same number of "
                    "[Verse]/[Chorus] blocks, similar line counts and syllable counts so "
                    "the words actually fit the melody). Finally call `coverSong` with the "
                    "source `id` and your `lyrics` (and optionally `style`).\n\n"
                    "All the same rules as `generateSong` apply: don't recite lyrics "
                    "out loud, announce briefly then SHUT UP, finished cover gets saved "
                    "to the local music library, etc.\n\n"
                    "If you call this with no `lyrics` or `style` overrides, Suno just "
                    "remixes the source as-is (different take, same words, same vibe).\n"
                    f"**Lyrics format (when overriding):**\n{LYRICS_FORMAT_DOC}\n\n"
                    f"**Style format (when overriding):**\n{STYLE_FORMAT_DOC}\n"
                    "**Invocation Condition:** Call when asked to parody, cover, remix, "
                    "redo, or rewrite an existing song from the library. Do NOT use this "
                    "for brand new original songs (use `generateSong`)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "id": {
                            "type": "STRING",
                            "description": "Source song uuid from searchSongLibrary.",
                        },
                        "lyrics": {
                            "type": "STRING",
                            "description": (
                                "Optional parody lyrics. Match the source's structure and "
                                "line count so the new words fit the melody. Omit to keep "
                                "the original lyrics. Same format as generateSong."
                            ),
                        },
                        "style": {
                            "type": "STRING",
                            "description": (
                                "Optional new style description. Omit to keep the source "
                                "song's style. Same format as generateSong."
                            ),
                        },
                    },
                    "required": ["id"],
                },
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
        if name == "searchSongLibrary":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.search_library(args.get("query"))
        if name == "getSongLyrics":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.get_lyrics(args.get("id", ""))
        if name == "coverSong":
            if suno is None:
                return {"result": "error", "message": "Suno integration is not enabled."}
            return await suno.cover(args.get("id", ""), args.get("lyrics"), args.get("style"))
        return None
