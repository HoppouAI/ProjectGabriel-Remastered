"""Chatbox text formatting + now-playing display helpers.

Mixed into GeminiLiveSession. All methods are pure-ish - only touch
self.audio for the current lyric and self.config for divider settings,
no side effects beyond returning a string.
"""

import re
import time


class ChatboxFormattersMixin:
    @staticmethod
    def _strip_audio_tags_for_chatbox(text: str) -> str:
        """Remove inline expressive audio tags (for example [whispers]) from chatbox text only."""
        if not text:
            return ""
        cleaned = re.sub(r"\[(?:[A-Za-z][A-Za-z\s,'-]{0,40})\]", " ", text)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        cleaned = ChatboxFormattersMixin._convert_markdown_italics_to_unicode(cleaned)
        return cleaned

    @staticmethod
    def _convert_markdown_italics_to_unicode(text: str) -> str:
        """Convert *text* markdown italics to Unicode small caps for VRChat chatbox display."""
        if not text or "*" not in text:
            return text

        small_caps_map = {
            "A": "ᴀ", "B": "ʙ", "C": "ᴄ", "D": "ᴅ", "E": "ᴇ", "F": "ꜰ", "G": "ɢ", "H": "ʜ",
            "I": "ɪ", "J": "ᴊ", "K": "ᴋ", "L": "ʟ", "M": "ᴍ", "N": "ɴ", "O": "ᴏ", "P": "ᴘ",
            "Q": "Q", "R": "ʀ", "S": "ꜱ", "T": "ᴛ", "U": "ᴜ", "V": "ᴠ", "W": "ᴡ", "X": "X",
            "Y": "ʏ", "Z": "ᴢ",
            "a": "ᴀ", "b": "ʙ", "c": "ᴄ", "d": "ᴅ", "e": "ᴇ", "f": "ꜰ", "g": "ɢ", "h": "ʜ",
            "i": "ɪ", "j": "ᴊ", "k": "ᴋ", "l": "ʟ", "m": "ᴍ", "n": "ɴ", "o": "ᴏ", "p": "ᴘ",
            "q": "q", "r": "ʀ", "s": "ꜱ", "t": "ᴛ", "u": "ᴜ", "v": "ᴠ", "w": "ᴡ", "x": "x",
            "y": "ʏ", "z": "ᴢ",
        }

        def convert_to_small_caps(match):
            content = match.group(1)
            return "".join(small_caps_map.get(ch, ch) for ch in content)

        return re.sub(r"\*([^*]+)\*", convert_to_small_caps, text)

    @staticmethod
    def _normalize_song_name(name: str) -> str:
        """Clean up a filename-based song name for display."""
        name = name.replace("_", " ").replace("-", " ")
        name = re.sub(r"\s+", " ", name).strip()
        return name.title()

    def _format_now_playing(self, progress_info: dict) -> str:
        """Format Now Playing display for chatbox."""
        name = self._normalize_song_name(progress_info["song_name"])
        position = progress_info["position"]
        duration = progress_info["duration"]
        progress = progress_info["progress"]

        pos_min, pos_sec = divmod(int(position), 60)
        dur_min, dur_sec = divmod(int(duration), 60)
        time_str = f"{pos_min}:{pos_sec:02d} / {dur_min}:{dur_sec:02d}"

        bar_width = 14
        exact = progress * bar_width
        filled = int(exact)
        fraction = exact - filled
        if filled >= bar_width:
            bar = "\u2588" * bar_width
        else:
            if fraction < 0.25:
                transition = "\u2591"
            elif fraction < 0.5:
                transition = "\u2592"
            elif fraction < 0.75:
                transition = "\u2593"
            else:
                transition = "\u2588"
            bar = "\u2588" * filled + transition + "\u2591" * (bar_width - filled - 1)

        lyric = self.audio.get_current_lyric()

        lines = []
        if lyric:
            max_lyric = 100
            if len(lyric) > max_lyric:
                lyric = lyric[:max_lyric - 3] + "..."
            lines.append("LYRICS")
            lines.append(lyric)
            lines.append("────────────")

        max_name = 100
        if len(name) > max_name:
            name = name[:max_name - 3] + "..."

        lines.append(name)
        lines.append(bar)
        lines.append(time_str)

        return "\n".join(lines)

    def _format_music_gen_display(self, music_gen) -> str:
        """Format Lyria music gen display for chatbox."""
        elapsed = int(music_gen.elapsed)
        prompts = music_gen.current_prompts

        tags = " ".join(f"[{p['text']}]" for p in prompts)

        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        if hours > 0:
            time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        elif minutes > 0:
            time_str = f"{minutes}:{seconds:02d}"
        else:
            time_str = f"0:{seconds:02d}"

        divider_char = self.config.get("vrchat", "idle_chatbox", "divider", default="\u2500")
        divider_length = self.config.get("vrchat", "idle_chatbox", "divider_length", default=14)
        divider = str(divider_char) * int(divider_length)

        lines = []
        if music_gen.is_paused:
            lines.append("\u23f8 PAUSED")
        else:
            lines.append("\u266b Live Music")
        if tags:
            lines.append(tags)
        lines.append(divider)
        lines.append(time_str)

        text = "\n".join(lines)
        if len(text) > 144:
            max_tags = 144 - len(lines[0]) - len(divider) - len(time_str) - 3
            if max_tags > 3:
                tags = tags[:max_tags - 3] + "..."
            lines = [lines[0], tags, divider, time_str]
            text = "\n".join(lines)
            if len(text) > 144:
                text = text[:144]
        return text

    def _format_web_search_display(self, active: dict) -> str:
        """Spinner + counting up timer for an in flight web search/read."""
        elapsed = max(0, int(time.time() - active.get("started_at", time.time())))
        kind = active.get("kind", "search")
        label = (active.get("label") or "").strip()
        # rotating braille spinner so theres visible motion at 1Hz refresh
        spinner_frames = "\u280B\u2819\u2839\u2838\u283C\u2834\u2826\u2827\u2807\u280F"
        spinner = spinner_frames[elapsed % len(spinner_frames)]

        if kind == "read":
            header = f"{spinner} Reading webpage"
            # urls can be huge, trim the middle
            if len(label) > 60:
                label = label[:30] + "..." + label[-25:]
        else:
            header = f"{spinner} Searching the web"

        divider_char = self.config.get("vrchat", "idle_chatbox", "divider", default="\u2500")
        divider_length = self.config.get("vrchat", "idle_chatbox", "divider_length", default=14)
        divider = str(divider_char) * int(divider_length)

        mins, secs = divmod(elapsed, 60)
        if mins:
            time_str = f"{mins}:{secs:02d}"
        else:
            time_str = f"{secs}s"

        lines = [header]
        if label:
            if len(label) > 80:
                label = label[:77] + "..."
            lines.append(label)
        lines.append(divider)
        lines.append(time_str)

        text = "\n".join(lines)
        if len(text) > 144:
            text = text[:144]
        return text
