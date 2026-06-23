"""Dynamic tool selection for the local backend.

Sending all ~100 tool declarations to LM Studio every turn burns a huge chunk
of the prompt (around 13k tokens here) and the model only ever needs a handful
of them per turn. This does progressive disclosure instead, the same pattern
bigger agents use: keep a small always-on core, then surface the rest on demand.

Two mechanisms, both cheap and local (no embeddings, no extra latency in the
common case):

  preselect   a lexical relevance scan over the user's utterance pulls in the
              top-N matching tools before the first LLM call, so most turns
              already have what they need.
  findTools   a meta tool the model can call to load anything it doesn't see,
              like the agent's own tool search. matches get activated and show
              up (with full schemas) on the next step of the turn.

The scorer just splits tool names (camelCase aware) and descriptions into words
and counts overlap with the query, weighting name hits over description hits. A
tiny synonym map covers the obvious paraphrases. Crude, but tools have very
distinctive names so it works well, and findTools is the safety net for misses.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
    "it", "this", "that", "you", "your", "me", "my", "i", "can", "could",
    "would", "please", "do", "did", "does", "be", "am", "are", "was", "were",
    "get", "got", "set", "go", "going", "want", "need", "make", "let", "us",
    "we", "they", "them", "he", "she", "him", "her", "what", "which", "how",
    "when", "where", "who", "why", "if", "then", "so", "just", "now", "some",
    "any", "all", "out", "up", "down", "from", "by", "at", "as", "about",
}

# query word -> extra words to also match against tool name/description tokens.
# keeps the common paraphrases working without per-tool curation.
_SYNONYMS = {
    "song": ["music", "play"],
    "songs": ["music"],
    "track": ["music"],
    "tune": ["music"],
    "lofi": ["music"],
    "beat": ["music"],
    "remember": ["memory", "save"],
    "forget": ["memory", "delete"],
    "recall": ["memory", "search"],
    "note": ["memory"],
    "explore": ["wander"],
    "roam": ["wander"],
    "walk": ["move", "wander", "waypoint"],
    "come": ["follow", "move"],
    "stop": ["stop", "cancel"],
    "louder": ["volume"],
    "quieter": ["volume"],
    "mute": ["mic", "voice"],
    "google": ["web", "search"],
    "search": ["search", "web"],
    "website": ["web", "webpage"],
    "friend": ["friend", "social"],
    "dm": ["message", "social"],
    "message": ["message", "social"],
    "avatar": ["avatar"],
    "bigger": ["scale"],
    "smaller": ["scale"],
    "taller": ["scale"],
    "world": ["world"],
    "invite": ["invite"],
    "join": ["invite", "instance"],
    "jump": ["jump"],
    "crouch": ["crouch"],
    "crawl": ["crawl"],
    "sound": ["soundboard"],
    "sfx": ["soundboard"],
    "meme": ["soundboard"],
    "mood": ["personality", "emotion"],
    "personality": ["personality"],
    "discord": ["discord"],
    "time": ["time"],
}


def _words(text: str) -> list[str]:
    """Lowercase word tokens, camelCase aware (playMusic -> play, music)."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) >= 2 and w not in _STOP}


def _expand(query_words: set[str]) -> set[str]:
    out = set(query_words)
    for w in query_words:
        out.update(_SYNONYMS.get(w, ()))
    return out


_FIND_TOOLS = {
    "type": "function",
    "function": {
        "name": "findTools",
        "description": (
            "Load extra tools you don't currently have. Your visible tool list "
            "is only a relevant subset. Call this whenever you want to do "
            "something and don't see a tool for it (control VRChat movement, "
            "play music or soundboard clips, manage friends or DMs, search the "
            "web, navigate or set waypoints, change avatar, etc). Pass a short "
            "description of the action; the matching tools become callable on "
            "your next step. Don't tell the user a tool is missing, just call "
            "findTools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "what you want to do, eg 'play a song', 'follow the player', 'search worlds'",
                },
            },
            "required": ["query"],
        },
    },
}


class ToolGate:
    """Picks a small relevant slice of the full tool catalog per turn."""

    def __init__(self, all_tools: list[dict], core_names: list[str], max_dynamic: int = 8,
                 activated_cap: int = 16):
        self._by_name: dict[str, dict] = {}
        self._index: dict[str, tuple[set[str], set[str]]] = {}
        for t in all_tools:
            fn = t.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            self._by_name[name] = t
            self._index[name] = (_content_words(name), _content_words(fn.get("description", "")))
        self._core = [n for n in core_names if n in self._by_name]
        self._max = max(0, int(max_dynamic))
        self._activated_cap = max(0, int(activated_cap))

    @property
    def total(self) -> int:
        return len(self._by_name)

    def _score(self, qwords: set[str], name: str) -> int:
        name_toks, desc_toks = self._index[name]
        s = 0
        for w in qwords:
            if w in name_toks:
                s += 3
            elif w in desc_toks:
                s += 1
        return s

    def _rank(self, query: str, skip: set[str]) -> list[tuple[int, str]]:
        qwords = _expand(_content_words(query))
        if not qwords:
            return []
        scored = [
            (self._score(qwords, name), name)
            for name in self._by_name
            if name not in skip
        ]
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda p: (-p[0], p[1]))
        return scored

    def select(self, query: str, activated: list[str]) -> list[dict]:
        """Tools to send this turn: findTools + core + activated + top-N for
        the query. Order is stable so an unchanged set keeps LM Studio's prompt
        cache warm."""
        chosen: "OrderedDict[str, None]" = OrderedDict()
        for n in self._core:
            chosen[n] = None
        for n in activated:
            if n in self._by_name:
                chosen[n] = None
        for _, name in self._rank(query, set(chosen))[: self._max]:
            chosen[name] = None
        return [_FIND_TOOLS] + [self._by_name[n] for n in chosen]

    def search(self, query: str, top: int = 8) -> list[str]:
        """Names matching a findTools query, best first."""
        return [name for _, name in self._rank(query, set(self._core))[:top]]

    def cap_activated(self, activated: list[str]) -> list[str]:
        if self._activated_cap and len(activated) > self._activated_cap:
            return activated[-self._activated_cap:]
        return activated
