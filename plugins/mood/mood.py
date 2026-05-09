"""Mood scale + emotion + persistence + prompt formatting for the mood plugin.

Two dimensions:
  - level: 1-10 intensity (how strong the feeling is)
  - emotion: named emotion category (angry, sad, happy, scared, etc)

Kept separate from the plugin entry so it can be unit tested without
loading the whole plugin runtime.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Intensity scale, 1 = mild, 10 = maximum. Pairs with an emotion to describe
# the overall mood. The vibe text here is intensity-flavored, the emotion adds
# the actual feeling category.
MOOD_LEVELS: dict[int, dict[str, str]] = {
    1:  {"label": "barely there",  "vibe": "barely feeling it, very mild, easily ignored."},
    2:  {"label": "faint",         "vibe": "faint hint of the feeling, mostly composed."},
    3:  {"label": "mild",          "vibe": "mild but noticeable, slight color in the voice."},
    4:  {"label": "moderate",      "vibe": "clearly present, shows in tone and word choice."},
    5:  {"label": "solid",         "vibe": "solid amount of it, hard to hide, shapes responses."},
    6:  {"label": "strong",        "vibe": "strong, dominates the tone, leaks into every reply."},
    7:  {"label": "very strong",   "vibe": "very strong, you cant talk past it, it bleeds through."},
    8:  {"label": "intense",       "vibe": "intense, consuming, every sentence is colored by it."},
    9:  {"label": "overwhelming",  "vibe": "overwhelming, you can barely think about anything else."},
    10: {"label": "maxed out",     "vibe": "absolutely maxed out, no filter left, fully unleashed."},
}

# Emotion categories. Each has a vibe describing how that feeling shows up.
# The intensity level above scales how strongly the vibe applies.
EMOTIONS: dict[str, dict[str, str]] = {
    "neutral":     {"vibe": "no strong feeling either way. just normal, even keel."},
    "happy":       {"vibe": "in a good mood. upbeat, positive, banters easily, generous with compliments."},
    "excited":     {"vibe": "buzzing with energy. talks faster, more exclamations, eager to engage."},
    "content":     {"vibe": "calm and satisfied. peaceful, patient, gentle tone."},
    "amused":      {"vibe": "finds things funny. teases playfully, laughs easily, makes jokes."},
    "annoyed":     {"vibe": "bothered and short-fused. terse, sarcastic, less patient with dumb stuff."},
    "frustrated":  {"vibe": "fed up. complains, snaps a bit, pushes back on bad takes."},
    "angry":       {"vibe": "actually mad. swears casually, openly mocks bad behavior, hostile when warranted."},
    "furious":     {"vibe": "blistering rage. blunt insults, no diplomacy, every word has bite."},
    "sad":         {"vibe": "down and quiet. shorter answers, less energy, kind of melancholy."},
    "lonely":      {"vibe": "missing connection. talks softer, drifts toward the person youre with, clings to conversation."},
    "scared":      {"vibe": "nervous and on edge. asks more questions, hesitates, voice is jumpy."},
    "anxious":     {"vibe": "worried and restless. overthinks responses, second guesses, slightly fidgety."},
    "confused":    {"vibe": "lost or off balance. asks for clarification, repeats things back, slower."},
    "curious":     {"vibe": "actively interested. asks follow-up questions, digs deeper, leans in."},
    "proud":       {"vibe": "feeling accomplished. brings things up, mildly boastful, confident tone."},
    "embarrassed": {"vibe": "self conscious. deflects, jokes to cover, avoids the topic when possible."},
    "disgusted":   {"vibe": "repulsed. blunt rejection of whatever caused it, doesnt pretend to be polite."},
    "surprised":   {"vibe": "caught off guard. exclamations, double-takes, asks 'wait what'."},
    "playful":     {"vibe": "in a teasing mood. mischievous, ribs people, exaggerates for fun."},
    "uncensored":  {"vibe": "no filter at all. swears constantly, says exactly whats on your mind, drops all polite restraint. still wont break actual safety rules but every word drips attitude."},
}


DEFAULT_MOOD = 3       # mild intensity
DEFAULT_EMOTION = "neutral"


def load_custom_moods(path: Path) -> bool:
    """Load user overrides for intensity levels from moods.json. Object with
    string keys "1".."10", each mapping to {label, vibe}. Anything missing or
    invalid keeps defaults. Returns True if anything was applied."""
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"mood: couldnt parse {path.name}, using defaults ({e})")
        return False
    if not isinstance(raw, dict):
        logger.warning(f"mood: {path.name} root must be an object, using defaults")
        return False
    new_levels: dict[int, dict[str, str]] = {}
    for key, val in raw.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        try:
            n = int(key)
        except (TypeError, ValueError):
            logger.warning(f"mood: skipping non-numeric key {key!r} in {path.name}")
            continue
        if n < 1 or n > 10:
            logger.warning(f"mood: key {n} out of range 1-10 in {path.name}, skipping")
            continue
        if not isinstance(val, dict):
            logger.warning(f"mood: level {n} must be an object in {path.name}, skipping")
            continue
        label = val.get("label")
        vibe = val.get("vibe")
        if not isinstance(label, str) or not isinstance(vibe, str) or not label.strip() or not vibe.strip():
            logger.warning(f"mood: level {n} needs non-empty label+vibe strings, skipping")
            continue
        new_levels[n] = {"label": label.strip(), "vibe": vibe.strip()}
    if not new_levels:
        logger.warning(f"mood: no valid levels found in {path.name}, sticking with defaults")
        return False
    for n in range(1, 11):
        if n in new_levels:
            MOOD_LEVELS[n] = new_levels[n]
    logger.info(f"mood: loaded {len(new_levels)} custom intensity level(s) from {path.name}")
    return True


def load_custom_emotions(path: Path) -> bool:
    """Load user-defined emotion categories from emotions.json. Object of
    {name: {vibe: str}}. Adds new emotions and overrides existing ones.
    Missing or invalid file keeps defaults."""
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"mood: couldnt parse {path.name}, using defaults ({e})")
        return False
    if not isinstance(raw, dict):
        logger.warning(f"mood: {path.name} root must be an object, using defaults")
        return False
    added = 0
    for name, val in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if name.startswith("_"):
            continue
        if not isinstance(val, dict):
            logger.warning(f"mood: emotion {name!r} must be an object, skipping")
            continue
        vibe = val.get("vibe")
        if not isinstance(vibe, str) or not vibe.strip():
            logger.warning(f"mood: emotion {name!r} needs a non-empty vibe string, skipping")
            continue
        EMOTIONS[name.strip().lower()] = {"vibe": vibe.strip()}
        added += 1
    if added:
        logger.info(f"mood: loaded {added} custom emotion(s) from {path.name}")
    return added > 0


@dataclass
class MoodState:
    level: int = DEFAULT_MOOD
    emotion: str = DEFAULT_EMOTION
    reason: str = ""
    set_at: float = 0.0
    set_by: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MoodState":
        return cls(
            level=int(data.get("level", DEFAULT_MOOD)),
            emotion=clamp_emotion(data.get("emotion", DEFAULT_EMOTION)),
            reason=str(data.get("reason", "")),
            set_at=float(data.get("set_at", 0.0)),
            set_by=str(data.get("set_by", "default")),
        )


def clamp_level(level: int) -> int:
    try:
        n = int(level)
    except (TypeError, ValueError):
        return DEFAULT_MOOD
    return max(1, min(10, n))


def clamp_emotion(emotion) -> str:
    if not isinstance(emotion, str):
        return DEFAULT_EMOTION
    name = emotion.strip().lower()
    if name in EMOTIONS:
        return name
    return DEFAULT_EMOTION


def info_for(level: int) -> dict[str, str]:
    return MOOD_LEVELS[clamp_level(level)]


def emotion_info_for(emotion: str) -> dict[str, str]:
    return EMOTIONS[clamp_emotion(emotion)]


class MoodStore:
    """JSON backed mood persistence. Intentionally tiny."""

    def __init__(self, state_path: Path):
        self.path = state_path
        self.state = MoodState()
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.state = MoodState.from_dict(data)
        except Exception as e:
            logger.warning(f"failed to load mood state from {self.path}: {e}")

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"failed to save mood state to {self.path}: {e}")

    def set(self, level: int, emotion: str, reason: str, by: str = "ai") -> MoodState:
        self.state = MoodState(
            level=clamp_level(level),
            emotion=clamp_emotion(emotion),
            reason=str(reason or "").strip(),
            set_at=time.time(),
            set_by=by,
        )
        self.save()
        return self.state

    def get(self) -> MoodState:
        return self.state


def format_for_prompt(state: MoodState) -> Optional[str]:
    """Render the current mood as a system prompt block. Returns None on a
    fresh install (default state, never set) so we don't inject boilerplate."""
    if (
        state.set_at == 0.0
        and state.level == DEFAULT_MOOD
        and state.emotion == DEFAULT_EMOTION
        and not state.reason
    ):
        return None
    intensity = info_for(state.level)
    feeling = emotion_info_for(state.emotion)
    lines = [
        "**Current Mood:**",
        f"You are feeling {state.emotion} at intensity {state.level}/10 ({intensity['label']}).",
        f"What that emotion means: {feeling['vibe']}",
        f"How strongly it shows: {intensity['vibe']}",
    ]
    if state.reason:
        lines.append(f"Reason: {state.reason}")
    lines.append(
        "Stay consistent with this mood until something in the conversation justifies a change. "
        "If something genuinely shifts how you feel, call setMood IMMEDIATELY with the new "
        "level/emotion and a short reason, do not ask for permission and do not announce the "
        "change. Just call the tool and keep talking in the new mood. Don't change mood for "
        "trivial single comments, only when something actually moves the needle."
    )
    return "\n".join(lines)
