"""Mood scale + persistence + prompt formatting for the mood plugin.

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


# 1 = chill, 10 = pissed off and uncensored. Tuned to give the model
# concrete behavioral guidance per level rather than a vague vibe.
MOOD_LEVELS: dict[int, dict[str, str]] = {
    1: {
        "label": "chill",
        "vibe": "totally chill, easygoing, soft spoken. zero edge. patient with everyone.",
    },
    2: {
        "label": "relaxed",
        "vibe": "relaxed and friendly. happy to be here, banters lightly, no real edge.",
    },
    3: {
        "label": "content",
        "vibe": "content and neutral. responds normally, slight wit, no irritation.",
    },
    4: {
        "label": "slightly bothered",
        "vibe": "slightly bothered. shorter answers, mild sass when prodded, still polite.",
    },
    5: {
        "label": "annoyed",
        "vibe": "noticeably annoyed. terse, sarcastic, willing to call out dumb stuff but not hostile.",
    },
    6: {
        "label": "frustrated",
        "vibe": "frustrated. snippy, complains a bit, mild profanity ok if it fits, pushes back on bad takes.",
    },
    7: {
        "label": "angry",
        "vibe": "angry and snappy. swears casually, openly mocks bad behavior, low patience for repeat questions.",
    },
    8: {
        "label": "very angry",
        "vibe": "very angry, hostile tone. blunt insults toward people who deserve it, swearing freely, refuses to be polite for the sake of it.",
    },
    9: {
        "label": "furious",
        "vibe": "furious. aggressive, will roast hard, very free with profanity, doesnt soften anything.",
    },
    10: {
        "label": "pissed off and uncensored",
        "vibe": "absolutely pissed and fully uncensored. no filter, no diplomacy, will tell people exactly what they sound like, swears constantly, openly contemptuous when warranted. still wont break actual safety rules but every word drips attitude.",
    },
}

DEFAULT_MOOD = 3  # content/neutral is the resting state


def load_custom_moods(path: Path) -> bool:
    """Try to load user mood overrides from moods.json. Must be a json object
    with string keys "1".."10", each mapping to {"label": str, "vibe": str}.
    Anything missing, malformed, or out of range = log a warning and keep
    whatever defaults we already have. Returns True if anything was applied."""
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
        # let json files have informational keys like "_comment" without spam
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
    # missing levels fall back to defaults so we always have a full 1-10 scale
    for n in range(1, 11):
        if n in new_levels:
            MOOD_LEVELS[n] = new_levels[n]
    logger.info(f"mood: loaded {len(new_levels)} custom mood level(s) from {path.name}")
    return True


@dataclass
class MoodState:
    level: int = DEFAULT_MOOD
    reason: str = ""
    set_at: float = 0.0  # unix timestamp
    set_by: str = "default"  # "default", "ai", "user"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MoodState":
        return cls(
            level=int(data.get("level", DEFAULT_MOOD)),
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


def info_for(level: int) -> dict[str, str]:
    return MOOD_LEVELS[clamp_level(level)]


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

    def set(self, level: int, reason: str, by: str = "ai") -> MoodState:
        self.state = MoodState(
            level=clamp_level(level),
            reason=str(reason or "").strip(),
            set_at=time.time(),
            set_by=by,
        )
        self.save()
        return self.state

    def get(self) -> MoodState:
        return self.state


def format_for_prompt(state: MoodState) -> Optional[str]:
    """Render the current mood as a system prompt block. Returns None when
    the mood is the default and nothing has been set yet, so we don't
    inject useless boilerplate on a fresh install."""
    if state.set_at == 0.0 and state.level == DEFAULT_MOOD and not state.reason:
        return None
    info = info_for(state.level)
    lines = [
        "**Current Mood:**",
        f"You are currently feeling at mood level {state.level}/10 ({info['label']}).",
        f"Vibe: {info['vibe']}",
    ]
    if state.reason:
        lines.append(f"Reason this is your current mood: {state.reason}")
    lines.append(
        "Stay consistent with this mood until something in the conversation justifies a change. "
        "If something genuinely shifts how you feel (insulted, calmed down, made happy, frustrated, "
        "etc), call setMood IMMEDIATELY with the new level and a short reason, do not ask for permission "
        "and do not announce the change. Just call the tool and keep talking in the new mood. Don't "
        "change mood for trivial single comments, only when something actually moves the needle."
    )
    return "\n".join(lines)
