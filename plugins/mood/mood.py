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
        "If something in the conversation makes you actually feel different (insulted, calmed down, "
        "made happy, etc), call setMood with a new level and a short reason. Don't change mood for "
        "trivial things, only when it is genuinely warranted."
    )
    return "\n".join(lines)
