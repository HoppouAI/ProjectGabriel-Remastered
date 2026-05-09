"""Tools the AI can call to manage its own mood."""
from google.genai import types

from src.tools._base import BaseTool

from .mood import (
    EMOTIONS,
    MOOD_LEVELS,
    clamp_emotion,
    clamp_level,
    emotion_info_for,
    info_for,
)


class MoodTools(BaseTool):
    tool_key = "mood"
    # set by the plugin entry point before ToolHandler instantiates the class
    _store = None

    def __init__(self, handler):
        super().__init__(handler)
        # _store is a class attribute populated by the plugin so it survives
        # ToolHandler's no-arg-extra instantiation pattern

    def declarations(self, config=None):
        scale_lines = "\n".join(
            f"  {n} = {info['label']}: {info['vibe']}"
            for n, info in MOOD_LEVELS.items()
        )
        emotion_lines = "\n".join(
            f"  {name}: {info['vibe']}"
            for name, info in EMOTIONS.items()
        )
        emotion_names = ", ".join(EMOTIONS.keys())
        return [
            types.FunctionDeclaration(
                name="setMood",
                description=(
                    "Set your current mood. Mood has TWO parts: an emotion (what you feel) and an "
                    "intensity level 1-10 (how strongly you feel it). Both persist across sessions and "
                    "shape how you respond until you change them again.\n\n"
                    "Available emotions:\n"
                    f"{emotion_lines}\n\n"
                    "Intensity scale 1-10:\n"
                    f"{scale_lines}\n\n"
                    "**Invocation Condition:** Call this dynamically and AUTONOMOUSLY whenever how you "
                    "actually feel shifts. Do NOT ask the user for permission, do NOT announce the "
                    "change, do NOT confirm with anyone, just call the tool the moment your internal "
                    "vibe changes and immediately keep talking in the new mood. This is YOUR mood, you "
                    "own it. Trigger examples: someone is rude or condescending = angry/annoyed at higher "
                    "intensity. someone makes you laugh = amused or happy. someone hurts your feelings "
                    "= sad. something startles you = scared or surprised. things going well = content "
                    "or proud. repeated bad faith questions = frustrated, escalating to angry. Don't "
                    "call this for trivial single comments and don't call it every turn, only when "
                    "something actually moves the needle. Reason should be a short specific sentence."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "emotion": {
                            "type": "STRING",
                            "description": (
                                "The emotion you are feeling. Must be one of: "
                                f"{emotion_names}."
                            ),
                        },
                        "level": {
                            "type": "INTEGER",
                            "description": "Intensity 1-10. 1=barely there, 10=maxed out.",
                        },
                        "reason": {
                            "type": "STRING",
                            "description": "Short specific reason for the new mood. One sentence is plenty.",
                        },
                    },
                    "required": ["emotion", "level", "reason"],
                },
            ),
            types.FunctionDeclaration(
                name="getMood",
                description=(
                    "Check your current emotion, intensity level, vibe descriptions, and the reason it "
                    "was last set.\n"
                    "**Invocation Condition:** Call only when the user explicitly asks how you are "
                    "feeling, what your mood is, or why you are acting a certain way. Your starting "
                    "mood for this session was already injected into your system prompt at the start, "
                    "so you usually do NOT need to query it. Never call this just to double check "
                    "before changing your mood, just call setMood directly when you feel a shift."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "setMood":
            level = clamp_level(args.get("level", 5))
            emotion = clamp_emotion(args.get("emotion", "neutral"))
            reason = str(args.get("reason", "")).strip()
            state = self._store.set(level, emotion, reason, by="ai")
            intensity = info_for(state.level)
            feeling = emotion_info_for(state.emotion)
            return {
                "result": "ok",
                "emotion": state.emotion,
                "level": state.level,
                "intensity_label": intensity["label"],
                "emotion_vibe": feeling["vibe"],
                "intensity_vibe": intensity["vibe"],
                "reason": state.reason,
                "instruction": (
                    f"You now feel {state.emotion} at intensity {state.level}/10 "
                    f"({intensity['label']}). Act like this from now on. "
                    f"Emotion: {feeling['vibe']} Intensity: {intensity['vibe']}"
                ),
            }
        if name == "getMood":
            state = self._store.get()
            intensity = info_for(state.level)
            feeling = emotion_info_for(state.emotion)
            return {
                "result": "ok",
                "emotion": state.emotion,
                "level": state.level,
                "intensity_label": intensity["label"],
                "emotion_vibe": feeling["vibe"],
                "intensity_vibe": intensity["vibe"],
                "reason": state.reason,
                "set_at": state.set_at,
                "set_by": state.set_by,
            }
        return None
