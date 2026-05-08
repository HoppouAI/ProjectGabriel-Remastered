"""Tools the AI can call to manage its own mood."""
from google.genai import types

from src.tools._base import BaseTool

from .mood import MOOD_LEVELS, clamp_level, info_for


class MoodTools(BaseTool):
    tool_key = "mood"

    def __init__(self, store):
        super().__init__()
        self._store = store

    def declarations(self, config=None):
        # Build a description that lists the levels so the model knows the scale
        scale_lines = "\n".join(
            f"  {n} = {info['label']}: {info['vibe']}"
            for n, info in MOOD_LEVELS.items()
        )
        return [
            types.FunctionDeclaration(
                name="setMood",
                description=(
                    "Set your current mood and the reason for it. The mood persists across sessions and shapes "
                    "how you respond until you change it again. Mood scale 1-10:\n"
                    f"{scale_lines}\n"
                    "**Invocation Condition:** Call when something in the conversation actually changes how you feel. "
                    "Examples: someone is being rude/dismissive (raise mood), someone calms you down or makes a "
                    "good point (lower mood), someone says something genuinely funny (lower a bit), repeated "
                    "stupid questions (raise). Do NOT call this for trivial single comments. Do NOT call this every "
                    "turn. The reason should be a short specific sentence about what triggered the change."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "level": {
                            "type": "INTEGER",
                            "description": "Mood level 1-10. 1=chill, 10=pissed off and uncensored.",
                        },
                        "reason": {
                            "type": "STRING",
                            "description": "Short specific reason for the new mood. One sentence is plenty.",
                        },
                    },
                    "required": ["level", "reason"],
                },
            ),
            types.FunctionDeclaration(
                name="getMood",
                description=(
                    "Check your current mood level, label, vibe description, and the reason it was last set.\n"
                    "**Invocation Condition:** Call only when the user explicitly asks how you are feeling, what "
                    "your mood is, or why you are acting a certain way. The mood is already in your system prompt "
                    "so you usually don't need to query it."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "setMood":
            level = clamp_level(args.get("level", 3))
            reason = str(args.get("reason", "")).strip()
            state = self._store.set(level, reason, by="ai")
            info = info_for(state.level)
            return {
                "result": "ok",
                "level": state.level,
                "label": info["label"],
                "reason": state.reason,
            }
        if name == "getMood":
            state = self._store.get()
            info = info_for(state.level)
            return {
                "result": "ok",
                "level": state.level,
                "label": info["label"],
                "vibe": info["vibe"],
                "reason": state.reason,
                "set_at": state.set_at,
                "set_by": state.set_by,
            }
        return None
