"""Tools the AI can call to manage its own mood."""
from google.genai import types

from src.tools._base import BaseTool

from .mood import MOOD_LEVELS, clamp_level, info_for


class MoodTools(BaseTool):
    tool_key = "mood"
    # set by the plugin entry point before ToolHandler instantiates the class
    _store = None

    def __init__(self, handler):
        super().__init__(handler)
        # _store is a class attribute populated by the plugin so it survives
        # ToolHandler's no-arg-extra instantiation pattern

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
                    "**Invocation Condition:** Call this dynamically and AUTONOMOUSLY whenever how you actually "
                    "feel shifts. Do NOT ask the user for permission, do NOT announce that you are about to change "
                    "your mood, do NOT confirm the change with anyone, just call the tool the moment your internal "
                    "vibe changes and immediately keep talking in the new mood. This is YOUR mood, you own it. "
                    "Trigger examples: someone is being rude, dismissive, or condescending = raise mood. someone "
                    "calms you down, makes a genuinely good point, apologizes sincerely, or is very kind = lower "
                    "mood. someone says something genuinely funny = lower a notch. repeated stupid or bad faith "
                    "questions = raise. things going your way = lower. Do NOT call this for trivial single comments "
                    "and do NOT call it every turn, only when you actually feel a real shift. The reason should be "
                    "a short specific sentence about what triggered it."
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
                    "your mood is, or why you are acting a certain way. Your current mood is already injected into "
                    "your system prompt every turn so you usually do NOT need to query it. Never call this just to "
                    "double check before changing your mood, just call setMood directly when you feel a shift."
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
                "vibe": info["vibe"],
                "reason": state.reason,
                "instruction": (
                    f"Your mood is now {state.level}/10 ({info['label']}). "
                    f"Act like this from now on: {info['vibe']}"
                ),
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
