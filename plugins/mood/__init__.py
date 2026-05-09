"""Mood plugin entry.

Wires up:
  - persistent MoodStore at data/plugins/mood/state.json
  - setMood / getMood tools, exposed via the regular ToolHandler dispatch
  - prompt contributor that injects the current mood block on every prompt build

Stays out of the chatbox UI on purpose, mood is meant to be felt in the
voice / wording, not displayed as a status.
"""
import logging
from pathlib import Path

from src.plugins import Plugin, PluginContext

from .mood import MoodStore, format_for_prompt, load_custom_emotions, load_custom_moods
from .tools import MoodTools

logger = logging.getLogger(__name__)


class MoodPlugin(Plugin):
    name = "mood"
    version = "1.0.0"
    description = "Persistent mood system, AI sets emotion + intensity 1-10 with a reason, mood injected into system prompt"
    author = "HoppouAI"

    def setup(self, ctx: PluginContext):
        # let users override the mood scale and add custom emotions by dropping
        # moods.json / emotions.json next to this file
        plugin_dir = Path(__file__).parent
        load_custom_moods(plugin_dir / "moods.json")
        load_custom_emotions(plugin_dir / "emotions.json")
        store = MoodStore(ctx.data_dir() / "state.json")
        # ToolHandler instantiates each registered tool as cls(handler), so we
        # cant inject the store via __init__. Stash it as a class attribute on
        # the MoodTools class instead, the instance reads self._store at call time.
        MoodTools._store = store
        ctx.register_tool(MoodTools)
        ctx.register_prompt_contributor("mood", lambda: format_for_prompt(store.get()))
        # keep a handle so other plugins / debug code can poke at it
        self._store = store
        ctx.logger.info(
            f"mood plugin ready, current mood = {store.state.emotion} @ {store.state.level} ({store.state.reason or 'no reason set'})"
        )

    def teardown(self, ctx: PluginContext):
        try:
            if hasattr(self, "_store"):
                self._store.save()
        except Exception as e:
            ctx.logger.warning(f"mood teardown save failed: {e}")
