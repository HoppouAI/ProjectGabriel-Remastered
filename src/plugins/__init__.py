"""Plugin system for ProjectGabriel.

Drop a folder into ./plugins/ with a plugin.yml manifest and an __init__.py
that subclasses Plugin, and it gets loaded on startup. See plugins/README.md
for the full author guide.
"""
from src.plugins.api import (
    DiscordPluginContext,
    Plugin,
    PluginContext,
    _bind_discord_session,
    collect_discord_prompt_contributions,
    collect_prompt_contributions,
    emit_discord_event,
    emit_event,
    get_stt_factory,
    get_tts_factory,
    iter_chatbox_sources,
    iter_discord_tool_classes,
)
from src.plugins.loader import PLUGIN_API_VERSION, PluginManager

__all__ = [
    "Plugin",
    "PluginContext",
    "DiscordPluginContext",
    "PluginManager",
    "PLUGIN_API_VERSION",
    "get_tts_factory",
    "get_stt_factory",
    "emit_event",
    "emit_discord_event",
    "iter_discord_tool_classes",
    "collect_discord_prompt_contributions",
]
