"""Plugin system for ProjectGabriel.

Drop a folder into ./plugins/ with a plugin.yml manifest and an __init__.py
that subclasses Plugin, and it gets loaded on startup. See plugins/README.md
for the full author guide.
"""
from src.plugins.api import (
    Plugin,
    PluginContext,
    emit_event,
    get_stt_factory,
    get_tts_factory,
    iter_chatbox_sources,
)
from src.plugins.loader import PLUGIN_API_VERSION, PluginManager

__all__ = [
    "Plugin",
    "PluginContext",
    "PluginManager",
    "PLUGIN_API_VERSION",
    "get_tts_factory",
    "get_stt_factory",
    "emit_event",
]
