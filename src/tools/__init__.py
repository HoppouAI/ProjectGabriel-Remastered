from google.genai import types

from src.emotions import generate_emotion_function_declarations
from src.tools import (  # noqa: F401
    avatar_scaling,
    emotions_tools,
    memory_tools,
    movement,
    music,
    personalities,
    soundboard,
    system,
    tracker,
    voice,
    vrchat_api,
    wanderer,
)
from src.tools import discord as discord_tools  # noqa: F401
from src.tools import music_gen as music_gen_tools  # noqa: F401
from src.tools import social as social_tools  # noqa: F401
from src.tools import web_search as web_search_tools  # noqa: F401

# Import all tool modules to trigger @register_tool decorators
from src.tools._base import get_registered_tools
from src.tools._handler import ToolHandler  # noqa: F401


def get_tool_declarations(config=None):
    function_decls = []
    for cls in get_registered_tools():
        # skip tool groups disabled in config.tools.<key>.enabled
        key = getattr(cls, "tool_key", None)
        if key and config and not config.get("tools", key, "enabled", default=True):
            continue
        instance = cls.__new__(cls)
        instance.handler = None
        decls = instance.declarations(config=config)
        function_decls.extend(decls)

    # Add emotion function declarations if enabled
    if config:
        emotion_decls = generate_emotion_function_declarations(config)
        for decl in emotion_decls:
            function_decls.append(types.FunctionDeclaration(
                name=decl["name"],
                description=decl["description"],
                parameters=decl["parameters"],
            ))

    tools = []
    if config and config.google_search_enabled:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    tools.append(types.Tool(function_declarations=function_decls))
    return tools
