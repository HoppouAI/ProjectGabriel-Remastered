"""Example plugin -- adds a sayHello tool the AI can call.

Minimal reference plugin. Copy this folder, rename it, edit plugin.yml, and
build from there. The full guide lives in plugins/README.md.
"""
import logging

from google.genai import types

from src.plugins import Plugin, PluginContext
from src.tools._base import BaseTool

logger = logging.getLogger(__name__)


class HelloTool(BaseTool):
    """Demo tool. Plugin tools use the same BaseTool API as the built in
    tools, the only difference is that the plugin loader hands them to the
    host through ctx.register_tool() inside setup()."""

    tool_key = "example_hello"

    def declarations(self, config=None):
        # honour the per-plugin enable toggle so the tool can be hidden
        # without uninstalling the plugin.
        if config is not None:
            if config.get("plugins", "example_hello", "enabled", default=True) is False:
                return []
        return [types.FunctionDeclaration(
            name="sayHello",
            description=(
                "Greets a person by name with an optional vibe. Demo tool from "
                "the example plugin, mainly here to confirm the plugin loader "
                "is wired up.\n"
                "**Invocation Condition:** Call when someone explicitly asks "
                "you to test the example plugin or to say hi via the demo tool."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING", "description": "person to greet"},
                    "vibe": {"type": "STRING", "description": "optional vibe like 'cheerful', 'grumpy', 'tired'"},
                },
                "required": ["name"],
            },
        )]

    async def handle(self, name, args):
        if name != "sayHello":
            return None
        who = args.get("name", "stranger")
        vibe = args.get("vibe") or "neutral"
        msg = f"hey {who}, hello from the example plugin (vibe: {vibe})"
        logger.info(msg)
        return {"result": "ok", "message": msg}


class HelloPlugin(Plugin):
    name = "example_hello"
    version = "1.0.0"
    description = "demo plugin, registers a sayHello tool and hooks the lifecycle events"
    author = "HoppouAI"

    def setup(self, ctx: PluginContext):
        ctx.register_tool(HelloTool)
        ctx.subscribe("startup", lambda: ctx.logger.info("startup event recieved"))
        ctx.subscribe("shutdown", lambda: ctx.logger.info("shutdown event recieved"))
        # message_in fires for both transcribed VRChat speech and text input
        # (e.g. Discord relays). source is "vrchat" or "text".
        ctx.subscribe("message_in", lambda text, source="?":
                      ctx.logger.info(f"<- ({source}) {text[:80]}"))
        # message_out fires whenever the AI finishes a turn (or gets interrupted)
        ctx.subscribe("message_out", lambda text:
                      ctx.logger.info(f"-> {text[:80]}"))
        ctx.logger.info(f"example plugin ready, loaded from {ctx.plugin_dir}")

    def teardown(self, ctx: PluginContext):
        ctx.logger.info("example plugin shutting down")


# The loader will find HelloPlugin via subclass scan. Exposing it explicitly
# here as `plugin` is also supported and a little less magical.
plugin = HelloPlugin
