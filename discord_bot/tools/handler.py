import logging

from google.genai import types

logger = logging.getLogger(__name__)


class DiscordToolHandler:
    """Dispatches tool calls for the Discord bot's Gemini session."""

    def __init__(self, config, relay_callback=None, personality_mgr=None):
        self.config = config
        self._relay_callback = relay_callback  # async callback to relay to main session
        self._personality_mgr = personality_mgr
        self._personality_prompt = None  # Set by PersonalityTool on switch
        self._discord_client = None  # Set by bot after login
        self._conversations = None  # Set by bot after init
        self._conversation_store = None  # Set by bot after init
        self._message_rag = None  # Set by bot when Discord RAG is enabled
        self._tool_sent_message = False  # Set by sendDiscordMessage, checked by bot to avoid double sends
        self._tools = []
        self._load_tools()

    def _load_tools(self):
        from discord_bot.tools.discord_actions import DiscordActionsTool
        from discord_bot.tools.gifs import DiscordGifTool
        from discord_bot.tools.memory import DiscordMemoryTool
        from discord_bot.tools.message_rag import DiscordMessageRagTool
        from discord_bot.tools.personalities import PersonalityTool
        from discord_bot.tools.relay import RelayTool
        from discord_bot.tools.system import DiscordSystemTool
        from discord_bot.tools.voice_control import VoiceControlTool
        self._tools = [
            DiscordMemoryTool(self),
            RelayTool(self),
            DiscordActionsTool(self),
            DiscordGifTool(self),
            DiscordMessageRagTool(self),
            DiscordSystemTool(self),
            PersonalityTool(self),
            VoiceControlTool(self),
        ]

    def set_discord_client(self, client):
        self._discord_client = client

    def get_declarations(self):
        """Return all tool declarations for the Gemini session config."""
        decls = []
        for tool in self._tools:
            decls.extend(tool.declarations())
        return [types.Tool(function_declarations=decls)]

    async def handle(self, function_call):
        """Handle a function call from Gemini and return a FunctionResponse."""
        name = function_call.name
        args = dict(function_call.args) if function_call.args else {}

        try:
            result = None
            for tool in self._tools:
                result = await tool.handle(name, args)
                if result is not None:
                    break
            if result is None:
                result = {"result": "error", "message": f"unknown function: {name}"}
        except Exception as e:
            logger.error(f"Discord tool {name} failed: {e}")
            result = {"result": "error", "message": str(e)}

        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response=result if result else {"result": "ok"},
        )
