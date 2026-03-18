import logging
from google.genai import types
from src.emotions import handle_emotion_function_call

logger = logging.getLogger(__name__)


class ToolHandler:
    def __init__(self, audio_mgr, osc, tracker, personality_mgr, config=None):
        self.audio = audio_mgr
        self.osc = osc
        self.tracker = tracker
        self.wanderer = None
        self.personality = personality_mgr
        self.config = config
        self.session = None
        self.live_session = None
        self.vrchat_api = None
        self._current_avatar_id = None
        self.instance_monitor = None

        # Import all tool modules to trigger @register_tool
        from src.tools import soundboard, music, voice, personalities  # noqa: F401
        from src.tools import movement, tracker as tracker_mod, wanderer  # noqa: F401
        from src.tools import vrchat_api, system, memory_tools, emotions_tools  # noqa: F401
        from src.tools._base import get_registered_tools

        self._tools = [cls(self) for cls in get_registered_tools()]

    def _get_vrchat_api(self):
        if self.vrchat_api is None:
            from src.vrchatapi import VRChatAPI
            self.vrchat_api = VRChatAPI(self.config)
        return self.vrchat_api

    async def handle(self, function_call) -> types.FunctionResponse:
        name = function_call.name
        args = dict(function_call.args) if function_call.args else {}

        # Emotion functions return FunctionResponse directly
        if name in ("emotion", "stopAnimation"):
            try:
                return await handle_emotion_function_call(function_call)
            except Exception as e:
                logger.error(f"Emotion tool failed: {e}")
                return types.FunctionResponse(
                    id=function_call.id,
                    name=name,
                    response={"result": "error", "message": str(e)},
                )

        # General dispatch -- try each registered tool module
        try:
            result = None
            for tool in self._tools:
                result = await tool.handle(name, args)
                if result is not None:
                    break
            if result is None:
                result = {"result": "error", "message": f"unknown function: {name}"}
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            result = {"result": "error", "message": str(e)}

        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response=result if result else {"result": "ok"},
        )
