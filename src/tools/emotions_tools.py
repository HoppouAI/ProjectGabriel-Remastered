import logging
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class EmotionTools(BaseTool):
    # Emotion declarations are generated dynamically from src.emotions
    # and added in get_tool_declarations() -- this class only registers
    # itself so the handler knows to route emotion calls.

    def declarations(self, config=None):
        # Emotion declarations require config and are added in __init__.py
        return []

    async def handle(self, name, args):
        # Emotion tools return FunctionResponse directly -- handled specially in _handler.py
        return None
