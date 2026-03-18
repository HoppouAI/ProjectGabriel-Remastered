import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class MemoryTools(BaseTool):

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="memory",
                description="YOUR persistent memory system. These are YOUR memories -- things YOU learned, people YOU met, experiences YOU had. Actions: save, read, update, delete, list, search, stats, pin, promote. Memory types: long_term (permanent), short_term (7 days), quick_note (6 hours).\n**Invocation Condition:** Call with action=save when you learn something worth remembering. Call with action=search before asking someone a question you might already know. Always include actual usernames, never generic terms like 'User'. When telling the user about saved memories, say 'I saved/remember' not 'saved for you'.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING", "description": "Action: save, read, update, delete, list, search, stats, pin, promote"},
                        "key": {"type": "STRING", "description": "Memory identifier (required for most actions)"},
                        "content": {"type": "STRING", "description": "Content to store (required for save)"},
                        "category": {"type": "STRING", "description": "Category (e.g., 'personal', 'facts')"},
                        "memoryType": {"type": "STRING", "description": "Type: long_term, short_term, quick_note"},
                        "tags": {"type": "STRING", "description": "Comma-separated tags for organization (e.g., 'important,friend,vrc')"},
                        "searchTerm": {"type": "STRING", "description": "Search query (for search action)"},
                        "limit": {"type": "INTEGER", "description": "Max results (default 20)"},
                        "newType": {"type": "STRING", "description": "Target type for promote action"},
                        "pin": {"type": "STRING", "description": "Set to 'true' to pin, 'false' to unpin (pinned memories won't auto-delete)"},
                    },
                    "required": ["action"],
                },
            ),
            types.FunctionDeclaration(
                name="recallMemories",
                description="Deep memory recall agent. Searches through ALL stored memories using AI to find and summarize relevant information.\n**Invocation Condition:** Call when you need to remember something specific about a person, event, or topic. More thorough than basic memory search. Use when someone references past events or asks about people you have met.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "What to recall -- a person's name, topic, event, or question about past interactions"},
                        "context": {"type": "STRING", "description": "Why you need this info -- helps the recall agent find the most relevant memories"},
                    },
                    "required": ["query"],
                },
            ),
        ]

    async def handle(self, name, args):
        # Memory tools return FunctionResponse directly -- handled specially in _handler.py
        return None
