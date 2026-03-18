import logging
import time
from google.genai import types
from src.tools._base import BaseTool, register_tool
from src.memory import (
    memory_system, recall_memories, _hash_content,
    MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_QUICK_NOTE,
)

logger = logging.getLogger(__name__)


@register_tool
class MemoryTools(BaseTool):

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="saveMemory",
                description="Save a memory. These are YOUR memories -- things YOU learned, people YOU met, experiences YOU had. Always include actual usernames. Say 'I saved/remember' not 'saved for you'.\n**Invocation Condition:** Call when you learn something worth remembering about a person, fact, or experience.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "key": {"type": "STRING", "description": "Short identifier (e.g., 'john_likes_cats')"},
                        "content": {"type": "STRING", "description": "What to remember"},
                        "category": {"type": "STRING", "description": "Category (e.g., 'personal', 'facts', 'preferences')"},
                        "memoryType": {"type": "STRING", "description": "long_term (permanent), short_term (7 days), or quick_note (6 hours)"},
                        "tags": {"type": "STRING", "description": "Comma-separated tags (e.g., 'friend,vrc,important')"},
                    },
                    "required": ["key", "content"],
                },
            ),
            types.FunctionDeclaration(
                name="searchMemories",
                description="Search through your stored memories by keyword or phrase.\n**Invocation Condition:** Call before asking someone a question you might already know the answer to.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "searchTerm": {"type": "STRING", "description": "Search query"},
                        "limit": {"type": "INTEGER", "description": "Max results (default 20)"},
                    },
                    "required": ["searchTerm"],
                },
            ),
            types.FunctionDeclaration(
                name="deleteMemory",
                description="Delete a specific memory by its key.\n**Invocation Condition:** Call when asked to forget something or when a memory is no longer relevant.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "key": {"type": "STRING", "description": "The memory key to delete"},
                    },
                    "required": ["key"],
                },
            ),
            types.FunctionDeclaration(
                name="listMemories",
                description="List your stored memories, optionally filtered by category or type.\n**Invocation Condition:** Call when you want to review what you remember about a topic or category.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "category": {"type": "STRING", "description": "Filter by category"},
                        "memoryType": {"type": "STRING", "description": "Filter: long_term, short_term, or quick_note"},
                        "limit": {"type": "INTEGER", "description": "Max results (default 50)"},
                    },
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="recallMemories",
                description="Deep memory recall agent. Uses AI to search ALL memories and summarize relevant information.\n**Invocation Condition:** Call when you need to remember something specific about a person, event, or topic. More thorough than searchMemories. Use when someone references past events or asks about people you met.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "What to recall -- person's name, topic, event, or question"},
                        "context": {"type": "STRING", "description": "Why you need this -- helps find the most relevant memories"},
                    },
                    "required": ["query"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "saveMemory":
            return await self._save(args)
        elif name == "searchMemories":
            return await self._search(args)
        elif name == "deleteMemory":
            return await self._delete(args)
        elif name == "listMemories":
            return await self._list(args)
        elif name == "recallMemories":
            return await self._recall(args)
        return None

    async def _save(self, args):
        key = args.get("key")
        content = args.get("content")
        if not key or not content:
            return {"result": "error", "message": "key and content required"}

        memory_type = args.get("memoryType", MEMORY_TYPE_LONG_TERM)
        tags_raw = args.get("tags")
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()] if isinstance(tags_raw, str) else None

        if key.startswith("note_") or memory_type == MEMORY_TYPE_QUICK_NOTE:
            now = time.time()
            content_hash = _hash_content(content)
            if now - memory_system._note_last_ts < memory_system.note_min_interval:
                return {"result": "ok", "skipped": True, "reason": "rate_limited"}
            if memory_system._note_last_hash == content_hash:
                return {"result": "ok", "skipped": True, "reason": "duplicate"}
            if memory_system.has_recent_duplicate(content_hash, memory_system.dedupe_window,
                    [MEMORY_TYPE_QUICK_NOTE, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_LONG_TERM]):
                return {"result": "ok", "skipped": True, "reason": "duplicate_db"}
            mem_type = memory_type if memory_type != MEMORY_TYPE_LONG_TERM else MEMORY_TYPE_QUICK_NOTE
            res = memory_system.save(
                key=key, content=content,
                category=args.get("category", "general"),
                memory_type=mem_type,
                tags=tags_list if tags_list else ["quick_note"],
            )
            if res.get("success"):
                memory_system._note_last_ts = now
                memory_system._note_last_hash = content_hash
            return {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        res = memory_system.save(
            key=key, content=content,
            category=args.get("category", "general"),
            memory_type=memory_type,
            tags=tags_list,
        )
        return {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

    async def _search(self, args):
        search_term = args.get("searchTerm")
        if not search_term:
            return {"result": "error", "message": "searchTerm required"}
        res = memory_system.search(term=search_term, limit=args.get("limit", 20))
        if res.get("success"):
            return {"result": "ok", "memories": res.get("memories"), "count": res.get("count")}
        return {"result": "error", "message": res.get("message")}

    async def _delete(self, args):
        key = args.get("key")
        if not key:
            return {"result": "error", "message": "key required"}
        res = memory_system.delete(key)
        return {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

    async def _list(self, args):
        res = memory_system.list_memories(
            category=args.get("category"),
            memory_type=args.get("memoryType"),
            limit=args.get("limit", 50),
        )
        if res.get("success"):
            return {"result": "ok", "memories": res.get("memories"), "count": res.get("count")}
        return {"result": "error", "message": res.get("message")}

    async def _recall(self, args):
        if self.osc:
            self.osc.send_chatbox("Thinking about the past...")
        api_key = self.config.api_key if self.config else ""
        personality_prompt = ""
        if self.personality:
            current = self.personality.get_current()
            personality_prompt = current.get("prompt", "")
        return await recall_memories(
            query=args.get("query", ""),
            context=args.get("context", ""),
            api_key=api_key,
            personality_prompt=personality_prompt,
        )
