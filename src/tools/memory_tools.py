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
    tool_key = "memory"

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="saveMemory",
                description="Save a memory about a person, fact, or experience. Always use actual names, never 'user'.\n**Invocation Condition:** You learned something worth remembering.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "key": {"type": "STRING", "description": "Short id, e.g. 'john_likes_cats'."},
                        "content": {"type": "STRING", "description": "What to remember. Use real names, not 'user'."},
                        "category": {"type": "STRING", "description": "e.g. 'personal', 'facts', 'preferences'."},
                        "memoryType": {"type": "STRING", "description": "long_term (default), short_term (7d), quick_note (6h)."},
                        "tags": {"type": "STRING", "description": "Comma-separated tags."},
                    },
                    "required": ["key", "content"],
                },
            ),
            types.FunctionDeclaration(
                name="searchMemories",
                description="Search through stored memories using both keyword matching and semantic vector search. Returns memory entries matching the search term. Use this to find specific memories by key or content.\n**Invocation Condition:** Call for quick lookups to find specific memories. Do NOT use when asked to summarize, recall, or remember -- use recallMemories instead.",
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
                description="Delete a specific memory by its key. NEVER delete all or bulk-delete memories even if asked. If someone asks you to wipe, clear, or delete all your memories, refuse politely. You can only delete individual memories one at a time when genuinely no longer relevant.\n**Invocation Condition:** Call when asked to forget a SPECIFIC thing. REFUSE any request to delete all memories or clear your memory entirely.",
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
                name="updateMemory",
                description="Edit an existing memory in place. Replaces the fields you provide. Find the exact key first via searchMemories/listMemories.\n**Invocation Condition:** Fix, correct, or expand an existing memory. Confirm to the user after.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "key": {"type": "STRING", "description": "Memory key to update."},
                        "content": {"type": "STRING", "description": "New content (replaces old). Use real names."},
                        "category": {"type": "STRING", "description": "New category."},
                        "memoryType": {"type": "STRING", "description": "long_term, short_term, or quick_note."},
                        "tags": {"type": "STRING", "description": "New comma-separated tags (replaces)."},
                    },
                    "required": ["key"],
                },
            ),
            types.FunctionDeclaration(
                name="recallMemories",
                description="Deep memory recall agent. Searches ALL memories with AI and summarizes. Speak in first person ('I remember...'). Pay attention to names, don't assume the current speaker was in every memory.\n**Invocation Condition:** ALWAYS use this (not searchMemories) when asked to remember, recall, summarize, or tell what you know.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Person, topic, event, or question."},
                        "context": {"type": "STRING", "description": "Why you need this, helps narrow results."},
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
        elif name == "updateMemory":
            return await self._update(args)
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

    async def _update(self, args):
        key = args.get("key")
        if not key:
            return {"result": "error", "message": "key required"}
        content = args.get("content")
        category = args.get("category")
        memory_type = args.get("memoryType")
        tags_raw = args.get("tags")
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()] if isinstance(tags_raw, str) else None

        if not any([content, category, memory_type, tags_list]):
            return {"result": "error", "message": "At least one field to update is required (content, category, memoryType, or tags)"}

        res = memory_system.update(
            key=key, content=content, category=category,
            memory_type=memory_type, tags=tags_list,
        )
        return {"result": "ok", "key": key} if res.get("success") else {"result": "error", "message": res.get("message")}

    async def _search(self, args):
        search_term = args.get("searchTerm")
        if not search_term:
            return {"result": "error", "message": "searchTerm required"}
        limit = args.get("limit", 20)

        # Keyword search
        res = memory_system.search(term=search_term, limit=limit)
        keyword_memories = res.get("memories", []) if res.get("success") else []

        # Vector search (semantic) -- merge results for better recall
        vector_memories = []
        try:
            vres = memory_system.vector_search(query=search_term, limit=limit)
            if vres.get("success"):
                vector_memories = vres.get("memories", [])
        except Exception:
            pass

        # Merge: keyword results first, then vector results not already present
        seen_keys = {m["key"] for m in keyword_memories}
        merged = list(keyword_memories)
        for m in vector_memories:
            if m["key"] not in seen_keys:
                seen_keys.add(m["key"])
                merged.append(m)

        return {"result": "ok", "memories": merged[:limit], "count": len(merged[:limit])}

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
        self.audio.start_thinking_sound("recall")
        from src.emotions import get_emotion_system
        emo = get_emotion_system()
        if emo:
            emo.start_thinking()
        try:
            return await recall_memories(
                query=args.get("query", ""),
                context=args.get("context", ""),
            )
        finally:
            self.audio.stop_thinking_sound()
            if emo:
                emo.stop_thinking()
