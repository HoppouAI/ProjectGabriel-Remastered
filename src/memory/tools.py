"""Gemini Live function declarations + dispatch + recall sub-agent."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ._helpers import (
    MEMORY_TYPE_LONG_TERM,
    MEMORY_TYPE_QUICK_NOTE,
    MEMORY_TYPE_SHORT_TERM,
    _hash_content,
)
from .system import memory_system

logger = logging.getLogger(__name__)


# Tool declarations for Gemini Live
MEMORY_FUNCTION_DECLARATIONS = [
    {
        "name": "memory",
        "description": "Persistent memory system. Actions: save, read, update, delete, list, search, stats, cleanup, pin, promote. Memory types: 'long_term' (permanent), 'short_term' (7 days), 'quick_note' (6 hours).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "read", "update", "delete", "list", "search", "stats", "cleanup", "pin", "promote"],
                    "description": "Memory operation to perform"
                },
                "key": {
                    "type": "string",
                    "description": "Memory identifier (required for save/read/update/delete/pin/promote)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to store (required for save)"
                },
                "category": {
                    "type": "string",
                    "description": "Category (e.g., 'personal', 'work', 'facts')"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["long_term", "short_term", "quick_note"],
                    "description": "Memory persistence type"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for organization"
                },
                "search_term": {
                    "type": "string",
                    "description": "Search query (for search action)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 20)"
                },
                "new_type": {
                    "type": "string",
                    "enum": ["short_term", "long_term"],
                    "description": "Target type for promote action"
                },
                "pin": {
                    "type": "boolean",
                    "description": "Pin status for pin action (pinned memories won't be auto-deleted)"
                }
            },
            "required": ["action"]
        }
    }
]


def get_memory_tools():
    """Get memory tool declarations for Gemini Live."""
    return MEMORY_FUNCTION_DECLARATIONS


async def handle_memory_function_call(function_call) -> Dict[str, Any]:
    """Handle memory function calls from Gemini Live."""
    from google.genai import types

    args = dict(function_call.args) if function_call.args else {}
    action = args.get("action", "")

    # Support both camelCase (new) and snake_case (legacy) parameter names
    memory_type_raw = args.get("memoryType") or args.get("memory_type")
    memory_type = memory_type_raw or MEMORY_TYPE_LONG_TERM
    search_term = args.get("searchTerm") or args.get("search_term")
    new_type = args.get("newType") or args.get("new_type")

    # Parse tags - can be array (legacy) or comma-separated string (new)
    tags_raw = args.get("tags")
    if isinstance(tags_raw, str):
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
    elif isinstance(tags_raw, list):
        tags_list = tags_raw
    else:
        tags_list = None

    # Parse pin - can be boolean (legacy) or string "true"/"false" (new)
    pin_raw = args.get("pin")
    if isinstance(pin_raw, str):
        pin_val = pin_raw.lower() in ("true", "1", "yes")
    elif isinstance(pin_raw, bool):
        pin_val = pin_raw
    else:
        pin_val = True  # default

    try:
        result: Dict[str, Any]

        if action == "save":
            key = args.get("key")
            content = args.get("content")
            if not key or not content:
                result = {"result": "error", "message": "key and content required"}
            else:
                # Rate limit for quick notes
                mem_type = memory_type
                if key.startswith("note_") or mem_type == MEMORY_TYPE_QUICK_NOTE:
                    now = time.time()
                    content_hash = _hash_content(content)

                    if now - memory_system._note_last_ts < memory_system.note_min_interval:
                        result = {"result": "ok", "skipped": True, "reason": "rate_limited"}
                    elif memory_system._note_last_hash == content_hash:
                        result = {"result": "ok", "skipped": True, "reason": "duplicate"}
                    elif memory_system.has_recent_duplicate(content_hash, memory_system.dedupe_window, [MEMORY_TYPE_QUICK_NOTE, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_LONG_TERM]):
                        result = {"result": "ok", "skipped": True, "reason": "duplicate_db"}
                    else:
                        res = memory_system.save(
                            key=key,
                            content=content,
                            category=args.get("category", "general"),
                            memory_type=mem_type if mem_type != MEMORY_TYPE_LONG_TERM else MEMORY_TYPE_QUICK_NOTE,
                            tags=tags_list if tags_list else ["quick_note"]
                        )
                        if res.get("success"):
                            memory_system._note_last_ts = now
                            memory_system._note_last_hash = content_hash
                        result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}
                else:
                    res = memory_system.save(
                        key=key,
                        content=content,
                        category=args.get("category", "general"),
                        memory_type=mem_type,
                        tags=tags_list
                    )
                    result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        elif action == "read":
            key = args.get("key")
            if not key:
                result = {"result": "error", "message": "key required"}
            else:
                res = memory_system.read(key)
                if res.get("success"):
                    result = {"result": "ok", "memory": res.get("memory")}
                else:
                    result = {"result": "error", "message": res.get("message")}

        elif action == "update":
            key = args.get("key")
            if not key:
                result = {"result": "error", "message": "key required"}
            else:
                res = memory_system.update(
                    key=key,
                    content=args.get("content"),
                    category=args.get("category"),
                    memory_type=memory_type_raw,
                    tags=tags_list
                )
                result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        elif action == "delete":
            key = args.get("key")
            if not key:
                result = {"result": "error", "message": "key required"}
            else:
                res = memory_system.delete(key)
                result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        elif action == "list":
            res = memory_system.list_memories(
                category=args.get("category"),
                memory_type=memory_type_raw,
                limit=args.get("limit", 50)
            )
            if res.get("success"):
                result = {"result": "ok", "memories": res.get("memories"), "count": res.get("count")}
            else:
                result = {"result": "error", "message": res.get("message")}

        elif action == "search":
            if not search_term:
                result = {"result": "error", "message": "searchTerm required"}
            else:
                res = memory_system.search(
                    term=search_term,
                    memory_type=memory_type_raw,
                    limit=args.get("limit", 20)
                )
                if res.get("success"):
                    result = {"result": "ok", "memories": res.get("memories"), "count": res.get("count")}
                else:
                    result = {"result": "error", "message": res.get("message")}

        elif action == "stats":
            res = memory_system.stats()
            if res.get("success"):
                result = {"result": "ok", "stats": res.get("stats")}
            else:
                result = {"result": "error", "message": res.get("message")}

        elif action == "cleanup":
            res = memory_system.cleanup_expired()
            result = {"result": "ok", "deleted": res}

        elif action == "pin":
            key = args.get("key")
            if not key:
                result = {"result": "error", "message": "key required"}
            else:
                read_res = memory_system.read(key)
                if not read_res.get("success"):
                    result = {"result": "error", "message": read_res.get("message")}
                else:
                    mem = read_res["memory"]
                    tags = mem.get("tags", [])
                    if pin_val and "pinned" not in tags:
                        tags.append("pinned")
                    elif not pin_val and "pinned" in tags:
                        tags = [t for t in tags if t != "pinned"]
                    res = memory_system.update(key=key, tags=tags)
                    result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        elif action == "promote":
            key = args.get("key")
            if not key or not new_type:
                result = {"result": "error", "message": "key and newType required"}
            elif new_type not in [MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_LONG_TERM]:
                result = {"result": "error", "message": "newType must be 'short_term' or 'long_term'"}
            else:
                res = memory_system.update(key=key, memory_type=new_type)
                result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        else:
            result = {"result": "error", "message": f"Unknown action: {action}"}

        return types.FunctionResponse(id=function_call.id, name=function_call.name, response=result)

    except Exception as e:
        logger.error(f"Memory function error ({action}): {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_call.name,
            response={"result": "error", "message": str(e)}
        )


async def recall_memories(query: str, context: str = "", api_key: str = "", personality_prompt: str = "") -> Dict[str, Any]:
    """
    Memory recall: uses vector search for semantic retrieval when RAG is enabled,
    falls back to keyword search otherwise. Returns formatted memories directly
    for the main Gemini session to summarize (no separate API call).
    """
    if not memory_system.is_available():
        return {"result": "error", "message": "Memory system unavailable"}

    # Use RAG vector search when enabled, otherwise legacy keyword path
    memories_found = []
    search_method = "keyword"

    if memory_system.rag_enabled:
        vector_result = memory_system.vector_search(query=query, limit=30)
        if vector_result.get("success") and vector_result.get("memories"):
            # filter out low-relevance noise below the configured threshold
            min_score = memory_system.vector_min_score
            memories_found = [
                m for m in vector_result["memories"]
                if m.get("score", 0) >= min_score
            ]
            if memories_found:
                search_method = "vector"

    # Legacy fallback: keyword search + list (used when RAG is off or vector search fails)
    if not memories_found:
        search_result = memory_system.search(term=query, limit=100)
        searched = search_result.get("memories", []) if search_result.get("success") else []
        if len(searched) < 5:
            all_memories = memory_system.list_memories(limit=200)
            memories_found = all_memories.get("memories", [])
        else:
            memories_found = searched

    if not memories_found:
        return {"result": "ok", "summary": "No memories stored yet.", "count": 0}

    # Format memories for the main model to process directly
    memory_lines = []
    for mem in memories_found:
        key = mem.get("key", "unknown")
        content = mem.get("content", "")
        created = mem.get("created_at", "unknown")
        category = mem.get("category", "general")
        score = mem.get("score")
        prefix = f"[{key}] ({category}, {created})"
        if score is not None:
            prefix += f" [relevance: {score}]"
        memory_lines.append(f"{prefix}: {content}")

    memories_block = "\n".join(memory_lines)

    instructions = (
        "These are YOUR memories. Summarize the relevant ones for the query below. "
        "Speak in first person ('I remember...'), use actual names from the memories, "
        "and don't assume the current speaker was involved in every memory. "
        "Be concise and include specific details like names, dates, and events."
    )

    logger.info(f"Recall completed via {search_method} search ({len(memories_found)} memories)")
    return {
        "result": "ok",
        "instructions": instructions,
        "query": query,
        "context": context,
        "memories": memories_block,
        "count": len(memories_found),
        "search_method": search_method,
    }
