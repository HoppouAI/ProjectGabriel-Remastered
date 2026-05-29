"""Persistent Memory System for ProjectGabriel.

Supports MongoDB (primary) and SQLite (fallback) backends.
RAG-enabled: Uses Gemini embeddings + MongoDB Atlas Vector Search for
semantic recall, or LM Studio + ChromaDB for a fully local setup.

Memory Types:
    long_term  -- Permanent memories
    short_term -- Auto-deleted after 7 days
    quick_note -- Auto-deleted after 6 hours

Usage:
    from src.memory import memory_system, get_memory_tools, handle_memory_function_call

Implementation is split into:
    _helpers.py  -- constants, config loader, hashing, generic-subject filter
    _base.py     -- MemorySystemBase (init + connect + cleanup + close)
    rag.py       -- RAGMixin (embeddings + vector search + chroma sync)
    crud.py      -- CRUDMixin (save/read/update/delete/list/search/stats)
    system.py    -- MemorySystem (composes mixins) + lazy global proxy
    tools.py     -- Gemini Live function declarations + dispatch + recall
    prompt.py    -- format helpers for the system prompt
"""

from ._helpers import (
    MEMORY_TYPE_LONG_TERM,
    MEMORY_TYPE_QUICK_NOTE,
    MEMORY_TYPE_SHORT_TERM,
    _hash_content,
)
from .prompt import format_memories_for_prompt, get_memory_content_for_prompt
from .system import MemorySystem, memory_system
from .tools import (
    MEMORY_FUNCTION_DECLARATIONS,
    get_memory_tools,
    handle_memory_function_call,
    recall_memories,
)

__all__ = [
    "MemorySystem",
    "memory_system",
    "MEMORY_TYPE_LONG_TERM",
    "MEMORY_TYPE_SHORT_TERM",
    "MEMORY_TYPE_QUICK_NOTE",
    "_hash_content",
    "MEMORY_FUNCTION_DECLARATIONS",
    "get_memory_tools",
    "handle_memory_function_call",
    "recall_memories",
    "format_memories_for_prompt",
    "get_memory_content_for_prompt",
]
