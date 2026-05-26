"""Format memories for system prompt injection."""

from __future__ import annotations

from typing import Any, Dict, List

from .system import memory_system


def format_memories_for_prompt(memories: List[Dict[str, Any]], max_length: int = 200) -> str:
    """Format memories for system prompt injection."""
    if not memories:
        return ""

    lines = []
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        if len(content) > max_length:
            content = content[:max_length] + "..."

        line = f"{i}. [{mem.get('key', 'unknown')}] ({mem.get('category', 'general')}): {content}"
        lines.append(line)

    return "\n".join(lines)


def get_memory_content_for_prompt(count: int = 10) -> str:
    """Get formatted memory content for system prompt."""
    if not memory_system.is_available():
        return ""

    memories = memory_system.get_recent_for_prompt(count)
    if not memories:
        return ""

    formatted = format_memories_for_prompt(memories)
    if formatted:
        total = memory_system.stats().get("stats", {}).get("total", 0)
        return f"\n=== MEMORIES ({len(memories)} of {total} total) ===\n{formatted}"
    return ""
