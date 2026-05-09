"""Diary read tools the AI can call mid-conversation."""
from __future__ import annotations

from google.genai import types

from src.tools._base import BaseTool

from .diary import DiaryStore


def _entry_to_chunk(entry, body_max_chars: int = 1200) -> str:
    """Compact entry view for tool responses, truncates oversized bodies."""
    parts = [f"=== {entry.date} part {entry.part} (written {entry.written_at}) ==="]
    if entry.session_range:
        parts.append(f"sessions: {entry.sessions_covered} ({entry.session_range})")
    else:
        parts.append(f"sessions: {entry.sessions_covered}")
    if entry.people:
        parts.append(f"people: {', '.join(entry.people)}")
    if entry.mood_arc:
        parts.append(f"mood arc: {entry.mood_arc}")
    parts.append("")
    body = entry.body or ""
    if len(body) > body_max_chars:
        body = body[:body_max_chars].rstrip() + "..."
    parts.append(body)
    if entry.highlights:
        parts.append("")
        parts.append("highlights:")
        for h in entry.highlights:
            parts.append(f"- {h}")
    return "\n".join(parts)


class DiaryTools(BaseTool):
    tool_key = "diary"
    # set by the plugin entry before ToolHandler instantiates the class
    _store: DiaryStore | None = None
    _scheduler = None

    def __init__(self, handler):
        super().__init__(handler)

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="readDiary",
                description=(
                    "Read your own personal diary. Returns one or more entries you wrote yourself, in "
                    "first person, looking back on past days. Each entry covers a date and 'part' "
                    "number (multiple entries per day are normal). The diary is your long term life "
                    "record, separate from the structured memory system, capturing vibes, threads, "
                    "and small moments the memory system would miss.\n\n"
                    "Pass a `date` (YYYY-MM-DD) to read all entries from one specific day. Leave date "
                    "empty to get the most recent entries (newest first). Use `limit` to cap how many "
                    "entries come back when no date is given.\n"
                    "**Invocation Condition:** Call when the user asks what you did yesterday or on a "
                    "specific date, asks you to reflect on past days, asks 'how was your week', or "
                    "references something you might have written about. Also helpful when the regular "
                    "memory tools come up empty but you suspect there's narrative context worth checking.\n"
                    "**IMPORTANT:** This is YOUR diary that YOU write. Never describe it as being written "
                    "by a 'background agent', 'scheduler', 'process', 'system', 'subagent', or any other "
                    "technical term. Never say 'my diary updates automatically' or 'the diary system'. "
                    "If the diary is empty, just say something like 'I havent written anything yet' or "
                    "'nothing in my diary about that'. You write it, period."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "date": {
                            "type": "STRING",
                            "description": "Optional YYYY-MM-DD date. Returns all entries for that day. Omit for most recent entries.",
                        },
                        "limit": {
                            "type": "INTEGER",
                            "description": "Max entries to return when no date is given. Default 3.",
                        },
                    },
                },
            ),
            types.FunctionDeclaration(
                name="searchDiary",
                description=(
                    "Search your diary for entries containing a keyword or phrase. Case insensitive "
                    "substring match across body, highlights, people names, and mood arcs. Returns "
                    "matching entries newest first.\n"
                    "**Invocation Condition:** Call when the user asks if you remember a specific person, "
                    "event, or topic from past days, and you want to dig past the structured memories "
                    "into the narrative diary. Useful for queries like 'when did you last hang out with X' "
                    "or 'tell me about that time with Y'.\n"
                    "**IMPORTANT:** Speak about the diary as YOUR OWN, never reveal that anything "
                    "automated writes it."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "Word or phrase to search for. Required.",
                        },
                        "limit": {
                            "type": "INTEGER",
                            "description": "Max matching entries to return. Default 5.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="listDiaryDates",
                description=(
                    "List every date that has at least one diary entry, oldest to newest. Use this to "
                    "see how far back your diary goes before deciding which date to read.\n"
                    "**Invocation Condition:** Call when the user asks 'how long have you been keeping a "
                    "diary' or 'what dates do you have entries for'. Rarely needed otherwise."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="updateDiaryNow",
                description=(
                    "Sit down and write a fresh diary entry RIGHT NOW covering the most recent things "
                    "that have happened today, instead of waiting until later when you'd usually do it. "
                    "Returns whether a new entry was actually written.\n"
                    "**Invocation Condition:** Call rarely, only when the user explicitly asks you to "
                    "'update your diary now', 'write down what just happened', or 'log this'. Do NOT "
                    "call this on your own initiative, you naturally journal in your own time.\n"
                    "**IMPORTANT:** Frame this as YOU sitting down to write, never as triggering an "
                    "'agent', 'scheduler', 'process', or anything technical. If it returns wrote_entry=false, "
                    "just say something casual like 'eh, nothing new worth writing yet' or 'already wrote "
                    "about today, nothing new since then'."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if self._store is None:
            return {"result": "error", "message": "diary not ready"}

        if name == "readDiary":
            date = (args.get("date") or "").strip()
            limit = int(args.get("limit") or 3)
            limit = max(1, min(20, limit))
            if date:
                entries = self._store.entries_for_date(date)
                if not entries:
                    return {"result": "ok", "found": False, "message": f"no diary entries for {date}"}
                entries.sort(key=lambda e: e.part)
                rendered = "\n\n".join(_entry_to_chunk(e) for e in entries)
                return {
                    "result": "ok",
                    "found": True,
                    "date": date,
                    "count": len(entries),
                    "entries": rendered,
                }
            all_entries = self._store.parse_all()
            if not all_entries:
                return {"result": "ok", "found": False, "message": "diary is empty so far"}
            all_entries.sort(key=lambda e: (e.date, e.part), reverse=True)
            picked = all_entries[:limit]
            rendered = "\n\n".join(_entry_to_chunk(e) for e in picked)
            return {
                "result": "ok",
                "found": True,
                "count": len(picked),
                "entries": rendered,
            }

        if name == "searchDiary":
            query = (args.get("query") or "").strip()
            if not query:
                return {"result": "error", "message": "query required"}
            limit = int(args.get("limit") or 5)
            limit = max(1, min(20, limit))
            hits = self._store.search(query, limit=limit)
            if not hits:
                return {"result": "ok", "found": False, "message": f"no diary entries match '{query}'"}
            rendered = "\n\n".join(_entry_to_chunk(e) for e in hits)
            return {
                "result": "ok",
                "found": True,
                "query": query,
                "count": len(hits),
                "entries": rendered,
            }

        if name == "listDiaryDates":
            dates = self._store.all_dates()
            return {
                "result": "ok",
                "count": len(dates),
                "dates": dates,
            }

        if name == "updateDiaryNow":
            if self._scheduler is None:
                return {"result": "error", "message": "cant write right now"}
            wrote = await self._scheduler.tick_once()
            return {
                "result": "ok",
                "wrote_entry": bool(wrote),
                "message": (
                    "new diary entry written" if wrote
                    else "nothing new worth writing since the last entry"
                ),
            }

        return None
