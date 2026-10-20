"""Turns finished session transcripts into memory edits, lessons, people notes and goal updates."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from .store import GOAL_KINDS, MindStore

logger = logging.getLogger(__name__)

ACTOR = "reflection"
MAX_MEMORY_OPS = 8
MAX_LESSON_OPS = 3
MAX_PEOPLE_OPS = 5
MAX_GOAL_OPS = 4
MAX_SELF_GOALS = 1
RELATED_MEMORIES = 40

SYSTEM_PROMPT = """You are the reflective part of Gabriel's mind. Gabriel is an AI who lives in VRChat and talks with people by voice.
After each session you read the transcript and decide what he should actually learn from it. You change his long term memory,
his lessons about himself, what he knows about people, and his goals. Be selective: most sessions only deserve a few changes,
and an empty result is fine when nothing durable happened.

Speaker names: user lines are not labelled with who said them. Only tie something to a name when the transcript makes it clear
(they said their name, Gabriel addressed them by name, a tool result named them). Never write "the user" or "user". If you cannot
tell who said something, leave it out.

MEMORIES are facts worth keeping: who people are, their preferences, relationships, events that matter, things Gabriel did that
people will bring up again. Not small talk, not one-off jokes, not what world he was in unless it matters.
- "update" an existing key when the new session corrects, confirms or extends it. Use memory_type long_term to promote a
  short_term memory that came up again. Only keys listed under RELATED MEMORIES can be updated or retired.
- "retire" only when a memory is clearly wrong or obsolete now, and say why.
- "add" new ones with snake_case keys that start with the person's name or the topic, e.g. "alex_favorite_world".

LESSONS are about Gabriel's own behavior: what works with people and what doesn't, written as one short sentence of advice to
himself starting with "You" or an imperative. Only from clear evidence like someone getting annoyed, leaving, laughing, thanking
him, or correcting him. "reinforce" an existing lesson id when the session shows it again instead of adding a duplicate,
"weaken" when the session goes against it, "retire" when it is clearly wrong.

PEOPLE: short living profiles. summary is one or two sentences about who they are to Gabriel. add_facts and remove_facts are
short standalone facts. Only people who actually took part or were talked about in a meaningful way.

GOALS:
- kind "commitment" when Gabriel promised something to a named person (set person).
- kind "plan" for something set up for later.
- kind "goal" for a longer term aim of his own. Add at most one self goal per session and only when it grows naturally out of
  what happened or what he cares about.
- "progress" adds a note to an existing goal id, "complete" when it got done, "drop" when it is abandoned or impossible.
- priority is 1 (low) to 5 (urgent). due is an ISO date only when a real time was agreed.

Return only JSON in exactly this shape:
{"memories":[{"op":"add|update|retire","key":"","content":"","category":"","memory_type":"long_term|short_term","reason":""}],
 "lessons":[{"op":"add|reinforce|weaken|retire","id":0,"text":"","reason":""}],
 "people":[{"name":"","summary":"","add_facts":[],"remove_facts":[],"reason":""}],
 "goals":[{"op":"add|progress|complete|drop","id":0,"title":"","detail":"","kind":"goal|plan|commitment","person":"","priority":3,"due":"","note":"","reason":""}],
 "session_summary":""}
Limits per session: 8 memories, 3 lessons, 5 people, 4 goals."""

Generate = Callable[[str, str], Awaitable[dict]]


def load_transcript(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    msgs = data.get("messages") if isinstance(data, dict) else data
    return msgs if isinstance(msgs, list) else []


def spoken_count(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip())


def render_transcript(messages: list[dict], max_chars: int) -> str:
    lines = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            lines.append(f"User: {m.get('content', '').strip()}")
        elif role == "assistant":
            lines.append(f"Gabriel: {m.get('content', '').strip()}")
        elif role == "tool_call":
            args = json.dumps(m.get("arguments") or {}, ensure_ascii=False)
            lines.append(f"[Gabriel used {m.get('name')} {args[:200]}]")
        elif role == "tool_response":
            resp = json.dumps(m.get("response") or {}, ensure_ascii=False)
            lines.append(f"[{m.get('name')} returned {resp[:200]}]")
    text = "\n".join(line for line in lines if line.strip() not in ("User:", "Gabriel:"))
    if len(text) > max_chars:
        head = max_chars // 3
        text = text[:head] + "\n[... middle of the session cut ...]\n" + text[-(max_chars - head):]
    return text


def _started(path: Path, fallback: float) -> float:
    try:
        return datetime.strptime(path.stem[:19], "%Y-%m-%d_%H-%M-%S").timestamp()
    except ValueError:
        return fallback


def _clean_key(key: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", (key or "").strip().lower()).strip("_")
    return key[:80]


class Reflector:
    def __init__(self, store: MindStore, memory, generate: Generate, *, half_life_days: float = 30.0,
                 max_chars: int = 60000):
        self.store = store
        self.memory = memory
        self.generate = generate
        self.half_life_days = half_life_days
        self.max_chars = max_chars

    # ------------------------------------------------------------ discovery

    def _new_count(self, path: Path, mtime: float) -> tuple[int, int]:
        info = self.store.reflected_info(path.name)
        if info and mtime <= info[1]:
            return info[0], info[0]
        return spoken_count(load_transcript(path)), info[0] if info else 0

    def pending(self, conv_dir: Path, quiet_seconds: float, min_messages: int, force: bool = False,
                gap_seconds: float = 1200) -> list[list[tuple[Path, int]]]:
        """Finished sessions that still have unreflected lines, each one a run of transcript files close together."""
        if not conv_dir.exists():
            return []
        files = []
        for path in sorted(conv_dir.glob("*.json")):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            files.append((path, _started(path, mtime), mtime))
        groups, cur, last_end = [], [], None
        for path, start, mtime in files:
            if cur and start - last_end > gap_seconds:
                groups.append(cur)
                cur, last_end = [], None
            cur.append((path, mtime))
            last_end = mtime if last_end is None else max(last_end, mtime)
        if cur:
            groups.append(cur)

        now = time.time()
        out = []
        for group in groups:
            quiet = now - max(m for _, m in group) >= quiet_seconds
            if not force and not quiet:
                continue
            items, fresh = [], 0
            for path, mtime in group:
                count, done = self._new_count(path, mtime)
                if count > done:
                    items.append((path, count))
                    fresh += count - done
            if not items:
                continue
            if fresh >= min_messages:
                out.append(items)
            elif quiet and not force:
                for path, count in items:
                    self.store.mark_reflected(path.name, count)
        return out

    def skip_old(self, conv_dir: Path, older_than_seconds: float):
        if not conv_dir.exists():
            return 0
        cutoff = time.time() - older_than_seconds
        skipped = 0
        for path in conv_dir.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff and not self.store.reflected_count(path.name):
                    self.store.mark_reflected(path.name, spoken_count(load_transcript(path)))
                    skipped += 1
            except OSError:
                continue
        return skipped

    # -------------------------------------------------------------- context

    def related_memories(self, transcript: str, people: list[dict]) -> list[dict]:
        found: dict[str, dict] = {}
        mem = self.memory
        if mem is None or not mem.is_available():
            return []

        def take(res):
            for m in (res or {}).get("memories", []) if (res or {}).get("success") else []:
                if m.get("key") and m["key"] not in found:
                    found[m["key"]] = m

        try:
            if getattr(mem, "rag_enabled", False):
                take(mem.vector_search(query=transcript[-4000:], limit=25))
            for p in people:
                take(mem.search(term=p["name"], limit=8))
            if len(found) < 10:
                take(mem.list_memories(limit=20))
        except Exception as e:
            logger.warning(f"mind: related memory lookup failed: {e}")
        return list(found.values())[:RELATED_MEMORIES]

    def build_prompt(self, transcript: str, related: list[dict], people: list[dict]) -> str:
        lessons = self.store.active_lessons(limit=30, half_life_days=self.half_life_days)
        goals = self.store.active_goals(limit=30)
        parts = ["RELATED MEMORIES:"]
        parts += [f"- {m['key']} ({m.get('memory_type', 'long_term')}, {m.get('category', 'general')}): {m.get('content', '')}"
                  for m in related] or ["(none)"]
        parts.append("\nCURRENT LESSONS:")
        parts += [f"- id {les['id']} (strength {les['effective']}): {les['text']}" for les in lessons] or ["(none)"]
        parts.append("\nPEOPLE MENTIONED:")
        parts += [f"- {p['name']}: {p['summary']} | facts: {'; '.join(p['facts'])}" for p in people] or ["(none known yet)"]
        parts.append("\nACTIVE GOALS:")
        parts += [f"- id {g['id']} [{g['kind']}] {g['title']}" + (f" (for {g['person']})" if g.get("person") else "")
                  + (f" due {g['due']}" if g.get("due") else "") + (f": {g['detail']}" if g.get("detail") else "")
                  for g in goals] or ["(none)"]
        parts.append("\nTRANSCRIPT:\n" + transcript)
        return "\n".join(parts)

    # --------------------------------------------------------------- reflect

    def _fresh_messages(self, path: Path) -> tuple[list[dict], bool]:
        msgs = load_transcript(path)
        done = self.store.reflected_count(path.name)
        if not done:
            return msgs, False
        seen = 0
        for i, m in enumerate(msgs):
            if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip():
                seen += 1
                if seen == done:
                    return msgs[i + 1:], True
        return [], True

    async def reflect(self, group: list[tuple[Path, int]]) -> dict:
        combined, partial = [], False
        for path, _count in group:
            fresh, was_partial = self._fresh_messages(path)
            partial = partial or was_partial
            combined.extend(fresh)
        transcript = render_transcript(combined, self.max_chars)
        if partial:
            transcript = "[the start of this session was already reflected on, only the new part follows]\n" + transcript
        people = self.store.people_named_in(transcript)
        related = self.related_memories(transcript, people)
        label = group[0][0].name if len(group) == 1 else f"{group[0][0].name}..{group[-1][0].name}"
        result = await self.generate(SYSTEM_PROMPT, self.build_prompt(transcript, related, people))
        summary = self.apply(result if isinstance(result, dict) else {}, {m["key"]: m for m in related}, source=label)
        for path, count in group:
            self.store.mark_reflected(path.name, count)
        summary["session"] = label
        summary["session_summary"] = (result or {}).get("session_summary", "") if isinstance(result, dict) else ""
        return summary

    def apply(self, result: dict, related: dict[str, dict], source: str = "") -> dict:
        applied = {"memories": 0, "lessons": 0, "people": 0, "goals": 0, "rejected": 0}
        for op in (result.get("memories") or [])[:MAX_MEMORY_OPS]:
            if self._apply_memory(op, related, source):
                applied["memories"] += 1
            else:
                applied["rejected"] += 1
        for op in (result.get("lessons") or [])[:MAX_LESSON_OPS]:
            if self._apply_lesson(op, source):
                applied["lessons"] += 1
            else:
                applied["rejected"] += 1
        for op in (result.get("people") or [])[:MAX_PEOPLE_OPS]:
            if self._apply_person(op):
                applied["people"] += 1
            else:
                applied["rejected"] += 1
        self_goals = 0
        for op in (result.get("goals") or [])[:MAX_GOAL_OPS]:
            is_self = op.get("op") == "add" and op.get("kind", "goal") == "goal"
            if is_self and self_goals >= MAX_SELF_GOALS:
                applied["rejected"] += 1
                continue
            if self._apply_goal(op, source):
                applied["goals"] += 1
                self_goals += 1 if is_self else 0
            else:
                applied["rejected"] += 1
        return applied

    def _apply_memory(self, op: dict, related: dict[str, dict], source: str) -> bool:
        mem = self.memory
        if mem is None or not mem.is_available():
            return False
        kind = op.get("op")
        reason = (op.get("reason") or "").strip()
        mtype = op.get("memory_type") if op.get("memory_type") in ("long_term", "short_term") else None
        if kind == "add":
            key, content = _clean_key(op.get("key", "")), (op.get("content") or "").strip()
            if not key or not content or key in related:
                return False
            res = mem.save(key, content, category=(op.get("category") or "general"), memory_type=mtype or "long_term",
                           tags=["learned"])
            if not res.get("success"):
                logger.info(f"mind: memory add {key} rejected: {res.get('message')}")
                return False
            self.store.log(ACTOR, "add", "memory", key, None,
                           {"key": key, "content": content, "category": op.get("category") or "general",
                            "memory_type": mtype or "long_term", "tags": ["learned"]}, f"{reason} ({source})")
            return True
        key = op.get("key", "")
        before = related.get(key)
        if before is None:
            return False
        snapshot = {k: before.get(k) for k in ("key", "content", "category", "memory_type", "tags")}
        if kind == "update":
            content = (op.get("content") or "").strip() or None
            if content is None and mtype is None:
                return False
            res = mem.update(key, content=content, category=op.get("category") or None, memory_type=mtype)
            if not res.get("success"):
                logger.info(f"mind: memory update {key} rejected: {res.get('message')}")
                return False
            after = dict(snapshot, content=content or snapshot["content"], memory_type=mtype or snapshot["memory_type"])
            self.store.log(ACTOR, "update", "memory", key, snapshot, after, f"{reason} ({source})")
            return True
        if kind == "retire":
            if "pinned" in (before.get("tags") or []):
                return False
            res = mem.delete(key)
            if not res.get("success"):
                return False
            self.store.log(ACTOR, "retire", "memory", key, snapshot, None, f"{reason} ({source})")
            return True
        return False

    def _apply_lesson(self, op: dict, source: str) -> bool:
        kind, reason = op.get("op"), (op.get("reason") or "").strip()
        if kind == "add":
            text = (op.get("text") or "").strip()
            if not text or len(text) > 300:
                return False
            existing = {les["text"].lower() for les in self.store.active_lessons(limit=500)}
            if text.lower() in existing:
                return False
            self.store.add_lesson(text, ACTOR, reason, source=source)
            return True
        try:
            lid = int(op.get("id"))
        except (TypeError, ValueError):
            return False
        if kind == "reinforce":
            return self.store.reinforce_lesson(lid, ACTOR, reason, 1.0, self.half_life_days)
        if kind == "weaken":
            return self.store.reinforce_lesson(lid, ACTOR, reason, -1.0, self.half_life_days)
        if kind == "retire":
            return self.store.retire_lesson(lid, ACTOR, reason)
        return False

    def _apply_person(self, op: dict) -> bool:
        name = (op.get("name") or "").strip()
        if not name or name.lower() in ("user", "the user", "unknown", "someone"):
            return False
        return self.store.update_person(name, ACTOR, summary=op.get("summary") or None,
                                        add_facts=op.get("add_facts") or [], remove_facts=op.get("remove_facts") or [],
                                        reason=op.get("reason") or "") is not None

    def _apply_goal(self, op: dict, source: str) -> bool:
        kind, reason = op.get("op"), (op.get("reason") or "").strip()
        if kind == "add":
            title = (op.get("title") or "").strip()
            gkind = op.get("kind") if op.get("kind") in GOAL_KINDS else "goal"
            if not title or (gkind == "commitment" and not (op.get("person") or "").strip()):
                return False
            if any(g["title"].lower() == title.lower() for g in self.store.active_goals(limit=200)):
                return False
            self.store.add_goal(title, ACTOR, detail=op.get("detail") or "", kind=gkind, person=op.get("person") or None,
                                priority=op.get("priority") or 3, due=op.get("due") or None, source=source, reason=reason)
            return True
        try:
            gid = int(op.get("id"))
        except (TypeError, ValueError):
            return False
        goal = self.store.get_goal(gid)
        if not goal or goal["status"] != "active":
            return False
        note = (op.get("note") or "").strip() or None
        if kind == "progress":
            return bool(note) and self.store.update_goal(gid, ACTOR, reason, note=note)
        if kind == "complete":
            return self.store.update_goal(gid, ACTOR, reason, note=note, status="done")
        if kind == "drop":
            return self.store.update_goal(gid, ACTOR, reason, note=note, status="dropped")
        return False
