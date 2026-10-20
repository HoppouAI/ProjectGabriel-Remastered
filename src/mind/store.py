"""SQLite store for lessons, people, goals, the audit log and which transcripts were reflected."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GOAL_KINDS = ("goal", "plan", "commitment")
GOAL_OPEN = "active"
GOAL_STATUSES = (GOAL_OPEN, "done", "dropped")
LESSON_START = 1.0
LESSON_MAX = 5.0
LESSON_RETIRE_BELOW = 0.15
MAX_FACTS = 25


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _days_since(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return max(0.0, (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 86400)
    except ValueError:
        return 0.0


class MindStore:
    def __init__(self, path: str | Path = "data/mind.sqlite"):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        with self._lock:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    strength REAL NOT NULL DEFAULT 1.0,
                    evidence INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reinforced_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS people (
                    name TEXT PRIMARY KEY COLLATE NOCASE,
                    summary TEXT NOT NULL DEFAULT '',
                    facts_json TEXT NOT NULL DEFAULT '[]',
                    first_seen TEXT,
                    last_seen TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT 'goal',
                    person TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    priority INTEGER NOT NULL DEFAULT 3,
                    due TEXT,
                    notes_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    reason TEXT,
                    undone INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS reflected (
                    file TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL,
                    reflected_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state (k TEXT PRIMARY KEY, v TEXT);
                CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
                CREATE INDEX IF NOT EXISTS idx_lessons_status ON lessons(status);
            """)
            self._db.commit()

    def close(self):
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ audit

    def log(self, actor: str, action: str, target_type: str, target: Any, before: Any, after: Any, reason: str = "") -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO audit (ts, actor, action, target_type, target, before_json, after_json, reason) VALUES (?,?,?,?,?,?,?,?)",
                (_now(), actor, action, target_type, str(target),
                 json.dumps(before, ensure_ascii=False) if before is not None else None,
                 json.dumps(after, ensure_ascii=False) if after is not None else None,
                 reason or ""),
            )
            self._db.commit()
            return cur.lastrowid

    def audit_entries(self, limit: int = 30) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["before"] = json.loads(d.pop("before_json")) if d["before_json"] else None
            d["after"] = json.loads(d.pop("after_json")) if d["after_json"] else None
            out.append(d)
        return out

    def undo(self, audit_id: int, memory=None) -> dict:
        with self._lock:
            row = self._db.execute("SELECT * FROM audit WHERE id = ?", (audit_id,)).fetchone()
        if row is None:
            return {"success": False, "message": f"no audit entry {audit_id}"}
        if row["undone"]:
            return {"success": False, "message": f"audit entry {audit_id} was already undone"}
        before = json.loads(row["before_json"]) if row["before_json"] else None
        kind, target = row["target_type"], row["target"]
        if kind == "memory":
            if memory is None:
                return {"success": False, "message": "memory system needed to undo a memory change"}
            if before is None:
                res = memory.delete(target)
            else:
                existing = memory.read(target) if hasattr(memory, "read") else {"success": False}
                if existing.get("success"):
                    res = memory.update(target, content=before["content"], category=before.get("category"),
                                        memory_type=before.get("memory_type"), tags=before.get("tags"))
                else:
                    res = memory.save(target, before["content"], category=before.get("category", "general"),
                                      memory_type=before.get("memory_type", "long_term"), tags=before.get("tags"))
            if not res.get("success", False):
                return {"success": False, "message": res.get("message", "memory undo failed")}
        else:
            table, pk = {"lesson": ("lessons", "id"), "person": ("people", "name"), "goal": ("goals", "id")}[kind]
            with self._lock:
                if before is None:
                    self._db.execute(f"DELETE FROM {table} WHERE {pk} = ?", (target,))
                else:
                    cols = list(before)
                    self._db.execute(
                        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [before[c] for c in cols],
                    )
                self._db.commit()
        with self._lock:
            self._db.execute("UPDATE audit SET undone = 1 WHERE id = ?", (audit_id,))
            self._db.commit()
        self.log("undo", "undo", kind, target, None, before, f"undid audit #{audit_id}")
        return {"success": True, "message": f"undid #{audit_id} ({row['action']} {kind} {target})"}

    # ---------------------------------------------------------------- lessons

    def _lesson_row(self, lesson_id: int) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        return dict(r) if r else None

    def effective_strength(self, lesson: dict, half_life_days: float) -> float:
        if half_life_days <= 0:
            return lesson["strength"]
        return lesson["strength"] * 0.5 ** (_days_since(lesson["reinforced_at"]) / half_life_days)

    def add_lesson(self, text: str, actor: str, reason: str = "", source: str = "") -> int:
        now = _now()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO lessons (text, strength, evidence, source, created_at, updated_at, reinforced_at) VALUES (?,?,?,?,?,?,?)",
                (text.strip(), LESSON_START, 1, source, now, now, now),
            )
            self._db.commit()
            lid = cur.lastrowid
        self.log(actor, "add", "lesson", lid, None, self._lesson_row(lid), reason)
        return lid

    def reinforce_lesson(self, lesson_id: int, actor: str, reason: str = "", delta: float = 1.0, half_life_days: float = 0) -> bool:
        before = self._lesson_row(lesson_id)
        if not before or before["status"] != "active":
            return False
        current = self.effective_strength(before, half_life_days)
        strength = max(0.0, min(LESSON_MAX, current + delta))
        now = _now()
        with self._lock:
            self._db.execute(
                "UPDATE lessons SET strength = ?, evidence = evidence + ?, updated_at = ?, reinforced_at = ? WHERE id = ?",
                (strength, 1 if delta > 0 else 0, now, now, lesson_id),
            )
            self._db.commit()
        self.log(actor, "reinforce" if delta > 0 else "weaken", "lesson", lesson_id, before, self._lesson_row(lesson_id), reason)
        if strength < LESSON_RETIRE_BELOW:
            self.retire_lesson(lesson_id, actor, "too weak after being contradicted")
        return True

    def retire_lesson(self, lesson_id: int, actor: str, reason: str = "") -> bool:
        before = self._lesson_row(lesson_id)
        if not before or before["status"] != "active":
            return False
        with self._lock:
            self._db.execute("UPDATE lessons SET status = 'retired', updated_at = ? WHERE id = ?", (_now(), lesson_id))
            self._db.commit()
        self.log(actor, "retire", "lesson", lesson_id, before, self._lesson_row(lesson_id), reason)
        return True

    def active_lessons(self, limit: int = 50, half_life_days: float = 0) -> list[dict]:
        with self._lock:
            rows = [dict(r) for r in self._db.execute("SELECT * FROM lessons WHERE status = 'active'").fetchall()]
        for r in rows:
            r["effective"] = round(self.effective_strength(r, half_life_days), 3)
        rows.sort(key=lambda r: r["effective"], reverse=True)
        return rows[:limit]

    def decay_lessons(self, half_life_days: float) -> int:
        retired = 0
        for lesson in self.active_lessons(limit=10_000, half_life_days=half_life_days):
            if lesson["effective"] < LESSON_RETIRE_BELOW:
                if self.retire_lesson(lesson["id"], "decay", "faded, not reinforced in a long time"):
                    retired += 1
        return retired

    # ----------------------------------------------------------------- people

    def get_person(self, name: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM people WHERE name = ?", (name.strip(),)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["facts"] = json.loads(d.pop("facts_json") or "[]")
        return d

    def _person_raw(self, name: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM people WHERE name = ?", (name.strip(),)).fetchone()
        return dict(r) if r else None

    def update_person(self, name: str, actor: str, summary: str | None = None,
                      add_facts: Iterable[str] = (), remove_facts: Iterable[str] = (), reason: str = "") -> dict | None:
        name = name.strip()
        if not name:
            return None
        before = self._person_raw(name)
        facts = json.loads(before["facts_json"]) if before else []
        lowered = {f.lower() for f in remove_facts if f}
        facts = [f for f in facts if f.lower() not in lowered]
        for fact in add_facts:
            fact = (fact or "").strip()
            if fact and fact.lower() not in {f.lower() for f in facts}:
                facts.append(fact)
        facts = facts[-MAX_FACTS:]
        now = _now()
        new_summary = summary.strip() if summary else (before["summary"] if before else "")
        if before and new_summary == before["summary"] and facts == json.loads(before["facts_json"]):
            return self.get_person(name)
        with self._lock:
            self._db.execute(
                "INSERT INTO people (name, summary, facts_json, first_seen, last_seen, updated_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET summary = excluded.summary, facts_json = excluded.facts_json, updated_at = excluded.updated_at",
                (name, new_summary, json.dumps(facts, ensure_ascii=False), now, None, now),
            )
            self._db.commit()
        self.log(actor, "update" if before else "add", "person", before["name"] if before else name, before, self._person_raw(name), reason)
        return self.get_person(name)

    def mark_seen(self, name: str):
        now = _now()
        with self._lock:
            self._db.execute("UPDATE people SET last_seen = ?, first_seen = COALESCE(first_seen, ?) WHERE name = ?", (now, now, name.strip()))
            self._db.commit()

    def people_named_in(self, text: str, limit: int = 8) -> list[dict]:
        lowered = text.lower()
        with self._lock:
            names = [r["name"] for r in self._db.execute("SELECT name FROM people").fetchall()]
        hits = [n for n in names if len(n) >= 3 and n.lower() in lowered]
        hits.sort(key=lambda n: lowered.count(n.lower()), reverse=True)
        return [p for p in (self.get_person(n) for n in hits[:limit]) if p]

    # ------------------------------------------------------------------ goals

    def _goal_raw(self, goal_id: int) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
        return dict(r) if r else None

    def get_goal(self, goal_id: int) -> dict | None:
        raw = self._goal_raw(goal_id)
        if not raw:
            return None
        raw["notes"] = json.loads(raw.pop("notes_json") or "[]")
        return raw

    def add_goal(self, title: str, actor: str, detail: str = "", kind: str = "goal", person: str | None = None,
                 priority: int = 3, due: str | None = None, source: str = "", reason: str = "") -> int:
        kind = kind if kind in GOAL_KINDS else "goal"
        priority = max(1, min(5, int(priority or 3)))
        now = _now()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO goals (title, detail, kind, person, priority, due, source, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (title.strip(), (detail or "").strip(), kind, (person or "").strip() or None, priority, due or None, source, now, now),
            )
            self._db.commit()
            gid = cur.lastrowid
        self.log(actor, "add", "goal", gid, None, self._goal_raw(gid), reason)
        return gid

    def update_goal(self, goal_id: int, actor: str, reason: str = "", note: str | None = None, **fields) -> bool:
        before = self._goal_raw(goal_id)
        if not before:
            return False
        sets, vals = [], []
        for col in ("title", "detail", "kind", "person", "status", "priority", "due"):
            if col in fields and fields[col] is not None:
                val = fields[col]
                if col == "kind" and val not in GOAL_KINDS:
                    continue
                if col == "status" and val not in GOAL_STATUSES:
                    continue
                if col == "priority":
                    val = max(1, min(5, int(val)))
                sets.append(f"{col} = ?")
                vals.append(val)
        if note:
            notes = json.loads(before["notes_json"] or "[]")
            notes.append({"ts": _now(), "note": note.strip()})
            sets.append("notes_json = ?")
            vals.append(json.dumps(notes[-20:], ensure_ascii=False))
        if not sets:
            return False
        sets.append("updated_at = ?")
        vals.append(_now())
        with self._lock:
            self._db.execute(f"UPDATE goals SET {', '.join(sets)} WHERE id = ?", (*vals, goal_id))
            self._db.commit()
        action = fields.get("status") if fields.get("status") in ("done", "dropped") else "update"
        self.log(actor, action, "goal", goal_id, before, self._goal_raw(goal_id), reason or (note or ""))
        return True

    def active_goals(self, limit: int = 50, kind: str | None = None) -> list[dict]:
        q, args = "SELECT id FROM goals WHERE status = 'active'", []
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        q += " ORDER BY priority DESC, CASE WHEN due IS NULL THEN 1 ELSE 0 END, due, created_at LIMIT ?"
        args.append(limit)
        with self._lock:
            ids = [r["id"] for r in self._db.execute(q, args).fetchall()]
        return [self.get_goal(i) for i in ids]

    def open_items_for(self, person: str) -> list[dict]:
        with self._lock:
            ids = [r["id"] for r in self._db.execute(
                "SELECT id FROM goals WHERE status = 'active' AND person = ? COLLATE NOCASE ORDER BY priority DESC",
                (person.strip(),),
            ).fetchall()]
        return [self.get_goal(i) for i in ids]

    # -------------------------------------------------------------- reflected

    def reflected_count(self, file: str) -> int:
        with self._lock:
            r = self._db.execute("SELECT message_count FROM reflected WHERE file = ?", (file,)).fetchone()
        return r["message_count"] if r else 0

    def reflected_info(self, file: str) -> tuple[int, float] | None:
        with self._lock:
            r = self._db.execute("SELECT message_count, reflected_at FROM reflected WHERE file = ?", (file,)).fetchone()
        if not r:
            return None
        try:
            return r["message_count"], datetime.fromisoformat(r["reflected_at"]).timestamp()
        except ValueError:
            return r["message_count"], 0.0

    def mark_reflected(self, file: str, message_count: int):
        with self._lock:
            self._db.execute(
                "INSERT INTO reflected (file, message_count, reflected_at) VALUES (?,?,?) "
                "ON CONFLICT(file) DO UPDATE SET message_count = excluded.message_count, reflected_at = excluded.reflected_at",
                (file, message_count, _now()),
            )
            self._db.commit()

    def get_state(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            r = self._db.execute("SELECT v FROM state WHERE k = ?", (key,)).fetchone()
        return r["v"] if r else default

    def set_state(self, key: str, value: str):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO state (k, v) VALUES (?, ?)", (key, value))
            self._db.commit()
