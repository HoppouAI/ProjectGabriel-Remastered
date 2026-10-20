"""Mind tests: store, reflection apply rules, undo, transcript discovery, reminders and tools, all offline."""
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mind.prompt import build_mind_block  # noqa: E402
from src.mind.reflect import Reflector, render_transcript  # noqa: E402
from src.mind.runtime import MindRuntime  # noqa: E402
from src.mind.store import MindStore  # noqa: E402


class FakeMemory:
    rag_enabled = False

    def __init__(self, items=()):
        self.items = {m["key"]: dict(m) for m in items}

    def is_available(self):
        return True

    def save(self, key, content, category="general", memory_type="long_term", tags=None):
        if "the user" in content.lower():
            return {"success": False, "message": "use a real name"}
        self.items[key] = {"key": key, "content": content, "category": category, "memory_type": memory_type, "tags": tags or []}
        return {"success": True}

    def update(self, key, content=None, category=None, memory_type=None, tags=None):
        if key not in self.items:
            return {"success": False, "message": "missing"}
        m = self.items[key]
        for k, v in (("content", content), ("category", category), ("memory_type", memory_type), ("tags", tags)):
            if v is not None:
                m[k] = v
        return {"success": True}

    def delete(self, key):
        return {"success": self.items.pop(key, None) is not None}

    def read(self, key):
        return {"success": key in self.items}

    def search(self, term, limit=20):
        hits = [m for m in self.items.values() if term.lower() in (m["key"] + m["content"]).lower()]
        return {"success": True, "memories": hits[:limit]}

    def list_memories(self, limit=50):
        return {"success": True, "memories": list(self.items.values())[:limit]}


class FakeConfig:
    def __init__(self, conv_dir, **over):
        self.values = {"conversation_dir": str(conv_dir), "quiet_minutes": 0, "min_messages": 2, "backfill_days": 0, **over}

    def mind(self, key, default=None):
        return self.values.get(key, default)


class FakeMonitor:
    def __init__(self):
        self.players = []

    def get_players(self):
        return [{"name": n} for n in self.players]


class FakeSession:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(text)


def write_transcript(folder: Path, name: str, lines, age_seconds=3600):
    msgs = [{"role": r, "content": c, "timestamp": "2026-10-19T10:00:00"} for r, c in lines]
    path = folder / name
    path.write_text(json.dumps({"session_start": "x", "messages": msgs}), encoding="utf-8")
    t = time.time() - age_seconds
    os.utime(path, (t, t))
    return path


def reflector_with(result, memory=None, store=None):
    store = store or MindStore(":memory:")
    calls = []

    async def gen(system, prompt):
        calls.append(prompt)
        return result

    return Reflector(store, memory or FakeMemory(), gen), store, calls


def test_lessons_reinforce_decay_and_retire():
    s = MindStore(":memory:")
    lid = s.add_lesson("Keep answers short when a group is talking.", "test")
    assert s.reinforce_lesson(lid, "test", delta=1.0)
    assert s.active_lessons()[0]["strength"] == 2.0
    old = (datetime.now() - timedelta(days=400)).isoformat(timespec="seconds")
    s._db.execute("UPDATE lessons SET reinforced_at = ? WHERE id = ?", (old, lid))
    assert s.decay_lessons(half_life_days=30) == 1
    assert s.active_lessons() == []
    actions = [e["action"] for e in s.audit_entries()]
    assert actions[0] == "retire" and "reinforce" in actions


def test_weaken_below_floor_retires():
    s = MindStore(":memory:")
    lid = s.add_lesson("Always sing.", "test")
    s.reinforce_lesson(lid, "test", delta=-1.0)
    assert s.active_lessons() == []


def test_apply_memory_rules():
    mem = FakeMemory([
        {"key": "alex_world", "content": "Alex likes The Black Cat", "category": "people", "memory_type": "short_term", "tags": []},
        {"key": "pinned_fact", "content": "Hoppou made me", "category": "core", "memory_type": "long_term", "tags": ["pinned"]},
        {"key": "unseen", "content": "not in context", "category": "x", "memory_type": "long_term", "tags": []},
    ])
    related = {k: mem.items[k] for k in ("alex_world", "pinned_fact")}
    r, store, _ = reflector_with({}, mem)
    out = r.apply({"memories": [
        {"op": "update", "key": "alex_world", "memory_type": "long_term", "reason": "came up again"},
        {"op": "retire", "key": "pinned_fact", "reason": "nope"},
        {"op": "retire", "key": "unseen", "reason": "not shown to it"},
        {"op": "add", "key": "Alex Birthday!", "content": "Alex's birthday is March 3", "category": "people"},
        {"op": "add", "key": "bad", "content": "the user likes cats"},
    ]}, related)
    assert out["memories"] == 2 and out["rejected"] == 3
    assert mem.items["alex_world"]["memory_type"] == "long_term"
    assert "pinned_fact" in mem.items and "unseen" in mem.items
    assert mem.items["alex_birthday"]["tags"] == ["learned"]


def test_apply_caps_and_self_goal_limit():
    r, store, _ = reflector_with({})
    out = r.apply({"goals": [
        {"op": "add", "kind": "goal", "title": "Learn every regular's favorite world"},
        {"op": "add", "kind": "goal", "title": "Second self goal"},
        {"op": "add", "kind": "commitment", "title": "Show the dance"},
        {"op": "add", "kind": "commitment", "title": "Show Alex the dance", "person": "Alex", "due": "2026-10-20"},
    ], "lessons": [{"op": "add", "text": f"lesson {i}"} for i in range(6)]}, {})
    assert out["goals"] == 2 and out["lessons"] == 3
    kinds = sorted(g["kind"] for g in store.active_goals())
    assert kinds == ["commitment", "goal"]


def test_goal_progress_complete_and_unknown_ids():
    r, store, _ = reflector_with({})
    gid = store.add_goal("Find a quiet world", "test")
    out = r.apply({"goals": [
        {"op": "progress", "id": gid, "note": "tried two worlds"},
        {"op": "complete", "id": 999},
        {"op": "complete", "id": gid, "note": "found one"},
    ]}, {})
    assert out["goals"] == 2 and out["rejected"] == 1
    g = store.get_goal(gid)
    assert g["status"] == "done" and len(g["notes"]) == 2


def test_people_profiles_merge_facts():
    r, store, _ = reflector_with({})
    r.apply({"people": [{"name": "Alex", "summary": "Regular who loves horror worlds.", "add_facts": ["from Ohio", "plays drums"]}]}, {})
    r.apply({"people": [{"name": "alex", "add_facts": ["plays drums", "has a cat"], "remove_facts": ["from Ohio"]},
                        {"name": "the user", "summary": "x"}]}, {})
    p = store.get_person("ALEX")
    assert p["summary"] == "Regular who loves horror worlds."
    assert p["facts"] == ["plays drums", "has a cat"]
    assert store.get_person("the user") is None


def test_undo_memory_lesson_and_goal():
    mem = FakeMemory([{"key": "k1", "content": "old", "category": "c", "memory_type": "long_term", "tags": []}])
    r, store, _ = reflector_with({}, mem)
    r.apply({"memories": [{"op": "update", "key": "k1", "content": "new"},
                          {"op": "add", "key": "k2", "content": "fresh fact about Sam"}],
             "lessons": [{"op": "add", "text": "Ask people their name early."}],
             "goals": [{"op": "add", "kind": "plan", "title": "Meet Sam at 8"}]}, {"k1": dict(mem.items["k1"])})
    by_target = {(e["target_type"], e["target"]): e["id"] for e in store.audit_entries()}
    assert store.undo(by_target[("memory", "k1")], memory=mem)["success"]
    assert mem.items["k1"]["content"] == "old"
    assert store.undo(by_target[("memory", "k2")], memory=mem)["success"]
    assert "k2" not in mem.items
    lesson_id = [t for (kind, t) in by_target if kind == "lesson"][0]
    assert store.undo(by_target[("lesson", lesson_id)])["success"]
    assert store.active_lessons() == []
    goal_id = [t for (kind, t) in by_target if kind == "goal"][0]
    assert store.undo(by_target[("goal", goal_id)])["success"]
    assert store.active_goals() == []
    assert not store.undo(by_target[("goal", goal_id)])["success"], "double undo should be refused"


def test_undo_restores_retired_memory():
    mem = FakeMemory([{"key": "stale", "content": "Sam lives in Texas", "category": "people", "memory_type": "long_term", "tags": []}])
    r, store, _ = reflector_with({}, mem)
    r.apply({"memories": [{"op": "retire", "key": "stale", "reason": "moved"}]}, {"stale": dict(mem.items["stale"])})
    assert "stale" not in mem.items
    assert store.undo(store.audit_entries()[0]["id"], memory=mem)["success"]
    assert mem.items["stale"]["content"] == "Sam lives in Texas"


def _stamp(age_seconds):
    return (datetime.now() - timedelta(seconds=age_seconds + 60)).strftime("%Y-%m-%d_%H-%M-%S") + ".json"


def test_pending_skips_short_recent_and_done(tmp_path):
    first, tiny, fresh = _stamp(7200), _stamp(3600), _stamp(5)
    write_transcript(tmp_path, first, [("user", "hi"), ("assistant", "hey"), ("user", "bye")], age_seconds=7200)
    write_transcript(tmp_path, tiny, [("user", "hi")], age_seconds=3600)
    r, store, _ = reflector_with({})
    groups = r.pending(tmp_path, quiet_seconds=600, min_messages=2)
    assert [[p.name for p, _ in g] for g in groups] == [[first]]
    assert store.reflected_count(tiny) == 1, "finished tiny session gets marked so it is not rescanned"
    store.mark_reflected(first, 3)
    assert r.pending(tmp_path, quiet_seconds=600, min_messages=2) == []
    write_transcript(tmp_path, fresh, [("user", "hi"), ("assistant", "yo")], age_seconds=5)
    assert r.pending(tmp_path, quiet_seconds=600, min_messages=2) == []
    forced = r.pending(tmp_path, quiet_seconds=600, min_messages=2, force=True)
    assert [[p.name for p, _ in g] for g in forced] == [[fresh]]


def test_reconnect_fragments_become_one_session(tmp_path):
    base = datetime(2026, 10, 19, 20, 0, 0)
    for i in range(4):
        start = base + timedelta(minutes=5 * i)
        name = start.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        path = write_transcript(tmp_path, name, [("user", f"part {i}"), ("assistant", f"reply {i}")])
        t = (start + timedelta(minutes=4)).timestamp()
        os.utime(path, (t, t))
    late = base + timedelta(hours=3)
    path = write_transcript(tmp_path, late.strftime("%Y-%m-%d_%H-%M-%S") + ".json", [("user", "later"), ("assistant", "much later")])
    os.utime(path, (late.timestamp() + 60,) * 2)
    r, store, calls = reflector_with({"session_summary": "ok"})
    groups = r.pending(tmp_path, quiet_seconds=600, min_messages=6, gap_seconds=1200)
    assert len(groups) == 1 and len(groups[0]) == 4, "four 2 line fragments make one 8 line session, the lone later one is too short"
    res = asyncio.run(r.reflect(groups[0]))
    assert all(f"part {i}" in calls[0] for i in range(4))
    assert ".." in res["session"]
    assert r.pending(tmp_path, quiet_seconds=600, min_messages=6, gap_seconds=1200) == []


def test_reflect_sends_only_new_part_after_partial(tmp_path):
    path = write_transcript(tmp_path, "s.json", [("user", "first part"), ("assistant", "reply one"),
                                                 ("user", "second part"), ("assistant", "reply two")])
    r, store, calls = reflector_with({"session_summary": "ok"})
    store.mark_reflected("s.json", 2)
    res = asyncio.run(r.reflect([(path, 4)]))
    assert "first part" not in calls[0] and "second part" in calls[0]
    assert store.reflected_count("s.json") == 4 and res["session_summary"] == "ok"


def test_render_transcript_trims_middle():
    msgs = [{"role": "user", "content": f"line {i} " + "x" * 50} for i in range(200)]
    text = render_transcript(msgs, 2000)
    assert "line 0" in text and "line 199" in text and "middle of the session cut" in text and len(text) < 2100


def test_runtime_first_run_skips_backlog(tmp_path):
    write_transcript(tmp_path, "old.json", [("user", "a"), ("assistant", "b")], age_seconds=10 * 86400)
    write_transcript(tmp_path, "new.json", [("user", "a"), ("assistant", "b")], age_seconds=3600)

    async def go():
        rt = MindRuntime(FakeConfig(tmp_path, backfill_days=2), store=MindStore(":memory:"), memory=FakeMemory(),
                         generate=lambda s, p: asyncio.sleep(0, result={}))
        rt.start()
        await rt.stop()
        assert rt.store.reflected_count("old.json") == 2
        results = await rt.reflect_pending()
        assert [x["session"] for x in results] == ["new.json"]
    asyncio.run(go())


def test_presence_reminder_and_cooldown(tmp_path):
    async def go():
        mon, sess = FakeMonitor(), FakeSession()
        rt = MindRuntime(FakeConfig(tmp_path), session=sess, instance_monitor=mon, store=MindStore(":memory:"),
                         memory=FakeMemory(), generate=lambda s, p: asyncio.sleep(0, result={}))
        rt.store.add_goal("Show Alex the dance", "test", kind="commitment", person="Alex")
        mon.players = ["Bob"]
        await rt.check_presence()
        mon.players = ["Bob", "alex"]
        await rt.check_presence()
        await asyncio.sleep(0)
        assert len(sess.sent) == 1 and "alex just joined" in sess.sent[0] and "Show Alex the dance" in sess.sent[0]
        mon.players = ["Bob"]
        await rt.check_presence()
        mon.players = ["Bob", "Alex"]
        await rt.check_presence()
        await asyncio.sleep(0)
        assert len(sess.sent) == 1, "should not remind again inside the cooldown"
    asyncio.run(go())


def test_idle_prompt_prefers_present_promise(tmp_path):
    mon = FakeMonitor()
    rt = MindRuntime(FakeConfig(tmp_path), instance_monitor=mon, store=MindStore(":memory:"), memory=FakeMemory(),
                     generate=lambda s, p: None)
    assert rt.idle_prompt() is None
    rt.store.add_goal("Explore a new world every day", "test", kind="goal")
    assert "Explore a new world" in rt.idle_prompt()
    rt.store.add_goal("Teach Sam a magic trick", "test", kind="commitment", person="Sam")
    mon.players = ["Sam"]
    assert "Sam is here" in rt.idle_prompt()
    rt.config.values["idle_goals"] = False
    assert rt.idle_prompt() is None


def test_prompt_block_contents():
    s = MindStore(":memory:")
    assert build_mind_block(s) == ""
    s.add_lesson("Do not interrupt people mid story.", "t")
    gid = s.add_goal("Show Alex the dance", "t", kind="commitment", person="Alex", due="2026-10-20")
    s.update_goal(gid, "t", note="practiced it")
    s.update_person("Alex", "t", summary="Drummer from Ohio.", add_facts=["has a cat"])
    block = build_mind_block(s, present=["Alex", "Nobody"])
    assert "Do not interrupt" in block
    assert "[promise #1] Show Alex the dance (for Alex), due 2026-10-20. Last note: practiced it" in block
    assert "- Alex: Drummer from Ohio. (has a cat)" in block


def test_tools_round_trip(tmp_path):
    from src.tools.mind_tools import MindTools

    async def go():
        rt = MindRuntime(FakeConfig(tmp_path), store=MindStore(":memory:"), memory=FakeMemory(),
                         generate=lambda s, p: asyncio.sleep(0, result={}))
        tool = MindTools(None)
        assert await tool.handle("somethingElse", {}) is None
        bad = await tool.handle("goals", {"action": "add", "kind": "commitment", "title": "x"})
        assert bad["result"] == "error"
        added = await tool.handle("goals", {"action": "add", "kind": "commitment", "title": "Bring Sam to the club", "person": "Sam"})
        gid = added["added"]["id"]
        await tool.handle("goals", {"action": "progress", "id": gid, "note": "asked Sam"})
        done = await tool.handle("goals", {"action": "complete", "id": gid, "note": "went together"})
        assert done["status"] == "done"
        assert (await tool.handle("goals", {"action": "list"}))["open"] == []
        await tool.handle("personProfile", {"action": "note", "name": "Sam", "fact": "likes jazz"})
        prof = await tool.handle("personProfile", {"action": "get", "name": "sam"})
        assert prof["profile"]["facts"] == ["likes jazz"]
        lid = (await tool.handle("lessons", {"action": "add", "text": "Ask before playing loud music."}))["id"]
        assert (await tool.handle("lessons", {"action": "list"}))["lessons"][0]["id"] == lid
        assert (await tool.handle("lessons", {"action": "retire", "id": lid}))["result"] == "ok"
        assert (await tool.handle("reflectNow", {}))["message"] == "nothing new to reflect on yet"
        assert {e["actor"] for e in rt.store.audit_entries()} == {"gabriel"}
    asyncio.run(go())


def test_tools_hidden_when_disabled():
    from src.tools.mind_tools import MindTools

    class Off:
        mind_enabled = False
    assert MindTools(None).declarations(Off()) == []
    assert len(MindTools(None).declarations(None)) == 4
