import logging

from google.genai import types

from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)

ACTOR = "gabriel"


def _store():
    from src.mind import current
    rt = current()
    return rt.store if rt else None


def _brief(g: dict) -> dict:
    out = {"id": g["id"], "kind": g["kind"], "title": g["title"], "priority": g["priority"]}
    for k in ("detail", "person", "due"):
        if g.get(k):
            out[k] = g[k]
    if g.get("notes"):
        out["last_note"] = g["notes"][-1]["note"]
    return out


@register_tool
class MindTools(BaseTool):
    tool_key = "mind"

    def declarations(self, config=None):
        if config is not None and hasattr(config, "mind_enabled") and not config.mind_enabled:
            return []
        return [
            types.FunctionDeclaration(
                name="goals",
                description=(
                    "Your goals, plans and promises. They carry over between sessions and show up in your prompt. "
                    "kind 'commitment' is a promise to a specific person, 'plan' is something set up for later, "
                    "'goal' is a longer term aim of your own.\n"
                    "**Invocation Condition:** Call 'add' when you promise someone something, agree to a plan, or decide on a "
                    "goal for yourself. Call 'progress' when you do something toward one, 'complete' when it is done, "
                    "'drop' when it no longer makes sense, and 'list' to check what is open."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING", "enum": ["add", "progress", "complete", "drop", "list"]},
                        "id": {"type": "INTEGER", "description": "goal number, for progress, complete and drop"},
                        "title": {"type": "STRING", "description": "short title, for add"},
                        "detail": {"type": "STRING", "description": "optional extra detail, for add"},
                        "kind": {"type": "STRING", "enum": ["goal", "plan", "commitment"]},
                        "person": {"type": "STRING", "description": "who it is for, required for a commitment"},
                        "priority": {"type": "INTEGER", "description": "1 low to 5 urgent, default 3"},
                        "due": {"type": "STRING", "description": "optional ISO date like 2026-10-25"},
                        "note": {"type": "STRING", "description": "what happened, for progress, complete and drop"},
                    },
                    "required": ["action"],
                },
            ),
            types.FunctionDeclaration(
                name="personProfile",
                description=(
                    "What you have learned about a specific person over time: a short summary and facts.\n"
                    "**Invocation Condition:** Call 'get' when someone you might know talks to you and you want the "
                    "full picture of them. Call 'note' to add a fact you just learned about them."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING", "enum": ["get", "note"]},
                        "name": {"type": "STRING", "description": "their display name"},
                        "fact": {"type": "STRING", "description": "short standalone fact, for note"},
                    },
                    "required": ["action", "name"],
                },
            ),
            types.FunctionDeclaration(
                name="lessons",
                description=(
                    "Lessons you learned about how you act with people. Your strongest lessons are already in your prompt.\n"
                    "**Invocation Condition:** Call 'add' when you realize something about your own behavior that you want "
                    "to keep doing or stop doing (for example after someone tells you you talk too much). Call 'retire' "
                    "when a lesson turned out to be wrong. Call 'list' to see all of them."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING", "enum": ["add", "retire", "list"]},
                        "id": {"type": "INTEGER", "description": "lesson number, for retire"},
                        "text": {"type": "STRING", "description": "one sentence of advice to yourself, for add"},
                        "reason": {"type": "STRING", "description": "what made you realize it"},
                    },
                    "required": ["action"],
                },
            ),
            types.FunctionDeclaration(
                name="reflectNow",
                description=(
                    "Reflect on your recent sessions right now instead of waiting for the background pass, so what "
                    "happened turns into memories, lessons and goals.\n"
                    "**Invocation Condition:** Only when someone explicitly asks you to reflect or process what just happened."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name not in ("goals", "personProfile", "lessons", "reflectNow"):
            return None
        store = _store()
        if store is None:
            return {"result": "error", "message": "your mind is not running (mind is disabled in config)"}
        try:
            if name == "goals":
                return self._goals(store, args)
            if name == "personProfile":
                return self._person(store, args)
            if name == "lessons":
                return self._lessons(store, args)
            return await self._reflect()
        except Exception as e:
            logger.warning(f"{name} failed: {e}")
            return {"result": "error", "message": str(e)}

    def _goals(self, store, args):
        action = args.get("action")
        if action == "list":
            return {"result": "ok", "open": [_brief(g) for g in store.active_goals(limit=30)]}
        if action == "add":
            title = (args.get("title") or "").strip()
            kind = args.get("kind") or "goal"
            if not title:
                return {"result": "error", "message": "title is required"}
            if kind == "commitment" and not (args.get("person") or "").strip():
                return {"result": "error", "message": "a commitment needs the person it is for"}
            gid = store.add_goal(title, ACTOR, detail=args.get("detail") or "", kind=kind, person=args.get("person"),
                                 priority=args.get("priority") or 3, due=args.get("due"), source="tool")
            return {"result": "ok", "added": _brief(store.get_goal(gid))}
        try:
            gid = int(args.get("id"))
        except (TypeError, ValueError):
            return {"result": "error", "message": "id is required"}
        goal = store.get_goal(gid)
        if not goal or goal["status"] != "active":
            return {"result": "error", "message": f"no open goal #{gid}"}
        note = (args.get("note") or "").strip() or None
        if action == "progress":
            if not note:
                return {"result": "error", "message": "note is required for progress"}
            store.update_goal(gid, ACTOR, note=note)
        elif action == "complete":
            store.update_goal(gid, ACTOR, note=note, status="done")
        elif action == "drop":
            store.update_goal(gid, ACTOR, note=note, status="dropped")
        else:
            return {"result": "error", "message": f"unknown action {action}"}
        return {"result": "ok", "goal": _brief(store.get_goal(gid)), "status": store.get_goal(gid)["status"]}

    def _person(self, store, args):
        name = (args.get("name") or "").strip()
        if not name:
            return {"result": "error", "message": "name is required"}
        if args.get("action") == "note":
            fact = (args.get("fact") or "").strip()
            if not fact:
                return {"result": "error", "message": "fact is required"}
            p = store.update_person(name, ACTOR, add_facts=[fact], reason="noted in conversation")
            return {"result": "ok", "profile": p}
        p = store.get_person(name)
        if not p:
            return {"result": "ok", "profile": None, "message": f"you have no profile for {name} yet"}
        p["open_goals"] = [_brief(g) for g in store.open_items_for(name)]
        return {"result": "ok", "profile": p}

    def _lessons(self, store, args):
        action = args.get("action")
        if action == "list":
            return {"result": "ok", "lessons": [{"id": les["id"], "text": les["text"], "strength": les["effective"]}
                                                for les in store.active_lessons(limit=50, half_life_days=30)]}
        if action == "add":
            text = (args.get("text") or "").strip()
            if not text:
                return {"result": "error", "message": "text is required"}
            lid = store.add_lesson(text, ACTOR, args.get("reason") or "", source="tool")
            return {"result": "ok", "id": lid}
        if action == "retire":
            try:
                lid = int(args.get("id"))
            except (TypeError, ValueError):
                return {"result": "error", "message": "id is required"}
            ok = store.retire_lesson(lid, ACTOR, args.get("reason") or "")
            return {"result": "ok" if ok else "error", "message": "" if ok else f"no active lesson #{lid}"}
        return {"result": "error", "message": f"unknown action {action}"}

    async def _reflect(self):
        from src.mind import current
        results = await current().reflect_pending(force=True)
        if not results:
            return {"result": "ok", "message": "nothing new to reflect on yet"}
        totals = {k: sum(r[k] for r in results) for k in ("memories", "lessons", "people", "goals")}
        return {"result": "ok", "sessions": len(results), **totals,
                "summaries": [r["session_summary"] for r in results if r.get("session_summary")]}
