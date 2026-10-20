"""System prompt section with his lessons, open goals and promises, and who is around right now."""

from __future__ import annotations

from collections.abc import Iterable

from .store import MindStore


def _goal_line(g: dict) -> str:
    label = {"commitment": "promise", "plan": "plan", "goal": "goal"}.get(g["kind"], "goal")
    line = f"- [{label} #{g['id']}] {g['title']}"
    if g.get("person"):
        line += f" (for {g['person']})"
    if g.get("due"):
        line += f", due {g['due']}"
    if g.get("notes"):
        line += f". Last note: {g['notes'][-1]['note']}"
    return line


def build_mind_block(store: MindStore, *, lessons: int = 8, goals: int = 6, half_life_days: float = 30.0,
                     present: Iterable[str] | None = None) -> str:
    parts = []
    learned = store.active_lessons(limit=lessons, half_life_days=half_life_days)
    if learned:
        parts.append("=== WHAT YOU HAVE LEARNED ===\nLessons from your own past sessions. Let them shape how you act, without quoting them.")
        parts += [f"- {les['text']}" for les in learned]
    open_goals = store.active_goals(limit=goals)
    if open_goals:
        parts.append("\n=== YOUR GOALS AND PROMISES ===\nThese carry over between sessions. Keep promises when the person is around, "
                     "work on goals when there is time, and use the goals tool to note progress, finish, or drop them.")
        parts += [_goal_line(g) for g in open_goals]
    here = []
    for name in present or []:
        p = store.get_person(name)
        if p and (p["summary"] or p["facts"]):
            facts = "; ".join(p["facts"][-5:])
            here.append(f"- {p['name']}: {p['summary']}" + (f" ({facts})" if facts else ""))
    if here:
        parts.append("\n=== PEOPLE YOU KNOW WHO ARE HERE ===")
        parts += here
    return "\n".join(parts).strip()
