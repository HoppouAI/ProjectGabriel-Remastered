"""python -m src.mind log | undo <id> | lessons | goals | people <name>"""

import argparse
import json
import sys

from .store import MindStore


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.mind", description="look at and undo what Gabriel's mind changed")
    ap.add_argument("--db", default="data/mind.sqlite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="recent changes")
    lg.add_argument("limit", nargs="?", type=int, default=30)
    un = sub.add_parser("undo", help="undo one change by its number from the log")
    un.add_argument("id", type=int)
    sub.add_parser("lessons", help="active lessons")
    sub.add_parser("goals", help="open goals and promises")
    pp = sub.add_parser("people", help="profile for one person")
    pp.add_argument("name")
    args = ap.parse_args(argv)

    store = MindStore(args.db)
    if args.cmd == "log":
        for e in reversed(store.audit_entries(args.limit)):
            mark = " (undone)" if e["undone"] else ""
            print(f"#{e['id']} {e['ts']} {e['actor']}: {e['action']} {e['target_type']} {e['target']}{mark}")
            if e["reason"]:
                print(f"    why: {e['reason']}")
    elif args.cmd == "undo":
        memory = None
        try:
            from src.memory import memory_system
            memory = memory_system
        except Exception as e:
            print(f"memory system not available ({e}), memory changes cant be undone right now")
        res = store.undo(args.id, memory=memory)
        print(res["message"])
        return 0 if res["success"] else 1
    elif args.cmd == "lessons":
        for les in store.active_lessons(limit=100, half_life_days=30):
            print(f"#{les['id']} [{les['effective']}] {les['text']}")
    elif args.cmd == "goals":
        for g in store.active_goals(limit=100):
            who = f" for {g['person']}" if g.get("person") else ""
            print(f"#{g['id']} [{g['kind']} p{g['priority']}] {g['title']}{who}")
    elif args.cmd == "people":
        print(json.dumps(store.get_person(args.name), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
