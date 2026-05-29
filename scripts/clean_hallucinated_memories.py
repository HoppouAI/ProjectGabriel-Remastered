"""Scrub fake timestamp-shaped usernames out of saved memories.

The model used to hallucinate usernames in the shape MINUTES_EPOCH:SECONDS
(e.g. "29661927:52" or "hello000000_29664978_41") when it didnt know who
was talking. Those got persisted to long term memory and then the model
saw them again on the next session, reinforcing the pattern.

This script walks every memory in the configured backend (sqlite/mongo)
plus the chroma collection if local rag is on, and:

  --dry-run     just lists offenders, makes no changes (default)
  --delete      removes any memory whose key or content matches
  --scrub       rewrites the bad token to "Unknown" instead of deleting

Run from repo root:
    .\\.venv\\Scripts\\python.exe scripts\\clean_hallucinated_memories.py --dry-run
    .\\.venv\\Scripts\\python.exe scripts\\clean_hallucinated_memories.py --delete
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.memory import memory_system  # noqa: E402
from src.memory._helpers import _HALLUCINATED_USERNAME_RE  # noqa: E402

logger = logging.getLogger("clean_memories")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def _iter_all_memories():
    """Yield every (key, content, category, memory_type) tuple from the active backend."""
    if not memory_system.is_available():
        raise SystemExit("memory_system not available, check config.yml")

    if memory_system.backend == "sqlite":
        with memory_system._sqlite_lock:
            rows = memory_system.sqlite_conn.execute(
                "SELECT key, content, category, memory_type FROM memories"
            ).fetchall()
        for row in rows:
            yield row["key"], row["content"], row["category"], row["memory_type"]
    else:
        cursor = memory_system.collection.find({}, {"embedding": 0})
        for doc in cursor:
            yield (
                doc.get("key"),
                doc.get("content", ""),
                doc.get("category", "general"),
                doc.get("memory_type", "long_term"),
            )


def _scrub(text: str) -> str:
    return _HALLUCINATED_USERNAME_RE.sub("Unknown", text)


def main():
    ap = argparse.ArgumentParser(description="Clean hallucinated usernames from memory store.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="just show what would change (default)")
    g.add_argument("--delete", action="store_true", help="delete every memory that matches")
    g.add_argument("--scrub", action="store_true", help="rewrite the bad token to 'Unknown' instead of deleting")
    args = ap.parse_args()
    if not (args.delete or args.scrub):
        args.dry_run = True

    logger.info(f"backend: {memory_system.backend}, rag: {memory_system.rag_provider if memory_system.rag_enabled else 'off'}")
    logger.info(f"mode: {'delete' if args.delete else 'scrub' if args.scrub else 'dry-run'}")

    offenders = []
    for key, content, category, mem_type in _iter_all_memories():
        key_hit = bool(_HALLUCINATED_USERNAME_RE.search(key or ""))
        content_hit = bool(_HALLUCINATED_USERNAME_RE.search(content or ""))
        if key_hit or content_hit:
            offenders.append((key, content, category, mem_type, key_hit, content_hit))

    if not offenders:
        logger.info("no polluted memories found, nothing to do")
        return

    logger.info(f"found {len(offenders)} polluted memories")
    for key, content, _cat, _mt, kh, ch in offenders:
        flags = ("K" if kh else "-") + ("C" if ch else "-")
        snip = (content or "")[:140].replace("\n", " ")
        logger.info(f"  [{flags}] {key}: {snip}")

    if args.dry_run:
        logger.info("dry-run, no changes made. use --delete or --scrub to actually fix.")
        return

    deleted = 0
    scrubbed = 0
    failed = 0
    for key, content, category, mem_type, _kh, _ch in offenders:
        try:
            if args.delete:
                res = memory_system.delete(key)
                if res.get("success"):
                    deleted += 1
                else:
                    failed += 1
                    logger.warning(f"delete failed for {key}: {res.get('message')}")
            else:
                new_content = _scrub(content or "")
                if new_content == content:
                    continue
                res = memory_system.update(key, content=new_content)
                if res.get("success"):
                    scrubbed += 1
                else:
                    # update path now rejects content that still matches the pattern,
                    # so if scrub somehow left a hit we fall back to delete to be safe.
                    logger.warning(f"update failed for {key}: {res.get('message')}, deleting instead")
                    res2 = memory_system.delete(key)
                    if res2.get("success"):
                        deleted += 1
                    else:
                        failed += 1
        except Exception as e:
            failed += 1
            logger.error(f"error on {key}: {e}")

    logger.info(f"done. deleted={deleted} scrubbed={scrubbed} failed={failed}")


if __name__ == "__main__":
    main()
