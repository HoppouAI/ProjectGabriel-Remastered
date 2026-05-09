"""One-shot importer for the OLD ProjectGabriel memory.db schema.

Old schema: memory(id, key, value, memory_type, tags, created_at, last_accessed, access_count)
New schema: memories(key, content, category, memory_type, tags, content_hash, created_at, ...)

Imports every row from the old sqlite db into whatever backend the current
MemorySystem is configured for (mongo or sqlite). Old keys are prefixed with
`legacy_` so nothing in the current memory gets overwritten. Original
created_at timestamps are preserved by patching the row after save().

Run from project root: .venv\\Scripts\\python.exe scripts\\import_old_memories.py
Default source: C:/Users/myhom/Documents/GitHub/ProjectGabriel/memory.db
Override with --source <path>.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import MemorySystem  # noqa: E402

DEFAULT_SOURCE = "C:/Users/myhom/Documents/GitHub/ProjectGabriel/memory.db"
LEGACY_PREFIX = "legacy_"
LEGACY_TAG = "legacy_import"


def parse_iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def load_old_rows(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"old db not found: {db_path}")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT key, value, memory_type, tags, created_at, last_accessed, access_count FROM memory"
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def normalize_tags(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(t) for t in parsed]
    return []


def patch_timestamp(mem: MemorySystem, key: str, original_created):
    """Force the row's created_at back to the original timestamp from old db."""
    if original_created is None:
        return
    if mem.backend == "sqlite":
        try:
            iso = original_created.isoformat()
            with mem._sqlite_lock:
                mem.sqlite_conn.execute(
                    "UPDATE memories SET created_at = ?, updated_at = ? WHERE key = ?",
                    (iso, iso, key),
                )
                mem.sqlite_conn.commit()
        except Exception as e:
            print(f"  ! failed to patch created_at for {key}: {e}")
    else:
        try:
            mem.collection.update_one(
                {"key": key},
                {"$set": {"created_at": original_created, "updated_at": original_created}},
            )
        except Exception as e:
            print(f"  ! failed to patch created_at for {key}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="path to old memory.db")
    ap.add_argument("--prefix", default=LEGACY_PREFIX, help="prefix for imported keys (default: legacy_)")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, dont write")
    ap.add_argument("--no-prefix", action="store_true", help="dont prefix keys (will overwrite same-named keys!)")
    args = ap.parse_args()

    src = Path(args.source)
    print(f"source : {src}")
    rows = load_old_rows(src)
    print(f"found  : {len(rows)} rows in old memory table")
    print()

    if args.dry_run:
        print("=== DRY RUN, no writes ===")
        for r in rows:
            new_key = r["key"] if args.no_prefix else f"{args.prefix}{r['key']}"
            print(f"  {new_key:40s} | {r['memory_type']:12s} | {r['value']}")
        return

    mem = MemorySystem()
    if not mem.is_available():
        print("memory backend is not available, aborting")
        sys.exit(1)
    print(f"backend: {mem.backend}")
    if mem.backend == "mongo":
        print(f"db     : {mem.mongo_db}/{mem.mongo_collection_name}")
    else:
        print(f"sqlite : {mem.sqlite_path}")
    print()

    saved = 0
    skipped = 0
    failed = 0
    dupes = 0
    seen_keys: set[str] = set()
    for r in rows:
        old_key = r["key"]
        new_key = old_key if args.no_prefix else f"{args.prefix}{old_key}"
        if new_key in seen_keys:
            dupes += 1
            print(f"  ~ dupe key, skipping second occurrence: {new_key} (value was: {r['value']!r})")
            continue
        seen_keys.add(new_key)
        # build content from key + value so context is preserved (old format was key:value pairs)
        content = f"{old_key.replace('_', ' ')}: {r['value']}"
        memory_type = r["memory_type"] or "long_term"
        tags = normalize_tags(r["tags"])
        if LEGACY_TAG not in tags:
            tags.append(LEGACY_TAG)
        original_created = parse_iso(r["created_at"])

        result = mem.save(
            key=new_key,
            content=content,
            category="legacy",
            memory_type=memory_type,
            tags=tags,
        )
        if result.get("success"):
            patch_timestamp(mem, new_key, original_created)
            saved += 1
            print(f"  + {new_key}")
        else:
            msg = result.get("message", "unknown")
            if "rejected" in msg.lower():
                skipped += 1
                print(f"  - skipped {new_key}: {msg}")
            else:
                failed += 1
                print(f"  x failed  {new_key}: {msg}")

    print()
    print(f"done. saved={saved} dupes_skipped={dupes} rejected={skipped} failed={failed}")


if __name__ == "__main__":
    main()
