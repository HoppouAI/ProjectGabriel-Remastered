"""
ProjectGabriel - Memory Migration Utility
Migrate memories between SQLite, MongoDB, and ChromaDB backends.
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yaml
except ImportError:
    yaml = None

try:
    from colorama import just_fix_windows_console
    just_fix_windows_console()
except ImportError:
    pass

# ── Colors ──
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
WHITE = "\033[97m"
MAGENTA = "\033[95m"
GRAY = "\033[90m"


def banner():
    w = 46
    print()
    print(f"  {CYAN}{'=' * w}{RST}")
    print(f"  {CYAN}|{RST}{WHITE}{BOLD}{'  Memory Migration Utility':^{w-2}}{RST}{CYAN}|{RST}")
    print(f"  {CYAN}|{RST}{DIM}{'ProjectGabriel':^{w-2}}{RST}{CYAN}|{RST}")
    print(f"  {CYAN}{'=' * w}{RST}")
    print()


def info(msg):
    print(f"  {GREEN}>{RST} {msg}")


def warn(msg):
    print(f"  {YELLOW}!{RST} {msg}")


def error(msg):
    print(f"  {RED}x{RST} {msg}")


def dim(msg):
    print(f"  {DIM}{msg}{RST}")


def progress_bar(current, total, prefix="", width=30):
    """inline progress bar that overwrites itself"""
    pct = current / total if total > 0 else 1.0
    filled = int(width * pct)
    bar = f"{'█' * filled}{'░' * (width - filled)}"
    sys.stdout.write(f"\r  {CYAN}>{RST} {prefix} {bar} {current}/{total}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def prompt_choice(question, options):
    """Show numbered menu, return selected option dict."""
    print(f"\n  {WHITE}{BOLD}{question}{RST}")
    for i, opt in enumerate(options, 1):
        label = opt["label"]
        desc = opt.get("desc", "")
        desc_str = f" {DIM}({desc}){RST}" if desc else ""
        print(f"    {CYAN}{i}.{RST} {label}{desc_str}")

    while True:
        try:
            raw = input(f"\n  {MAGENTA}>{RST} Pick a number [1-{len(options)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                print()
                return options[idx]
        except (ValueError, EOFError):
            pass
        error(f"Enter a number between 1 and {len(options)}")


def confirm(msg):
    """yes/no confirmation"""
    try:
        ans = input(f"  {YELLOW}?{RST} {msg} [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    except EOFError:
        return False


# ── Config loading ──
def load_config():
    cfg_path = Path("config.yml")
    if not cfg_path.exists():
        # try from project root
        cfg_path = Path(__file__).resolve().parent.parent / "config.yml"
    if cfg_path.exists() and yaml:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("memory", {})
    return {}


# ── Backend connections ──
def connect_sqlite(path):
    """Connect to SQLite db and return (conn, count)."""
    p = Path(path)
    if not p.exists():
        return None, 0
    conn = sqlite3.connect(str(p), check_same_thread=False)
    try:
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    except Exception:
        count = 0
    return conn, count


def connect_mongo(uri, db_name, collection_name):
    """Connect to MongoDB, returns (collection, count)."""
    try:
        from pymongo import MongoClient
    except ImportError:
        error("pymongo not installed. Run: uv pip install pymongo")
        return None, 0
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        coll = client[db_name][collection_name]
        count = coll.count_documents({})
        return coll, count
    except Exception as e:
        error(f"MongoDB connection failed: {e}")
        return None, 0


def connect_chroma(chroma_dir, create=False):
    """Connect to ChromaDB, returns (collection, count)."""
    try:
        import chromadb
    except ImportError:
        error("chromadb not installed. Run: uv pip install chromadb")
        return None, 0

    p = Path(chroma_dir)
    if not p.exists() and not create:
        return None, 0
    p.mkdir(parents=True, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=str(p))
        coll = client.get_or_create_collection(name="memories", metadata={"hnsw:space": "cosine"})
        count = coll.count()
        return coll, count
    except Exception as e:
        error(f"ChromaDB connection failed: {e}")
        return None, 0


# ── Read all memories from source ──
def read_sqlite(conn):
    """Read all memories from SQLite. Returns list of dicts."""
    cur = conn.execute(
        "SELECT key, content, category, memory_type, tags_json, content_hash, "
        "created_at, updated_at, access_count FROM memories"
    )
    rows = cur.fetchall()
    memories = []
    for r in rows:
        memories.append({
            "key": r[0],
            "content": r[1],
            "category": r[2],
            "memory_type": r[3],
            "tags": json.loads(r[4] or "[]"),
            "content_hash": r[5],
            "created_at": r[6],
            "updated_at": r[7],
            "access_count": r[8] or 0,
        })
    return memories


def read_mongo(collection):
    """Read all memories from MongoDB. Returns list of dicts."""
    memories = []
    for doc in collection.find({}):
        created = doc.get("created_at")
        if hasattr(created, "isoformat"):
            created = created.isoformat()
        updated = doc.get("updated_at")
        if hasattr(updated, "isoformat"):
            updated = updated.isoformat()

        memories.append({
            "key": doc.get("key"),
            "content": doc.get("content", ""),
            "category": doc.get("category", "general"),
            "memory_type": doc.get("memory_type", "long_term"),
            "tags": doc.get("tags", []),
            "content_hash": doc.get("content_hash", ""),
            "created_at": str(created) if created else None,
            "updated_at": str(updated) if updated else None,
            "access_count": doc.get("access_count", 0),
        })
    return memories


def read_chroma(collection):
    """Read all memories from ChromaDB. Returns list of dicts."""
    result = collection.get(include=["metadatas", "documents"])
    memories = []
    if result and result.get("ids"):
        for doc_id, meta, doc in zip(result["ids"], result["metadatas"], result["documents"]):
            memories.append({
                "key": doc_id,
                "content": doc or "",
                "category": meta.get("category", "general"),
                "memory_type": meta.get("memory_type", "long_term"),
                "tags": json.loads(meta.get("tags_json", "[]")),
                "content_hash": "",
                "created_at": meta.get("created_at"),
                "updated_at": None,
                "access_count": int(meta.get("access_count", 0)),
            })
    return memories


# ── Write memories to destination ──
def write_sqlite(conn, memories):
    """Write memories to SQLite."""
    import hashlib
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            memory_type TEXT NOT NULL DEFAULT 'long_term',
            tags_json TEXT DEFAULT '[]',
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            access_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON memories(content_hash)")

    written = 0
    skipped = 0
    for i, mem in enumerate(memories):
        content_hash = mem.get("content_hash") or hashlib.md5(mem["content"].encode()).hexdigest()[:16]
        created = mem.get("created_at") or datetime.utcnow().isoformat()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (key, content, category, memory_type, tags_json, content_hash, created_at, updated_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mem["key"], mem["content"], mem.get("category", "general"),
                mem.get("memory_type", "long_term"),
                json.dumps(mem.get("tags", []), ensure_ascii=False),
                content_hash, created, mem.get("updated_at"),
                mem.get("access_count", 0),
            ))
            written += 1
        except Exception:
            skipped += 1
        progress_bar(i + 1, len(memories), "Writing to SQLite")

    conn.commit()
    return written, skipped


def write_mongo(collection, memories):
    """Write memories to MongoDB."""
    import hashlib
    from datetime import datetime as dt
    written = 0
    skipped = 0
    for i, mem in enumerate(memories):
        content_hash = mem.get("content_hash") or hashlib.md5(mem["content"].encode()).hexdigest()[:16]
        now = dt.utcnow()
        try:
            created = mem.get("created_at")
            if isinstance(created, str):
                try:
                    created = dt.fromisoformat(created)
                except (ValueError, TypeError):
                    created = now

            collection.update_one(
                {"key": mem["key"]},
                {
                    "$set": {
                        "content": mem["content"],
                        "category": mem.get("category", "general"),
                        "memory_type": mem.get("memory_type", "long_term"),
                        "tags": mem.get("tags", []),
                        "content_hash": content_hash,
                        "updated_at": now,
                    },
                    "$setOnInsert": {
                        "created_at": created or now,
                        "access_count": mem.get("access_count", 0),
                    },
                },
                upsert=True,
            )
            written += 1
        except Exception:
            skipped += 1
        progress_bar(i + 1, len(memories), "Writing to MongoDB")

    return written, skipped


def write_chroma(collection, memories, lm_studio_url=None, model_name=None):
    """Write memories to ChromaDB with embeddings from LM Studio."""
    if not lm_studio_url or not model_name:
        error("LM Studio URL and model name required for ChromaDB migration")
        return 0, len(memories)

    try:
        import httpx
    except ImportError:
        error("httpx not installed. Run: uv pip install httpx")
        return 0, len(memories)

    client = httpx.Client(timeout=120)

    # test connectivity first
    info("Testing LM Studio connection...")
    try:
        resp = client.get(f"{lm_studio_url}/v1/models")
        resp.raise_for_status()
        info(f"Connected to LM Studio at {lm_studio_url}")
    except Exception as e:
        error(f"Can't reach LM Studio at {lm_studio_url}: {e}")
        error("Make sure LM Studio is running with the embedding model loaded")
        client.close()
        return 0, len(memories)

    written = 0
    skipped = 0
    batch_size = 32

    for i in range(0, len(memories), batch_size):
        batch = memories[i:i + batch_size]
        texts = [f"{m.get('category', 'general')}: {m.get('content', '')}" for m in batch]

        try:
            resp = client.post(
                f"{lm_studio_url}/v1/embeddings",
                json={"model": model_name, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
        except Exception as e:
            warn(f"Embedding batch failed at index {i}: {e}")
            skipped += len(batch)
            progress_bar(min(i + batch_size, len(memories)), len(memories), "Writing to ChromaDB")
            continue

        ids, embeds, docs, metas = [], [], [], []
        for mem, emb in zip(batch, embeddings):
            ids.append(mem["key"])
            embeds.append(emb)
            docs.append(mem.get("content", ""))
            metas.append({
                "category": mem.get("category", "general"),
                "memory_type": mem.get("memory_type", "long_term"),
                "tags_json": json.dumps(mem.get("tags", []), ensure_ascii=False),
                "created_at": str(mem.get("created_at", "")),
                "access_count": int(mem.get("access_count", 0)),
            })

        try:
            collection.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
            written += len(ids)
        except Exception as e:
            warn(f"ChromaDB upsert failed: {e}")
            skipped += len(ids)

        progress_bar(min(i + batch_size, len(memories)), len(memories), "Writing to ChromaDB")

    client.close()
    return written, skipped


# ── Main ──
def main():
    os.chdir(Path(__file__).resolve().parent.parent)  # project root

    banner()

    cfg = load_config()

    # ── Pick source ──
    source = prompt_choice("Where are you migrating FROM?", [
        {"label": "SQLite", "value": "sqlite", "desc": "local .sqlite file"},
        {"label": "MongoDB", "value": "mongo", "desc": "MongoDB Atlas cloud"},
        {"label": "ChromaDB", "value": "chroma", "desc": "local ChromaDB vector DB"},
    ])

    # ── Pick destination ──
    dest_options = [
        {"label": "SQLite", "value": "sqlite", "desc": "local .sqlite file"},
        {"label": "MongoDB", "value": "mongo", "desc": "MongoDB Atlas cloud"},
        {"label": "ChromaDB", "value": "chroma", "desc": "local ChromaDB vector DB (needs LM Studio)"},
    ]
    # filter out same-as-source
    dest_options = [o for o in dest_options if o["value"] != source["value"]]
    dest = prompt_choice("Where are you migrating TO?", dest_options)

    src_type = source["value"]
    dst_type = dest["value"]

    info(f"Migration: {src_type.upper()} -> {dst_type.upper()}")
    print()

    # ── Connect source ──
    info("Connecting to source...")
    memories = []

    if src_type == "sqlite":
        default_path = cfg.get("sqlite_path", "gabriel_memories.sqlite")
        dim(f"  Default path from config: {default_path}")
        dim("  Press Enter to use the default")
        custom = input(f"  {MAGENTA}>{RST} SQLite path [{default_path}]: ").strip()
        path = custom if custom else default_path
        conn, count = connect_sqlite(path)
        if conn is None:
            error(f"SQLite file not found: {path}")
            return
        info(f"Connected to SQLite: {count} memories found")
        memories = read_sqlite(conn)
        conn.close()

    elif src_type == "mongo":
        uri = cfg.get("mongo_uri", "")
        db_name = cfg.get("mongo_db", "gabriel")
        coll_name = cfg.get("mongo_collection", "memories")
        if not uri:
            uri = input(f"  {MAGENTA}>{RST} MongoDB URI: ").strip()
        else:
            dim(f"  Using URI from config.yml")
        coll, count = connect_mongo(uri, db_name, coll_name)
        if coll is None:
            return
        info(f"Connected to MongoDB: {count} memories found")
        memories = read_mongo(coll)

    elif src_type == "chroma":
        default_dir = cfg.get("chroma_dir", "gabriel_chroma_db")
        dim("  Press Enter to use the default")
        custom = input(f"  {MAGENTA}>{RST} ChromaDB directory [{default_dir}]: ").strip()
        chroma_dir = custom if custom else default_dir
        coll, count = connect_chroma(chroma_dir)
        if coll is None:
            error(f"ChromaDB not found at: {chroma_dir}")
            return
        info(f"Connected to ChromaDB: {count} memories found")
        memories = read_chroma(coll)

    if not memories:
        warn("No memories found in source. Nothing to migrate.")
        return

    # filter out memories with missing keys
    before = len(memories)
    memories = [m for m in memories if m.get("key")]
    if len(memories) < before:
        warn(f"Skipped {before - len(memories)} memories with missing keys")

    info(f"Read {len(memories)} memories from {src_type}")
    print()

    # ── Connect destination ──
    info("Setting up destination...")

    if dst_type == "sqlite":
        default_path = cfg.get("sqlite_path", "gabriel_memories.sqlite")
        dim("  Press Enter to use the default")
        custom = input(f"  {MAGENTA}>{RST} SQLite output path [{default_path}]: ").strip()
        path = custom if custom else default_path
        if Path(path).exists():
            _, existing = connect_sqlite(path)
            warn(f"SQLite file already exists with {existing} memories")
            if not confirm("Continue? Existing keys will be overwritten."):
                info("Cancelled.")
                return
        conn = sqlite3.connect(str(path), check_same_thread=False)
        print()
        written, skipped = write_sqlite(conn, memories)
        conn.close()

    elif dst_type == "mongo":
        uri = cfg.get("mongo_uri", "")
        db_name = cfg.get("mongo_db", "gabriel")
        coll_name = cfg.get("mongo_collection", "memories")
        if not uri:
            uri = input(f"  {MAGENTA}>{RST} MongoDB URI: ").strip()
        else:
            dim(f"  Using URI from config.yml")
        coll, existing = connect_mongo(uri, db_name, coll_name)
        if coll is None:
            return
        if existing > 0:
            warn(f"Collection already has {existing} memories")
            if not confirm("Continue? Existing keys will be overwritten."):
                info("Cancelled.")
                return
        print()
        written, skipped = write_mongo(coll, memories)

    elif dst_type == "chroma":
        default_dir = cfg.get("chroma_dir", "gabriel_chroma_db")
        dim("  Press Enter on each to use the recommended default")
        custom = input(f"  {MAGENTA}>{RST} ChromaDB directory [{default_dir}]: ").strip()
        chroma_dir = custom if custom else default_dir

        default_url = cfg.get("lm_studio_url", "http://localhost:1234")
        custom_url = input(f"  {MAGENTA}>{RST} LM Studio URL [{default_url}]: ").strip()
        lm_url = custom_url if custom_url else default_url

        default_model = cfg.get("local_embedding_model", "text-embedding-embeddinggemma-300m-qat")
        custom_model = input(f"  {MAGENTA}>{RST} Embedding model [{default_model}]: ").strip()
        model = custom_model if custom_model else default_model

        coll, existing = connect_chroma(chroma_dir, create=True)
        if coll is None:
            return
        if existing > 0:
            warn(f"ChromaDB already has {existing} entries")
            if not confirm("Continue? Existing keys will be overwritten."):
                info("Cancelled.")
                return
        print()
        written, skipped = write_chroma(coll, memories, lm_url, model)

    # ── Summary ──
    print()
    print(f"  {CYAN}{'=' * 36}{RST}")
    print(f"  {GREEN}{BOLD}Migration complete!{RST}")
    print(f"  {WHITE}Written:{RST} {written}")
    if skipped:
        print(f"  {YELLOW}Skipped:{RST} {skipped}")
    print(f"  {CYAN}{'=' * 36}{RST}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n  {YELLOW}Cancelled by user.{RST}")
        sys.exit(0)
