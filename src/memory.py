"""
Persistent Memory System for ProjectGabriel
Supports MongoDB (primary) and SQLite (fallback) backends.

Memory Types:
- long_term: Permanent memories
- short_term: Auto-deleted after 7 days
- quick_note: Auto-deleted after 6 hours

Usage:
    from src.memory import memory_system, get_memory_tools, handle_memory_function_call
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.collection import Collection, ReturnDocument
    from pymongo.errors import PyMongoError
    MONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    Collection = None
    ReturnDocument = None
    PyMongoError = Exception
    ASCENDING = 1
    DESCENDING = -1
    MONGO_AVAILABLE = False

logger = logging.getLogger(__name__)

MEMORY_TYPE_LONG_TERM = "long_term"
MEMORY_TYPE_SHORT_TERM = "short_term"
MEMORY_TYPE_QUICK_NOTE = "quick_note"


def _load_config() -> Dict[str, Any]:
    """Load config.yml and return memory section."""
    config_path = Path("config.yml")
    if not config_path.exists() or yaml is None:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("memory", {}) if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug(f"Config load error: {e}")
        return {}


def _hash_content(text: str) -> str:
    """Generate SHA256 hash for deduplication."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


class MemorySystem:
    """Unified memory storage with MongoDB and SQLite backends."""

    def __init__(self):
        self.config = _load_config()
        self.backend = self.config.get("backend", "sqlite").lower()
        
        # MongoDB settings
        self.mongo_uri = os.environ.get("GABRIEL_MONGO_URI", self.config.get("mongo_uri", ""))
        self.mongo_db = os.environ.get("GABRIEL_MONGO_DB", self.config.get("mongo_db", "gabriel"))
        self.mongo_collection_name = os.environ.get("GABRIEL_MONGO_COLLECTION", self.config.get("mongo_collection", "memories"))
        
        # SQLite settings
        self.sqlite_path = self.config.get("sqlite_path", "gabriel_memories.sqlite")
        
        # TTL settings
        self.quick_note_ttl_hours = float(self.config.get("quick_note_ttl_hours", 6))
        self.short_term_ttl_days = float(self.config.get("short_term_ttl_days", 7))
        self.note_min_interval = float(self.config.get("note_min_interval_seconds", 120))
        self.dedupe_window = float(self.config.get("dedupe_window_seconds", 300))
        
        # State
        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self._sqlite_lock = threading.RLock()
        self._cleanup_running = False
        self._cleanup_thread: Optional[threading.Thread] = None
        self._note_last_ts: float = 0
        self._note_last_hash: str = ""

        self._connect()
        if self.is_available():
            self._start_cleanup_thread()

    def _connect(self):
        """Connect to storage backend."""
        if self.backend == "mongo" and MONGO_AVAILABLE and self.mongo_uri:
            self._connect_mongo()
        else:
            self._connect_sqlite()

    def _connect_mongo(self):
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")
            self.collection = self.client[self.mongo_db][self.mongo_collection_name]
            self._init_mongo_indexes()
            self.backend = "mongo"
            logger.info(f"Memory connected to MongoDB: {self.mongo_db}.{self.mongo_collection_name}")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}, falling back to SQLite")
            self.collection = None
            self._connect_sqlite()

    def _init_mongo_indexes(self):
        """Create MongoDB indexes."""
        if self.collection is None:
            return
        try:
            # Use same index names as old system to avoid conflicts
            self.collection.create_index([("key", ASCENDING)], unique=True, name="idx_key_unique")
            self.collection.create_index([("category", ASCENDING)], name="idx_category")
            self.collection.create_index([("memory_type", ASCENDING)], name="idx_memory_type")
            self.collection.create_index([("created_at", DESCENDING)], name="idx_created_at")
            self.collection.create_index([("memory_type", ASCENDING), ("created_at", DESCENDING)], name="idx_memory_type_created")
            self.collection.create_index([("content_hash", ASCENDING)], name="idx_content_hash")
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")

    def _connect_sqlite(self):
        """Connect to SQLite."""
        try:
            path = Path(self.sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._sqlite_lock:
                self.sqlite_conn = sqlite3.connect(str(path), check_same_thread=False)
                self.sqlite_conn.row_factory = sqlite3.Row
                self.sqlite_conn.execute("PRAGMA journal_mode=WAL")
                self._init_sqlite_tables()
            
            self.backend = "sqlite"
            logger.info(f"Memory connected to SQLite: {path}")
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            self.sqlite_conn = None

    def _init_sqlite_tables(self):
        """Create SQLite tables."""
        if not self.sqlite_conn:
            return
        with self._sqlite_lock:
            self.sqlite_conn.execute("""
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
            self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
            self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
            self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")
            self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON memories(content_hash)")
            self.sqlite_conn.commit()

    def is_available(self) -> bool:
        """Check if storage is ready."""
        if self.backend == "sqlite":
            return self.sqlite_conn is not None
        return self.collection is not None

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_running:
            return
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("Memory cleanup thread started")

    def _cleanup_loop(self):
        """Background cleanup of expired memories."""
        while self._cleanup_running:
            try:
                if self.is_available():
                    self.cleanup_expired()
                time.sleep(600)  # Every 10 minutes
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(60)

    def cleanup_expired(self) -> Dict[str, int]:
        """Delete expired short-term and quick-note memories."""
        if not self.is_available():
            return {"quick_notes": 0, "short_term": 0}

        now = datetime.utcnow()
        quick_cutoff = now - timedelta(hours=self.quick_note_ttl_hours)
        short_cutoff = now - timedelta(days=self.short_term_ttl_days)
        
        quick_deleted = 0
        short_deleted = 0

        try:
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    c1 = self.sqlite_conn.execute(
                        "DELETE FROM memories WHERE memory_type = ? AND created_at < ? AND tags_json NOT LIKE ?",
                        (MEMORY_TYPE_QUICK_NOTE, quick_cutoff.isoformat(), '%"pinned"%')
                    )
                    quick_deleted = c1.rowcount or 0
                    
                    c2 = self.sqlite_conn.execute(
                        "DELETE FROM memories WHERE memory_type = ? AND created_at < ? AND tags_json NOT LIKE ?",
                        (MEMORY_TYPE_SHORT_TERM, short_cutoff.isoformat(), '%"pinned"%')
                    )
                    short_deleted = c2.rowcount or 0
                    self.sqlite_conn.commit()
            else:
                r1 = self.collection.delete_many({
                    "memory_type": MEMORY_TYPE_QUICK_NOTE,
                    "created_at": {"$lt": quick_cutoff},
                    "tags": {"$nin": ["pinned"]}
                })
                quick_deleted = r1.deleted_count if r1 else 0
                
                r2 = self.collection.delete_many({
                    "memory_type": MEMORY_TYPE_SHORT_TERM,
                    "created_at": {"$lt": short_cutoff},
                    "tags": {"$nin": ["pinned"]}
                })
                short_deleted = r2.deleted_count if r2 else 0

            if quick_deleted or short_deleted:
                logger.info(f"Cleaned up {quick_deleted} quick notes, {short_deleted} short-term memories")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

        return {"quick_notes": quick_deleted, "short_term": short_deleted}

    def save(
        self,
        key: str,
        content: str,
        category: str = "general",
        memory_type: str = MEMORY_TYPE_LONG_TERM,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Save a memory."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        valid_types = [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_QUICK_NOTE]
        if memory_type not in valid_types:
            return {"success": False, "message": f"Invalid memory type: {memory_type}"}

        tags_list = list(tags) if tags else []
        content_hash = _hash_content(content)
        now = datetime.utcnow()

        try:
            if self.backend == "sqlite":
                tags_json = json.dumps(tags_list, ensure_ascii=False)
                now_iso = now.isoformat()
                
                with self._sqlite_lock:
                    self.sqlite_conn.execute("""
                        INSERT INTO memories (key, content, category, memory_type, tags_json, content_hash, created_at, updated_at, access_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                        ON CONFLICT(key) DO UPDATE SET
                            content = excluded.content,
                            category = excluded.category,
                            memory_type = excluded.memory_type,
                            tags_json = excluded.tags_json,
                            content_hash = excluded.content_hash,
                            updated_at = excluded.updated_at
                    """, (key, content, category, memory_type, tags_json, content_hash, now_iso, now_iso))
                    self.sqlite_conn.commit()
            else:
                self.collection.find_one_and_update(
                    {"key": key},
                    {
                        "$set": {
                            "content": content,
                            "category": category,
                            "memory_type": memory_type,
                            "tags": tags_list,
                            "content_hash": content_hash,
                            "updated_at": now,
                        },
                        "$setOnInsert": {"created_at": now, "access_count": 0},
                    },
                    upsert=True,
                    return_document=ReturnDocument.AFTER,
                )

            logger.info(f"Memory saved: {key} ({memory_type})")
            return {"success": True, "key": key, "memory_type": memory_type}

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return {"success": False, "message": str(e)}

    def read(self, key: str) -> Dict[str, Any]:
        """Read a memory by key."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    row = self.sqlite_conn.execute(
                        "SELECT * FROM memories WHERE key = ?", (key,)
                    ).fetchone()
                    
                    if not row:
                        return {"success": False, "message": f"Memory '{key}' not found"}
                    
                    self.sqlite_conn.execute(
                        "UPDATE memories SET access_count = access_count + 1 WHERE key = ?", (key,)
                    )
                    self.sqlite_conn.commit()
                    
                    tags = json.loads(row["tags_json"]) if row["tags_json"] else []
                    memory = {
                        "key": row["key"],
                        "content": row["content"],
                        "category": row["category"],
                        "memory_type": row["memory_type"],
                        "tags": tags,
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "access_count": row["access_count"] + 1,
                    }
            else:
                doc = self.collection.find_one_and_update(
                    {"key": key},
                    {"$inc": {"access_count": 1}},
                    return_document=ReturnDocument.AFTER,
                )
                if not doc:
                    return {"success": False, "message": f"Memory '{key}' not found"}
                
                memory = self._format_doc(doc)

            logger.info(f"Memory read: {key}")
            return {"success": True, "memory": memory}

        except Exception as e:
            logger.error(f"Read failed: {e}")
            return {"success": False, "message": str(e)}

    def update(
        self,
        key: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        if memory_type and memory_type not in [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_QUICK_NOTE]:
            return {"success": False, "message": f"Invalid memory type: {memory_type}"}

        try:
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    row = self.sqlite_conn.execute("SELECT id FROM memories WHERE key = ?", (key,)).fetchone()
                    if not row:
                        return {"success": False, "message": f"Memory '{key}' not found"}

                    updates = []
                    values = []
                    if content is not None:
                        updates.extend(["content = ?", "content_hash = ?"])
                        values.extend([content, _hash_content(content)])
                    if category is not None:
                        updates.append("category = ?")
                        values.append(category)
                    if memory_type is not None:
                        updates.append("memory_type = ?")
                        values.append(memory_type)
                    if tags is not None:
                        updates.append("tags_json = ?")
                        values.append(json.dumps(tags, ensure_ascii=False))

                    if updates:
                        updates.append("updated_at = ?")
                        values.append(datetime.utcnow().isoformat())
                        values.append(key)
                        
                        self.sqlite_conn.execute(
                            f"UPDATE memories SET {', '.join(updates)} WHERE key = ?", values
                        )
                        self.sqlite_conn.commit()
            else:
                if not self.collection.find_one({"key": key}):
                    return {"success": False, "message": f"Memory '{key}' not found"}

                updates = {"updated_at": datetime.utcnow()}
                if content is not None:
                    updates["content"] = content
                    updates["content_hash"] = _hash_content(content)
                if category is not None:
                    updates["category"] = category
                if memory_type is not None:
                    updates["memory_type"] = memory_type
                if tags is not None:
                    updates["tags"] = tags

                self.collection.update_one({"key": key}, {"$set": updates})

            logger.info(f"Memory updated: {key}")
            return {"success": True, "key": key}

        except Exception as e:
            logger.error(f"Update failed: {e}")
            return {"success": False, "message": str(e)}

    def delete(self, key: str) -> Dict[str, Any]:
        """Delete a memory."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    cur = self.sqlite_conn.execute("DELETE FROM memories WHERE key = ?", (key,))
                    deleted = cur.rowcount or 0
                    self.sqlite_conn.commit()
            else:
                res = self.collection.delete_one({"key": key})
                deleted = res.deleted_count if res else 0

            if deleted:
                logger.info(f"Memory deleted: {key}")
                return {"success": True, "key": key}
            return {"success": False, "message": f"Memory '{key}' not found"}

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return {"success": False, "message": str(e)}

    def list_memories(
        self,
        category: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """List memories with optional filters."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            memories = []
            
            if self.backend == "sqlite":
                where = []
                params = []
                if category:
                    where.append("category = ?")
                    params.append(category)
                if memory_type:
                    where.append("memory_type = ?")
                    params.append(memory_type)

                where_sql = f" WHERE {' AND '.join(where)}" if where else ""
                params.append(limit)

                with self._sqlite_lock:
                    rows = self.sqlite_conn.execute(
                        f"SELECT * FROM memories{where_sql} ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?",
                        params
                    ).fetchall()

                for row in rows:
                    content = row["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": row["key"],
                        "content": content,
                        "category": row["category"],
                        "memory_type": row["memory_type"],
                        "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
                        "created_at": row["created_at"],
                        "access_count": row["access_count"],
                    })
            else:
                filters = {}
                if category:
                    filters["category"] = category
                if memory_type:
                    filters["memory_type"] = memory_type

                cursor = self.collection.find(filters).sort("updated_at", DESCENDING).limit(limit)
                for doc in cursor:
                    content = doc.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": doc.get("key"),
                        "content": content,
                        "category": doc.get("category", "general"),
                        "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
                        "tags": doc.get("tags", []),
                        "created_at": self._serialize_dt(doc.get("created_at")),
                        "access_count": doc.get("access_count", 0),
                    })

            return {"success": True, "memories": memories, "count": len(memories)}

        except Exception as e:
            logger.error(f"List failed: {e}")
            return {"success": False, "message": str(e)}

    def search(self, term: str, memory_type: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Search memories by content or key."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            memories = []
            
            if self.backend == "sqlite":
                like = f"%{term}%"
                where = ["(key LIKE ? OR content LIKE ?)"]
                params: List[Any] = [like, like]
                
                if memory_type:
                    where.append("memory_type = ?")
                    params.append(memory_type)
                
                params.append(limit)

                with self._sqlite_lock:
                    rows = self.sqlite_conn.execute(
                        f"SELECT * FROM memories WHERE {' AND '.join(where)} ORDER BY access_count DESC LIMIT ?",
                        params
                    ).fetchall()

                for row in rows:
                    content = row["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": row["key"],
                        "content": content,
                        "category": row["category"],
                        "memory_type": row["memory_type"],
                        "created_at": row["created_at"],
                    })
            else:
                import re
                regex = {"$regex": re.escape(term), "$options": "i"}
                query: Dict[str, Any] = {"$or": [{"key": regex}, {"content": regex}]}
                if memory_type:
                    query["memory_type"] = memory_type

                cursor = self.collection.find(query).sort("access_count", DESCENDING).limit(limit)
                for doc in cursor:
                    content = doc.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": doc.get("key"),
                        "content": content,
                        "category": doc.get("category", "general"),
                        "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
                        "created_at": self._serialize_dt(doc.get("created_at")),
                    })

            return {"success": True, "memories": memories, "count": len(memories), "term": term}

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"success": False, "message": str(e)}

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            counts = {"total": 0, "long_term": 0, "short_term": 0, "quick_note": 0}
            
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    rows = self.sqlite_conn.execute(
                        "SELECT memory_type, COUNT(1) as c FROM memories GROUP BY memory_type"
                    ).fetchall()
                for row in rows:
                    mt = row[0] or "unknown"
                    c = row[1] or 0
                    counts[mt] = c
                    counts["total"] += c
            else:
                pipeline = [{"$group": {"_id": "$memory_type", "count": {"$sum": 1}}}]
                for row in self.collection.aggregate(pipeline):
                    mt = row.get("_id") or "unknown"
                    c = row.get("count", 0)
                    counts[mt] = c
                    counts["total"] += c

            return {"success": True, "stats": counts}

        except Exception as e:
            logger.error(f"Stats failed: {e}")
            return {"success": False, "message": str(e)}

    def has_recent_duplicate(self, content_hash: str, window_seconds: float, types_list: Optional[List[str]] = None) -> bool:
        """Check if a similar memory was saved recently."""
        if not self.is_available():
            return False

        try:
            since = datetime.utcnow() - timedelta(seconds=window_seconds)
            
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    if types_list:
                        placeholders = ",".join(["?"] * len(types_list))
                        row = self.sqlite_conn.execute(
                            f"SELECT id FROM memories WHERE content_hash = ? AND created_at > ? AND memory_type IN ({placeholders}) LIMIT 1",
                            (content_hash, since.isoformat(), *types_list)
                        ).fetchone()
                    else:
                        row = self.sqlite_conn.execute(
                            "SELECT id FROM memories WHERE content_hash = ? AND created_at > ? LIMIT 1",
                            (content_hash, since.isoformat())
                        ).fetchone()
                return row is not None
            else:
                query: Dict[str, Any] = {"content_hash": content_hash, "created_at": {"$gt": since}}
                if types_list:
                    query["memory_type"] = {"$in": types_list}
                doc = self.collection.find_one(query, {"_id": 1})
                return doc is not None

        except Exception:
            return False

    def get_recent_for_prompt(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get memories for system prompt, prioritizing pinned and frequently accessed."""
        if not self.is_available():
            return []

        try:
            memories = []
            
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    # Prioritize: pinned first, then blend of access_count and recency
                    rows = self.sqlite_conn.execute(
                        "SELECT key, content, category, created_at, tags_json, access_count FROM memories "
                        "WHERE memory_type IN (?, ?) "
                        "ORDER BY "
                        "  CASE WHEN tags_json LIKE '%\"pinned\"%' THEN 0 ELSE 1 END, "
                        "  (access_count * 0.4 + (julianday(created_at) - julianday('2024-01-01')) * 0.6) DESC "
                        "LIMIT ?",
                        (MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, count)
                    ).fetchall()

                for row in rows:
                    memories.append({
                        "key": row["key"],
                        "content": row["content"],
                        "category": row["category"],
                        "created_at": row["created_at"],
                        "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
                    })
            else:
                # MongoDB: two-phase fetch — pinned first, then scored
                pinned = list(self.collection.find(
                    {"memory_type": {"$in": [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM]}, "tags": "pinned"},
                    {"key": 1, "content": 1, "category": 1, "created_at": 1, "tags": 1, "access_count": 1}
                ).sort("created_at", DESCENDING).limit(count))

                remaining = count - len(pinned)
                pinned_keys = {doc["key"] for doc in pinned}
                others = []
                if remaining > 0:
                    others = list(self.collection.find(
                        {
                            "memory_type": {"$in": [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM]},
                            "key": {"$nin": list(pinned_keys)},
                        },
                        {"key": 1, "content": 1, "category": 1, "created_at": 1, "tags": 1, "access_count": 1}
                    ).sort([("access_count", DESCENDING), ("created_at", DESCENDING)]).limit(remaining))

                for doc in pinned + others:
                    memories.append({
                        "key": doc.get("key"),
                        "content": doc.get("content"),
                        "category": doc.get("category", "general"),
                        "created_at": self._serialize_dt(doc.get("created_at")),
                        "tags": doc.get("tags", []),
                    })

            return memories

        except Exception as e:
            logger.error(f"Get recent failed: {e}")
            return []

    def _format_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Format MongoDB document for response."""
        return {
            "key": doc.get("key"),
            "content": doc.get("content"),
            "category": doc.get("category", "general"),
            "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
            "tags": doc.get("tags", []),
            "created_at": self._serialize_dt(doc.get("created_at")),
            "updated_at": self._serialize_dt(doc.get("updated_at")),
            "access_count": doc.get("access_count", 0),
        }

    @staticmethod
    def _serialize_dt(value) -> Optional[str]:
        """Convert datetime to ISO string."""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return None

    def close(self):
        """Close connections."""
        self._cleanup_running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2)
        
        if self.sqlite_conn:
            try:
                self.sqlite_conn.close()
            except Exception:
                pass
        
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass


# Global instance
memory_system = MemorySystem()


# Tool declarations for Gemini Live
MEMORY_FUNCTION_DECLARATIONS = [
    {
        "name": "memory",
        "description": "Persistent memory system. Actions: save, read, update, delete, list, search, stats, cleanup, pin, promote. Memory types: 'long_term' (permanent), 'short_term' (7 days), 'quick_note' (6 hours).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "read", "update", "delete", "list", "search", "stats", "cleanup", "pin", "promote"],
                    "description": "Memory operation to perform"
                },
                "key": {
                    "type": "string",
                    "description": "Memory identifier (required for save/read/update/delete/pin/promote)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to store (required for save)"
                },
                "category": {
                    "type": "string",
                    "description": "Category (e.g., 'personal', 'work', 'facts')"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["long_term", "short_term", "quick_note"],
                    "description": "Memory persistence type"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for organization"
                },
                "search_term": {
                    "type": "string",
                    "description": "Search query (for search action)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 20)"
                },
                "new_type": {
                    "type": "string",
                    "enum": ["short_term", "long_term"],
                    "description": "Target type for promote action"
                },
                "pin": {
                    "type": "boolean",
                    "description": "Pin status for pin action (pinned memories won't be auto-deleted)"
                }
            },
            "required": ["action"]
        }
    }
]


def get_memory_tools():
    """Get memory tool declarations for Gemini Live."""
    return MEMORY_FUNCTION_DECLARATIONS


async def handle_memory_function_call(function_call) -> Dict[str, Any]:
    """Handle memory function calls from Gemini Live."""
    from google.genai import types
    
    args = dict(function_call.args) if function_call.args else {}
    action = args.get("action", "")
    
    # Support both camelCase (new) and snake_case (legacy) parameter names
    memory_type_raw = args.get("memoryType") or args.get("memory_type")
    memory_type = memory_type_raw or MEMORY_TYPE_LONG_TERM
    search_term = args.get("searchTerm") or args.get("search_term")
    new_type = args.get("newType") or args.get("new_type")
    
    # Parse tags - can be array (legacy) or comma-separated string (new)
    tags_raw = args.get("tags")
    if isinstance(tags_raw, str):
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
    elif isinstance(tags_raw, list):
        tags_list = tags_raw
    else:
        tags_list = None
    
    # Parse pin - can be boolean (legacy) or string "true"/"false" (new)
    pin_raw = args.get("pin")
    if isinstance(pin_raw, str):
        pin_val = pin_raw.lower() in ("true", "1", "yes")
    elif isinstance(pin_raw, bool):
        pin_val = pin_raw
    else:
        pin_val = True  # default
    
    try:
        result: Dict[str, Any]
        
        if action == "save":
            key = args.get("key")
            content = args.get("content")
            if not key or not content:
                result = {"result": "error", "message": "key and content required"}
            else:
                # Rate limit for quick notes
                mem_type = memory_type
                if key.startswith("note_") or mem_type == MEMORY_TYPE_QUICK_NOTE:
                    now = time.time()
                    content_hash = _hash_content(content)
                    
                    if now - memory_system._note_last_ts < memory_system.note_min_interval:
                        result = {"result": "ok", "skipped": True, "reason": "rate_limited"}
                    elif memory_system._note_last_hash == content_hash:
                        result = {"result": "ok", "skipped": True, "reason": "duplicate"}
                    elif memory_system.has_recent_duplicate(content_hash, memory_system.dedupe_window, [MEMORY_TYPE_QUICK_NOTE, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_LONG_TERM]):
                        result = {"result": "ok", "skipped": True, "reason": "duplicate_db"}
                    else:
                        res = memory_system.save(
                            key=key,
                            content=content,
                            category=args.get("category", "general"),
                            memory_type=mem_type if mem_type != MEMORY_TYPE_LONG_TERM else MEMORY_TYPE_QUICK_NOTE,
                            tags=tags_list if tags_list else ["quick_note"]
                        )
                        if res.get("success"):
                            memory_system._note_last_ts = now
                            memory_system._note_last_hash = content_hash
                        result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}
                else:
                    res = memory_system.save(
                        key=key,
                        content=content,
                        category=args.get("category", "general"),
                        memory_type=mem_type,
                        tags=tags_list
                    )
                    result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        elif action == "read":
            key = args.get("key")
            if not key:
                result = {"result": "error", "message": "key required"}
            else:
                res = memory_system.read(key)
                if res.get("success"):
                    result = {"result": "ok", "memory": res.get("memory")}
                else:
                    result = {"result": "error", "message": res.get("message")}

        elif action == "update":
            key = args.get("key")
            if not key:
                result = {"result": "error", "message": "key required"}
            else:
                res = memory_system.update(
                    key=key,
                    content=args.get("content"),
                    category=args.get("category"),
                    memory_type=memory_type_raw,
                    tags=tags_list
                )
                result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        elif action == "delete":
            key = args.get("key")
            if not key:
                result = {"result": "error", "message": "key required"}
            else:
                res = memory_system.delete(key)
                result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        elif action == "list":
            res = memory_system.list_memories(
                category=args.get("category"),
                memory_type=memory_type_raw,
                limit=args.get("limit", 50)
            )
            if res.get("success"):
                result = {"result": "ok", "memories": res.get("memories"), "count": res.get("count")}
            else:
                result = {"result": "error", "message": res.get("message")}

        elif action == "search":
            if not search_term:
                result = {"result": "error", "message": "searchTerm required"}
            else:
                res = memory_system.search(
                    term=search_term,
                    memory_type=memory_type_raw,
                    limit=args.get("limit", 20)
                )
                if res.get("success"):
                    result = {"result": "ok", "memories": res.get("memories"), "count": res.get("count")}
                else:
                    result = {"result": "error", "message": res.get("message")}

        elif action == "stats":
            res = memory_system.stats()
            if res.get("success"):
                result = {"result": "ok", "stats": res.get("stats")}
            else:
                result = {"result": "error", "message": res.get("message")}

        elif action == "cleanup":
            res = memory_system.cleanup_expired()
            result = {"result": "ok", "deleted": res}

        elif action == "pin":
            key = args.get("key")
            if not key:
                result = {"result": "error", "message": "key required"}
            else:
                read_res = memory_system.read(key)
                if not read_res.get("success"):
                    result = {"result": "error", "message": read_res.get("message")}
                else:
                    mem = read_res["memory"]
                    tags = mem.get("tags", [])
                    if pin_val and "pinned" not in tags:
                        tags.append("pinned")
                    elif not pin_val and "pinned" in tags:
                        tags = [t for t in tags if t != "pinned"]
                    res = memory_system.update(key=key, tags=tags)
                    result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        elif action == "promote":
            key = args.get("key")
            if not key or not new_type:
                result = {"result": "error", "message": "key and newType required"}
            elif new_type not in [MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_LONG_TERM]:
                result = {"result": "error", "message": "newType must be 'short_term' or 'long_term'"}
            else:
                res = memory_system.update(key=key, memory_type=new_type)
                result = {"result": "ok"} if res.get("success") else {"result": "error", "message": res.get("message")}

        else:
            result = {"result": "error", "message": f"Unknown action: {action}"}

        return types.FunctionResponse(id=function_call.id, name=function_call.name, response=result)

    except Exception as e:
        logger.error(f"Memory function error ({action}): {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_call.name,
            response={"result": "error", "message": str(e)}
        )


def format_memories_for_prompt(memories: List[Dict[str, Any]], max_length: int = 200) -> str:
    """Format memories for system prompt injection."""
    if not memories:
        return ""
    
    lines = []
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        line = f"{i}. [{mem.get('key', 'unknown')}] ({mem.get('category', 'general')}): {content}"
        lines.append(line)
    
    return "\n".join(lines)


def get_memory_content_for_prompt(count: int = 10) -> str:
    """Get formatted memory content for system prompt."""
    if not memory_system.is_available():
        return ""
    
    memories = memory_system.get_recent_for_prompt(count)
    if not memories:
        return ""
    
    formatted = format_memories_for_prompt(memories)
    if formatted:
        total = memory_system.stats().get("stats", {}).get("total", 0)
        return f"\n=== MEMORIES ({len(memories)} of {total} total) ===\n{formatted}"
    return ""


async def recall_memories(query: str, context: str = "", api_key: str = "", personality_prompt: str = "") -> Dict[str, Any]:
    """
    Agentic memory recall: fetches ALL memories, sends them to Gemini Flash Lite
    to summarize relevant information in-character. 15s timeout.
    """
    if not memory_system.is_available():
        return {"result": "error", "message": "Memory system unavailable"}
    
    if not api_key:
        return {"result": "error", "message": "No API key available for recall agent"}

    # Fetch ALL memories (not search-filtered — let the sub-agent decide relevance)
    all_memories = memory_system.list_memories(limit=500)
    memories_found = all_memories.get("memories", [])

    if not memories_found:
        return {"result": "ok", "summary": "No memories stored yet.", "count": 0}

    # Format all memories for the sub-agent with key, content, and date
    memory_lines = []
    for mem in memories_found:
        key = mem.get("key", "unknown")
        content = mem.get("content", "")
        created = mem.get("created_at", "unknown")
        category = mem.get("category", "general")
        memory_lines.append(f"[{key}] ({category}, {created}): {content}")

    memories_block = "\n".join(memory_lines)

    system_prompt = (
        "You are a memory recall assistant. You have been given ALL stored memories and a search query. "
        "Your job is to find every relevant memory and provide a concise, accurate summary. "
        "Include specific details like names, dates, events, and quotes. "
        "If the query is about a person, include everything you know about them. "
        "If nothing is relevant, say so clearly. "
        "Keep your response under 300 words. Be direct and informative."
    )
    if personality_prompt:
        system_prompt += f"\n\nDeliver the summary in-character as: {personality_prompt[:300]}"

    user_prompt = f"QUERY: {query}"
    if context:
        user_prompt += f"\nCONTEXT: {context}"
    user_prompt += f"\n\n=== ALL MEMORIES ({len(memories_found)} total) ===\n{memories_block}"

    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=api_key)

        # 15-second timeout
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=user_prompt,
                config=gtypes.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=500,
                ),
            ),
            timeout=15.0,
        )

        summary = response.text if response.text else "Could not generate summary."
        return {"result": "ok", "summary": summary, "count": len(memories_found)}

    except asyncio.TimeoutError:
        logger.warning("Recall agent timed out after 15s")
        return {"result": "error", "message": "Couldn't summarize memories at this time."}
    except Exception as e:
        logger.error(f"Recall agent failed: {e}")
        return {"result": "error", "message": "Couldn't summarize memories at this time."}
