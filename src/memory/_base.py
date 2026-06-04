"""MemorySystem core: __init__, connect (mongo+sqlite), cleanup, close.

All the heavy logic (CRUD, RAG) lives in mixins that wrap this base.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from . import _helpers
from ._helpers import (
    ASCENDING,
    DESCENDING,
    Collection,
    MONGO_AVAILABLE,
    MEMORY_TYPE_LONG_TERM,
    MEMORY_TYPE_QUICK_NOTE,
    MEMORY_TYPE_SHORT_TERM,
    MongoClient,
    _load_config,
)

logger = logging.getLogger(__name__)


class MemorySystemBase:
    """Connection + lifecycle + serialize helpers. Mixed into MemorySystem."""

    def __init__(self):
        self.config = _load_config()
        self.backend = self.config.get("backend", "sqlite").lower()

        # MongoDB settings
        self.mongo_uri = os.environ.get("GABRIEL_MONGO_URI", self.config.get("mongo_uri", ""))
        self.mongo_db = os.environ.get("GABRIEL_MONGO_DB", self.config.get("mongo_db", "gabriel"))
        self.mongo_collection_name = os.environ.get("GABRIEL_MONGO_COLLECTION", self.config.get("mongo_collection", "memories"))

        # SQLite settings
        self.sqlite_path = self.config.get("sqlite_path", "data/gabriel_memories.sqlite")

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

        # RAG config (opt-in)
        self.rag_enabled = self.config.get("rag_enabled", False)
        self.rag_provider = self.config.get("rag_provider", "gemini").lower()  # "gemini" or "local"
        self._embedding_model = self.config.get("embedding_model", "gemini-embedding-001")
        self._embedding_dimensions = int(self.config.get("embedding_dims", 768))
        self._embedding_client = None
        self._vector_index_checked = False

        # Per-provider min score thresholds (local models produce lower similarity scores)
        self._score_gemini = float(self.config.get("vector_min_score_gemini", 0.82))
        self._score_local = float(self.config.get("vector_min_score_local", 0.55))
        # legacy fallback: if old single field exists, use it for the active provider
        legacy = self.config.get("vector_min_score")
        if legacy is not None:
            val = float(legacy)
            if self.rag_provider == "local":
                self._score_local = val
            else:
                self._score_gemini = val
        self.vector_min_score = self._score_local if self.rag_provider == "local" else self._score_gemini

        # Local RAG config (LM Studio + ChromaDB)
        self._lm_studio_url = self.config.get("lm_studio_url", "http://localhost:1234")
        self._local_embedding_model = self.config.get("local_embedding_model", "text-embedding-embeddinggemma-300m-qat")
        self._chroma_client = None
        self._chroma_collection = None
        self._chroma_path = self.config.get("chroma_dir", "data/gabriel_chroma_db")
        self._httpx_client = None

        self._connect()
        self._init_rag()
        if self.is_available():
            self._start_cleanup_thread()

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------
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
            self.collection.create_index([("updated_at", DESCENDING)], name="idx_updated_at")
            self.collection.create_index([("memory_type", ASCENDING), ("created_at", DESCENDING)], name="idx_memory_type_created")
            self.collection.create_index([("content_hash", ASCENDING)], name="idx_content_hash")
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")

    def _ensure_vector_index(self):
        """Create MongoDB Atlas Vector Search index if it doesn't exist."""
        if self._vector_index_checked or self.collection is None:
            return
        self._vector_index_checked = True
        try:
            db = self.client[self.mongo_db]
            existing = list(db[self.mongo_collection_name].list_search_indexes())
            for idx in existing:
                if idx.get("name") == "vector_index":
                    logger.debug("Vector search index already exists")
                    return
            from pymongo.operations import SearchIndexModel
            vector_index = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": self._embedding_dimensions,
                            "similarity": "cosine",
                        },
                        {
                            "type": "filter",
                            "path": "memory_type",
                        },
                    ],
                },
                name="vector_index",
                type="vectorSearch",
            )
            db[self.mongo_collection_name].create_search_index(vector_index)
            logger.info("Created MongoDB Atlas Vector Search index")
        except Exception as e:
            logger.warning(f"Could not create vector search index (may need Atlas UI): {e}")

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
            self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_updated ON memories(updated_at)")
            self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON memories(content_hash)")
            self.sqlite_conn.commit()

    def is_available(self) -> bool:
        """Check if storage is ready."""
        if self.backend == "sqlite":
            return self.sqlite_conn is not None
        return self.collection is not None

    # ------------------------------------------------------------------
    # cleanup background thread
    # ------------------------------------------------------------------
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_running:
            return
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Memory cleanup thread started")

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

    # ------------------------------------------------------------------
    # serialize helpers
    # ------------------------------------------------------------------
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

        if self._httpx_client:
            try:
                self._httpx_client.close()
            except Exception:
                pass
