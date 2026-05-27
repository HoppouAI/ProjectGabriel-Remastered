"""Orchestrator: composes the mixins and owns __init__ state."""

from __future__ import annotations

import os
import threading

from .backfill import BackfillMixin
from .documents import DocumentsMixin
from .embeddings import EmbeddingsMixin
from .helpers import Collection
from .init_backends import InitBackendsMixin
from .lifecycle import LifecycleMixin
from .search import SearchMixin


class DiscordMessageRag(
    InitBackendsMixin,
    EmbeddingsMixin,
    DocumentsMixin,
    BackfillMixin,
    SearchMixin,
    LifecycleMixin,
):
    """Hybrid RAG index for Discord messages."""

    def __init__(self, config):
        self.config = config
        self.cfg = config.get("discord_rag", default={}) or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.provider = self._normalize_provider(self.cfg.get("provider", "local"))
        self.index_on_message = bool(self.cfg.get("index_on_message", True))
        self.backfill_on_startup = bool(self.cfg.get("backfill_on_startup", True))
        self.auto_inject_enabled = bool(self.cfg.get("auto_inject", True))
        self.auto_inject_limit = int(self.cfg.get("auto_inject_limit", 3))
        self.auto_inject_max_chars = int(self.cfg.get("auto_inject_max_chars", 1600))
        self.search_limit = int(self.cfg.get("search_limit", 8))
        self.channel_scope_default = bool(self.cfg.get("channel_scope_default", True))
        self.auto_cross_channel_search = bool(self.cfg.get("auto_cross_channel_search", True))
        self.keyword_fallback_enabled = bool(self.cfg.get("keyword_fallback", True))
        self.exclude_recent_seconds = float(self.cfg.get("exclude_recent_seconds", 30))
        self.window_size = int(self.cfg.get("window_size", 6))
        self.window_stride = int(self.cfg.get("window_stride", 3))
        self.max_backfill_messages = int(self.cfg.get("max_backfill_messages", 25000))
        self.backfill_batch_size = int(self.cfg.get("backfill_batch_size", 32))
        self.embedding_model = str(self.cfg.get("embedding_model", "gemini-embedding-001"))
        self.embedding_dims = int(self.cfg.get("embedding_dims", 768))
        self.local_embedding_model = str(self.cfg.get("local_embedding_model", "text-embedding-embeddinggemma-300m-qat"))
        self.lm_studio_url = str(self.cfg.get("lm_studio_url", "http://localhost:1234")).rstrip("/")
        self.chroma_dir = str(self.cfg.get("chroma_dir", "discord_bot/data/message_chroma_db"))
        self.chroma_collection_name = str(self.cfg.get("chroma_collection", "discord_messages"))
        self.mongo_uri = os.environ.get("DISCORD_RAG_MONGO_URI") or os.environ.get("GABRIEL_MONGO_URI") or str(self.cfg.get("mongo_uri", ""))
        self.mongo_db = os.environ.get("DISCORD_RAG_MONGO_DB") or str(self.cfg.get("mongo_db", "gabriel"))
        self.mongo_collection_name = os.environ.get("DISCORD_RAG_MONGO_COLLECTION") or str(self.cfg.get("mongo_collection", "discord_messages"))
        self.vector_index_name = str(self.cfg.get("vector_index", "discord_message_vector_index"))
        self.score_gemini = float(self.cfg.get("vector_min_score_gemini", 0.82))
        self.score_local = float(self.cfg.get("vector_min_score_local", 0.55))
        legacy_score = self.cfg.get("vector_min_score")
        if legacy_score is not None:
            if self.provider == "local":
                self.score_local = float(legacy_score)
            else:
                self.score_gemini = float(legacy_score)
        self.vector_min_score = self.score_local if self.provider == "local" else self.score_gemini

        self._lock = threading.RLock()
        self._embedding_client = None
        self._httpx_client = None
        self._mongo_client = None
        self._collection: Collection | None = None
        self._chroma_client = None
        self._chroma_collection = None
        self._ready = False
        self._backfill_running = False

        if self.enabled:
            self._init_provider()
