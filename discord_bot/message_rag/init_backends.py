"""Backend init: mongo + chroma setup, vector-index ensure, provider routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .helpers import (
    ASCENDING,
    CHROMA_AVAILABLE,
    DESCENDING,
    HTTPX_AVAILABLE,
    MongoClient,
    PYMONGO_AVAILABLE,
    SearchIndexModel,
    _import_chromadb,
    chromadb,
    httpx,
    logger,
)


class InitBackendsMixin:
    @staticmethod
    def _normalize_provider(value: Any) -> str:
        provider = str(value or "local").strip().lower()
        if provider in ("mongo", "mongodb", "atlas", "gemini"):
            return "gemini"
        if provider in ("local", "chroma", "chromadb", "lmstudio", "lm_studio"):
            return "local"
        return provider

    def _init_provider(self):
        if self.provider == "gemini":
            self._init_mongo()
        elif self.provider == "local":
            self._init_chroma()
        else:
            logger.warning(f"Unknown Discord RAG provider '{self.provider}', disabling")
            self.enabled = False

    def _init_mongo(self):
        if not PYMONGO_AVAILABLE:
            logger.warning("Discord RAG Mongo provider needs pymongo")
            self.enabled = False
            return
        if not self.mongo_uri:
            logger.warning("Discord RAG Mongo provider needs mongo_uri or DISCORD_RAG_MONGO_URI")
            self.enabled = False
            return
        try:
            self._mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self._mongo_client.admin.command("ping")
            self._collection = self._mongo_client[self.mongo_db][self.mongo_collection_name]
            self._collection.create_index([("doc_id", ASCENDING)], unique=True, name="idx_doc_id_unique")
            self._collection.create_index([("channel_id", ASCENDING), ("created_at", DESCENDING)], name="idx_channel_created")
            self._collection.create_index([("author_ids", ASCENDING), ("created_at", DESCENDING)], name="idx_author_created")
            self._collection.create_index([("chunk_type", ASCENDING), ("created_at", DESCENDING)], name="idx_chunk_created")
            self._ensure_mongo_vector_index()
            self._ready = True
            logger.debug(f"Discord RAG enabled (gemini, MongoDB {self.mongo_db}.{self.mongo_collection_name})")
        except Exception as e:
            logger.warning(f"Discord RAG Mongo connection failed: {e}")
            self.enabled = False

    def _ensure_mongo_vector_index(self):
        if self._collection is None or self._mongo_client is None or SearchIndexModel is None:
            return
        try:
            existing = list(self._collection.list_search_indexes())
            if any(idx.get("name") == self.vector_index_name for idx in existing):
                return
            vector_index = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": self.embedding_dims,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "channel_id"},
                        {"type": "filter", "path": "chunk_type"},
                    ],
                },
                name=self.vector_index_name,
                type="vectorSearch",
            )
            self._collection.create_search_index(vector_index)
            logger.info("Created Discord RAG MongoDB vector search index")
        except Exception as e:
            logger.warning(f"Could not create Discord RAG vector index automatically: {e}")

    def _init_chroma(self):
        if not _import_chromadb():
            logger.warning("Discord RAG local provider needs chromadb")
            self.enabled = False
            return
        if not HTTPX_AVAILABLE:
            logger.warning("Discord RAG local provider needs httpx")
            self.enabled = False
            return
        try:
            # re-import so we pick up the module global set by _import_chromadb
            from . import helpers as _h
            path = Path(self.chroma_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = _h.chromadb.PersistentClient(path=str(path))
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._httpx_client = _h.httpx.Client(timeout=30)
            self._ready = True
            logger.debug(
                f"Discord RAG enabled (local, model={self.local_embedding_model}, "
                f"db={self.chroma_dir}, docs={self._chroma_collection.count()})"
            )
        except Exception as e:
            logger.warning(f"Discord RAG Chroma init failed: {e}")
            self.enabled = False
