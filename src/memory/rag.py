"""RAG: embedding generation, vector search, ChromaDB sync, backfill."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from . import _helpers
from ._helpers import (
    HTTPX_AVAILABLE,
    MEMORY_TYPE_LONG_TERM,
    _httpx,
    _import_chromadb,
)

logger = logging.getLogger(__name__)


class RAGMixin:
    """Embedding clients (Gemini + LM Studio) + Mongo/Chroma vector search."""

    def _init_rag(self):
        """Initialize RAG system based on provider config."""
        if not self.rag_enabled:
            logger.info("RAG disabled, using legacy keyword recall")
            return

        if self.rag_provider == "local":
            self._init_local_rag()
        elif self.rag_provider == "gemini":
            if self.backend != "mongo":
                logger.warning("Gemini RAG provider requires MongoDB backend, disabling RAG")
                self.rag_enabled = False
                return
            self._ensure_vector_index()
            logger.debug(f"RAG enabled (gemini, model={self._embedding_model}, dims={self._embedding_dimensions})")
        else:
            logger.warning(f"Unknown rag_provider '{self.rag_provider}', disabling RAG")
            self.rag_enabled = False

    def _init_local_rag(self):
        """Initialize local RAG with ChromaDB + LM Studio embeddings."""
        if not _import_chromadb():
            logger.warning("ChromaDB not installed, disabling local RAG (pip install chromadb)")
            self.rag_enabled = False
            return
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed, disabling local RAG (pip install httpx)")
            self.rag_enabled = False
            return

        try:
            from pathlib import Path
            db_path = Path(self._chroma_path)
            db_path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = _helpers.chromadb.PersistentClient(path=str(db_path))
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="memories",
                metadata={"hnsw:space": "cosine"},
            )
            self._httpx_client = _httpx.Client(timeout=30)
            logger.debug(
                f"RAG enabled (local, model={self._local_embedding_model}, "
                f"db={self._chroma_path}, memories={self._chroma_collection.count()})"
            )
            # auto sync existing memories into chroma on startup
            threading.Thread(target=self._auto_sync_chroma, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to initialize local RAG: {e}")
            self.rag_enabled = False

    def _auto_sync_chroma(self):
        """Background sync of existing memories into ChromaDB."""
        try:
            time.sleep(2)  # let the system settle
            result = self.sync_chroma()
            if result.get("synced", 0) > 0:
                logger.info(f"Auto-synced {result['synced']} memories to ChromaDB")
        except Exception as e:
            logger.debug(f"Auto-sync failed: {e}")

    # ------------------------------------------------------------------
    # embedding generation (gemini + local)
    # ------------------------------------------------------------------
    def _embed_local(self, text: str) -> Optional[List[float]]:
        """Generate embedding via LM Studio local server."""
        if self._httpx_client is None:
            return None
        try:
            resp = self._httpx_client.post(
                f"{self._lm_studio_url}/v1/embeddings",
                json={"model": self._local_embedding_model, "input": [text]},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.debug(f"Local embedding failed: {e}")
            return None

    def _embed_local_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """Generate embeddings via LM Studio in batches."""
        if self._httpx_client is None:
            return [None] * len(texts)
        all_embeddings: List[Optional[List[float]]] = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self._httpx_client.post(
                    f"{self._lm_studio_url}/v1/embeddings",
                    json={"model": self._local_embedding_model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                batch_embs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
                all_embeddings.extend(batch_embs)
            return all_embeddings
        except Exception as e:
            logger.debug(f"Local batch embedding failed: {e}")
            return all_embeddings + [None] * (len(texts) - len(all_embeddings))

    def _get_embedding_client(self):
        """Get or create Gemini client for embeddings."""
        if self._embedding_client is None:
            try:
                from google import genai
                api_key = self._get_api_key()
                if api_key:
                    self._embedding_client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.debug(f"Could not create embedding client: {e}")
        return self._embedding_client

    def _get_api_key(self) -> str:
        """Get API key from config for embedding generation."""
        try:
            from src.config import Config
            config = Config()
            return config.api_key
        except Exception:
            return ""

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector using configured provider."""
        if self.rag_provider == "local":
            return self._embed_local(text)
        # gemini provider
        client = self._get_embedding_client()
        if client is None:
            return None
        try:
            from google.genai import types as gtypes
            result = client.models.embed_content(
                model=self._embedding_model,
                contents=text,
                config=gtypes.EmbedContentConfig(
                    output_dimensionality=self._embedding_dimensions,
                ),
            )
            return result.embeddings[0].values if result.embeddings else None
        except Exception as e:
            logger.debug(f"Embedding generation failed: {e}")
            return None

    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if self.rag_provider == "local":
            return self._embed_local_batch(texts)
        # gemini provider
        client = self._get_embedding_client()
        if client is None:
            return [None] * len(texts)
        try:
            from google.genai import types as gtypes
            result = client.models.embed_content(
                model=self._embedding_model,
                contents=texts,
                config=gtypes.EmbedContentConfig(
                    output_dimensionality=self._embedding_dimensions,
                ),
            )
            return [emb.values if emb else None for emb in result.embeddings]
        except Exception as e:
            logger.debug(f"Batch embedding failed: {e}")
            return [None] * len(texts)

    # ------------------------------------------------------------------
    # vector search (router + mongo + chroma impls)
    # ------------------------------------------------------------------
    def vector_search(self, query: str, limit: int = 20, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """Semantic vector search using configured provider."""
        if self.rag_provider == "local":
            return self._vector_search_chroma(query, limit, memory_type)
        return self._vector_search_mongo(query, limit, memory_type)

    def _vector_search_mongo(self, query: str, limit: int = 20, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """Vector search via MongoDB Atlas."""
        if self.backend != "mongo" or self.collection is None:
            return {"success": False, "message": "Vector search requires MongoDB backend"}

        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return {"success": False, "message": "Could not generate query embedding"}

        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit,
                    }
                },
                {
                    "$project": {
                        "key": 1,
                        "content": 1,
                        "category": 1,
                        "memory_type": 1,
                        "tags": 1,
                        "created_at": 1,
                        "access_count": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]

            if memory_type:
                pipeline[0]["$vectorSearch"]["filter"] = {"memory_type": memory_type}

            memories = []
            for doc in self.collection.aggregate(pipeline):
                memories.append({
                    "key": doc.get("key"),
                    "content": doc.get("content", ""),
                    "category": doc.get("category", "general"),
                    "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
                    "tags": doc.get("tags", []),
                    "created_at": self._serialize_dt(doc.get("created_at")),
                    "access_count": doc.get("access_count", 0),
                    "score": round(doc.get("score", 0), 4),
                })

            return {"success": True, "memories": memories, "count": len(memories)}

        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return {"success": False, "message": str(e)}

    def _vector_search_chroma(self, query: str, limit: int = 20, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """Vector search via local ChromaDB."""
        if self._chroma_collection is None:
            return {"success": False, "message": "ChromaDB not initialized"}

        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return {"success": False, "message": "Could not generate query embedding"}

        try:
            where_filter = {"memory_type": memory_type} if memory_type else None
            results = self._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter,
                include=["metadatas", "documents", "distances"],
            )

            memories = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0]
                metadatas = results["metadatas"][0]
                documents = results["documents"][0]

                for doc_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
                    similarity = 1.0 - dist  # cosine distance to similarity
                    memories.append({
                        "key": doc_id,
                        "content": doc,
                        "category": meta.get("category", "general"),
                        "memory_type": meta.get("memory_type", MEMORY_TYPE_LONG_TERM),
                        "tags": json.loads(meta.get("tags_json", "[]")),
                        "created_at": meta.get("created_at"),
                        "access_count": int(meta.get("access_count", 0)),
                        "score": round(similarity, 4),
                    })

            return {"success": True, "memories": memories, "count": len(memories)}

        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
            return {"success": False, "message": str(e)}

    # ------------------------------------------------------------------
    # chroma upsert / delete / batch sync
    # ------------------------------------------------------------------
    def _upsert_chroma(self, key: str, content: str, category: str, memory_type: str,
                       tags: List[str], created_at: str, access_count: int = 0):
        """Upsert a memory into ChromaDB with its embedding."""
        if self._chroma_collection is None:
            return
        embed_text = f"{category}: {content}"
        embedding = self.generate_embedding(embed_text)
        if embedding is None:
            return
        try:
            self._chroma_collection.upsert(
                ids=[key],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    "category": category,
                    "memory_type": memory_type,
                    "tags_json": json.dumps(tags, ensure_ascii=False),
                    "created_at": str(created_at) if created_at else "",
                    "access_count": access_count,
                }],
            )
        except Exception as e:
            logger.debug(f"ChromaDB upsert failed for {key}: {e}")

    def _delete_chroma(self, key: str):
        """Delete a memory from ChromaDB."""
        if self._chroma_collection is None:
            return
        try:
            self._chroma_collection.delete(ids=[key])
        except Exception as e:
            logger.debug(f"ChromaDB delete failed for {key}: {e}")

    def sync_chroma(self, batch_size: int = 20) -> Dict[str, Any]:
        """Populate ChromaDB from existing memories in the primary backend."""
        if self._chroma_collection is None:
            return {"success": False, "message": "ChromaDB not initialized"}

        try:
            all_mems = self._get_all_memories_raw()
            if not all_mems:
                return {"success": True, "synced": 0, "message": "No memories to sync"}

            existing_ids = set(self._chroma_collection.get()["ids"])
            to_sync = [m for m in all_mems if m.get("key") and m["key"] not in existing_ids]

            if not to_sync:
                return {"success": True, "synced": 0, "message": f"All {len(all_mems)} memories already in ChromaDB"}

            total = len(to_sync)
            synced = 0

            for i in range(0, total, batch_size):
                batch = to_sync[i:i + batch_size]
                texts = [f"{m.get('category', 'general')}: {m.get('content', '')}" for m in batch]
                embeddings = self.generate_embeddings_batch(texts)

                ids, embeds, docs, metas = [], [], [], []
                for mem, emb in zip(batch, embeddings):
                    if emb is not None:
                        ids.append(mem["key"])
                        embeds.append(emb)
                        docs.append(mem.get("content", ""))
                        metas.append({
                            "category": mem.get("category", "general"),
                            "memory_type": mem.get("memory_type", MEMORY_TYPE_LONG_TERM),
                            "tags_json": json.dumps(mem.get("tags", []), ensure_ascii=False),
                            "created_at": str(mem.get("created_at", "")),
                            "access_count": int(mem.get("access_count", 0)),
                        })

                if ids:
                    self._chroma_collection.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
                    synced += len(ids)

                if i + batch_size < total:
                    time.sleep(0.5)

            logger.info(f"ChromaDB sync complete: {synced}/{total} new memories indexed")
            return {"success": True, "synced": synced, "total": total}

        except Exception as e:
            logger.error(f"ChromaDB sync failed: {e}")
            return {"success": False, "message": str(e)}

    def _get_all_memories_raw(self) -> List[Dict[str, Any]]:
        """Get all memories from primary backend as dicts."""
        memories = []
        try:
            if self.backend == "sqlite" and self.sqlite_conn:
                with self._sqlite_lock:
                    cur = self.sqlite_conn.execute(
                        "SELECT key, content, category, memory_type, tags_json, created_at, access_count FROM memories"
                    )
                    for row in cur.fetchall():
                        memories.append({
                            "key": row[0], "content": row[1], "category": row[2],
                            "memory_type": row[3], "tags": json.loads(row[4] or "[]"),
                            "created_at": row[5], "access_count": row[6],
                        })
            elif self.backend == "mongo" and self.collection is not None:
                for doc in self.collection.find({}, {"key": 1, "content": 1, "category": 1,
                                                      "memory_type": 1, "tags": 1, "created_at": 1, "access_count": 1}):
                    memories.append({
                        "key": doc.get("key"), "content": doc.get("content", ""),
                        "category": doc.get("category", "general"),
                        "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
                        "tags": doc.get("tags", []),
                        "created_at": self._serialize_dt(doc.get("created_at")),
                        "access_count": doc.get("access_count", 0),
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch memories for sync: {e}")
        return memories

    def backfill_embeddings(self, batch_size: int = 20) -> Dict[str, Any]:
        """Generate embeddings for all memories that don't have one yet."""
        if self.backend != "mongo" or self.collection is None:
            return {"success": False, "message": "Backfill requires MongoDB backend"}

        try:
            docs = list(self.collection.find(
                {"embedding": {"$exists": False}},
                {"key": 1, "content": 1, "category": 1}
            ))
            if not docs:
                return {"success": True, "backfilled": 0, "message": "All memories already have embeddings"}

            total = len(docs)
            backfilled = 0

            for i in range(0, total, batch_size):
                batch = docs[i:i + batch_size]
                texts = [f"{d.get('category', 'general')}: {d.get('content', '')}" for d in batch]
                embeddings = self.generate_embeddings_batch(texts)

                for doc, embedding in zip(batch, embeddings):
                    if embedding is not None:
                        self.collection.update_one(
                            {"_id": doc["_id"]},
                            {"$set": {"embedding": embedding}}
                        )
                        backfilled += 1

                if i + batch_size < total:
                    time.sleep(1.5)  # Rate limit between batches (100 RPM on free tier)

            logger.info(f"Backfilled embeddings: {backfilled}/{total}")
            return {"success": True, "backfilled": backfilled, "total": total}

        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            return {"success": False, "message": str(e)}
