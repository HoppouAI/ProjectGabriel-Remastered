from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
    from pymongo.collection import Collection
    from pymongo.operations import SearchIndexModel
    PYMONGO_AVAILABLE = True
except ImportError:
    ASCENDING = 1
    DESCENDING = -1
    MongoClient = None
    UpdateOne = None
    Collection = None
    SearchIndexModel = None
    PYMONGO_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

chromadb = None
CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _import_chromadb():
    global chromadb, CHROMA_AVAILABLE
    if chromadb is not None:
        return True
    try:
        import chromadb as _chromadb
        chromadb = _chromadb
        CHROMA_AVAILABLE = True
        return True
    except ImportError:
        return False


def _utcnow() -> datetime:
    return datetime.utcnow()


def _hash_text(text: str, length: int = 24) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None)
    except Exception:
        return None


def _as_timestamp(value: Any) -> float:
    parsed = _parse_datetime(value)
    if parsed:
        return parsed.timestamp()
    return time.time()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if not value:
        return []
    try:
        loaded = json.loads(str(value))
        if isinstance(loaded, list):
            return [str(v) for v in loaded]
    except Exception:
        pass
    return []


def _clean_text(text: str, limit: int = 4000) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


class DiscordMessageRag:
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

    @staticmethod
    def _normalize_provider(value: Any) -> str:
        provider = str(value or "local").strip().lower()
        if provider in ("mongo", "mongodb", "atlas", "gemini"):
            return "gemini"
        if provider in ("local", "chroma", "chromadb", "lmstudio", "lm_studio"):
            return "local"
        return provider

    @property
    def ready(self) -> bool:
        return self.enabled and self._ready

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
            logger.info(f"Discord RAG enabled (gemini, MongoDB {self.mongo_db}.{self.mongo_collection_name})")
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
            path = Path(self.chroma_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(path))
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._httpx_client = httpx.Client(timeout=30)
            self._ready = True
            logger.info(
                f"Discord RAG enabled (local, model={self.local_embedding_model}, "
                f"db={self.chroma_dir}, docs={self._chroma_collection.count()})"
            )
        except Exception as e:
            logger.warning(f"Discord RAG Chroma init failed: {e}")
            self.enabled = False

    def _get_embedding_client(self):
        if self._embedding_client is not None:
            return self._embedding_client
        try:
            api_key = str(getattr(self.config, "api_key", "") or "").strip()
            if not api_key or api_key.upper().startswith("YOUR_"):
                return None
            from google import genai
            self._embedding_client = genai.Client(api_key=api_key)
            return self._embedding_client
        except Exception as e:
            logger.debug(f"Discord RAG Gemini client failed: {e}")
            return None

    def generate_embedding(self, text: str) -> list[float] | None:
        if self.provider == "local":
            return self._embed_local(text)
        return self._embed_gemini(text)

    def generate_embeddings_batch(self, texts: Sequence[str]) -> list[list[float] | None]:
        if not texts:
            return []
        if self.provider == "local":
            return self._embed_local_batch(list(texts))
        return self._embed_gemini_batch(list(texts))

    def _embed_gemini(self, text: str) -> list[float] | None:
        client = self._get_embedding_client()
        if client is None:
            return None
        try:
            from google.genai import types as gtypes
            result = client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=gtypes.EmbedContentConfig(output_dimensionality=self.embedding_dims),
            )
            return result.embeddings[0].values if result.embeddings else None
        except Exception as e:
            logger.debug(f"Discord RAG Gemini embedding failed: {e}")
            return None

    def _embed_gemini_batch(self, texts: list[str]) -> list[list[float] | None]:
        client = self._get_embedding_client()
        if client is None:
            return [None] * len(texts)
        try:
            from google.genai import types as gtypes
            result = client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
                config=gtypes.EmbedContentConfig(output_dimensionality=self.embedding_dims),
            )
            return [emb.values if emb else None for emb in result.embeddings]
        except Exception as e:
            logger.debug(f"Discord RAG Gemini batch embedding failed: {e}")
            return [None] * len(texts)

    def _embed_local(self, text: str) -> list[float] | None:
        if self._httpx_client is None:
            return None
        try:
            resp = self._httpx_client.post(
                f"{self.lm_studio_url}/v1/embeddings",
                json={"model": self.local_embedding_model, "input": [text]},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.debug(f"Discord RAG local embedding failed: {e}")
            return None

    def _embed_local_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float] | None]:
        if self._httpx_client is None:
            return [None] * len(texts)
        embeddings: list[list[float] | None] = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self._httpx_client.post(
                    f"{self.lm_studio_url}/v1/embeddings",
                    json={"model": self.local_embedding_model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                ordered = sorted(data["data"], key=lambda item: item["index"])
                embeddings.extend([item["embedding"] for item in ordered])
            return embeddings
        except Exception as e:
            logger.debug(f"Discord RAG local batch embedding failed: {e}")
            return embeddings + [None] * (len(texts) - len(embeddings))

    async def index_discord_message(self, message, channel_info: str = "", attachment_info: list[dict[str, Any]] | None = None):
        if not self.ready or not self.index_on_message:
            return {"success": False, "message": "Discord RAG is not ready"}
        doc = self._doc_from_discord_message(message, channel_info, attachment_info or [])
        if not doc:
            return {"success": False, "message": "No indexable message content"}
        return await asyncio.to_thread(self.upsert_document, doc, True)

    async def index_assistant_message(self, channel_id: str, content: str, channel_info: str = "", message_id: str | None = None):
        if not self.ready or not self.index_on_message:
            return {"success": False, "message": "Discord RAG is not ready"}
        content = _clean_text(content)
        if not content:
            return {"success": False, "message": "No content"}
        created_at = _utcnow().isoformat()
        msg_id = str(message_id or f"assistant_{_hash_text(f'{channel_id}:{created_at}:{content}')}")
        doc = self._build_document(
            doc_id=f"msg:{channel_id}:{msg_id}",
            chunk_type="message",
            role="assistant",
            content=content,
            channel_id=str(channel_id),
            channel_name=channel_info or "Discord channel",
            author_ids=["self"],
            author_names=["AI"],
            message_ids=[msg_id],
            created_at=created_at,
            attachments=[],
        )
        return await asyncio.to_thread(self.upsert_document, doc, True)

    def _doc_from_discord_message(self, message, channel_info: str, attachment_info: list[dict[str, Any]]):
        content = _clean_text(getattr(message, "clean_content", None) or getattr(message, "content", ""))
        attachments = [info.get("filename", "attachment") for info in attachment_info if isinstance(info, dict)]
        if not content and not attachments:
            return None
        author = getattr(message, "author", None)
        author_name = getattr(author, "display_name", None) or getattr(author, "name", "unknown")
        author_id = str(getattr(author, "id", "unknown"))
        channel = getattr(message, "channel", None)
        channel_id = str(getattr(channel, "id", "unknown"))
        created = getattr(message, "created_at", None)
        created_at = created.isoformat() if created else _utcnow().isoformat()
        message_id = str(getattr(message, "id", _hash_text(f"{channel_id}:{created_at}:{content}")))
        return self._build_document(
            doc_id=f"msg:{channel_id}:{message_id}",
            chunk_type="message",
            role="user",
            content=content,
            channel_id=channel_id,
            channel_name=channel_info or str(channel_id),
            author_ids=[author_id],
            author_names=[author_name],
            message_ids=[message_id],
            created_at=created_at,
            attachments=attachments,
        )

    def _build_document(
        self,
        doc_id: str,
        chunk_type: str,
        role: str,
        content: str,
        channel_id: str,
        channel_name: str,
        author_ids: list[str],
        author_names: list[str],
        message_ids: list[str],
        created_at: str,
        attachments: list[str],
    ) -> dict[str, Any]:
        content = _clean_text(content)
        created_ts = _as_timestamp(created_at)
        author_label = ", ".join(author_names) if author_names else "unknown"
        search_lines = [
            f"Discord {chunk_type}",
            f"Channel: {channel_name} ({channel_id})",
            f"Author: {author_label}",
            f"Role: {role}",
            f"Time: {created_at}",
            f"Content: {content}",
        ]
        if attachments:
            search_lines.append(f"Attachments: {', '.join(attachments)}")
        search_text = "\n".join(search_lines)
        return {
            "doc_id": doc_id,
            "chunk_type": chunk_type,
            "role": role,
            "content": content,
            "search_text": search_text,
            "channel_id": str(channel_id),
            "channel_name": channel_name,
            "author_ids": [str(a) for a in author_ids],
            "author_names": author_names,
            "message_ids": [str(m) for m in message_ids],
            "created_at": created_at,
            "created_ts": created_ts,
            "attachments": attachments,
            "content_hash": _hash_text(f"{channel_id}:{chunk_type}:{content}", 64),
        }

    def upsert_document(self, doc: dict[str, Any], skip_existing: bool = True) -> dict[str, Any]:
        if not self.ready:
            return {"success": False, "message": "Discord RAG is not ready"}
        if skip_existing and self._document_exists(doc["doc_id"]):
            return {"success": True, "skipped": True, "doc_id": doc["doc_id"]}
        embedding = self.generate_embedding(doc["search_text"])
        if embedding is None:
            return {"success": False, "message": "Could not generate embedding"}
        self._upsert_with_embedding(doc, embedding)
        return {"success": True, "doc_id": doc["doc_id"]}

    def _document_exists(self, doc_id: str) -> bool:
        try:
            if self.provider == "gemini" and self._collection is not None:
                return self._collection.find_one({"doc_id": doc_id}, {"_id": 1}) is not None
            if self.provider == "local" and self._chroma_collection is not None:
                existing = self._chroma_collection.get(ids=[doc_id])
                return bool(existing and existing.get("ids"))
        except Exception:
            return False
        return False

    def _existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        try:
            if self.provider == "gemini" and self._collection is not None:
                found = self._collection.find({"doc_id": {"$in": list(ids)}}, {"doc_id": 1})
                return {doc["doc_id"] for doc in found}
            if self.provider == "local" and self._chroma_collection is not None:
                existing = self._chroma_collection.get(ids=list(ids))
                return set(existing.get("ids", [])) if existing else set()
        except Exception as e:
            logger.debug(f"Discord RAG existing id check failed: {e}")
        return set()

    def _upsert_with_embedding(self, doc: dict[str, Any], embedding: list[float]):
        if self.provider == "gemini":
            self._upsert_mongo(doc, embedding)
        else:
            self._upsert_chroma([doc], [embedding])

    def _upsert_mongo(self, doc: dict[str, Any], embedding: list[float]):
        if self._collection is None:
            return
        created_at = _parse_datetime(doc.get("created_at")) or _utcnow()
        update = dict(doc)
        update["created_at"] = created_at
        update["updated_at"] = _utcnow()
        update["embedding"] = embedding
        self._collection.update_one({"doc_id": doc["doc_id"]}, {"$set": update}, upsert=True)

    def _upsert_chroma(self, docs: list[dict[str, Any]], embeddings: list[list[float]]):
        if self._chroma_collection is None or not docs:
            return
        self._chroma_collection.upsert(
            ids=[doc["doc_id"] for doc in docs],
            embeddings=embeddings,
            documents=[doc["search_text"] for doc in docs],
            metadatas=[self._chroma_metadata(doc) for doc in docs],
        )

    def _chroma_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_type": doc.get("chunk_type", "message"),
            "role": doc.get("role", "user"),
            "content": doc.get("content", "")[:1200],
            "channel_id": str(doc.get("channel_id", "")),
            "channel_name": str(doc.get("channel_name", "")),
            "author_ids_json": _safe_json(doc.get("author_ids", [])),
            "author_names_json": _safe_json(doc.get("author_names", [])),
            "message_ids_json": _safe_json(doc.get("message_ids", [])),
            "created_at": str(doc.get("created_at", "")),
            "created_ts": float(doc.get("created_ts", time.time())),
            "attachments_json": _safe_json(doc.get("attachments", [])),
            "content_hash": str(doc.get("content_hash", "")),
        }

    async def backfill_from_conversations(self, conversation_store) -> dict[str, Any]:
        if not self.ready:
            return {"success": False, "message": "Discord RAG is not ready"}
        if self._backfill_running:
            return {"success": False, "message": "Backfill already running"}
        self._backfill_running = True
        try:
            return await asyncio.to_thread(self._backfill_sync, Path(conversation_store._dir))
        finally:
            self._backfill_running = False

    def _backfill_sync(self, conversation_dir: Path) -> dict[str, Any]:
        if not conversation_dir.exists():
            return {"success": True, "indexed": 0, "skipped": 0, "message": "No conversation directory"}
        docs: list[dict[str, Any]] = []
        total_entries = 0
        for path in conversation_dir.glob("*.json"):
            if total_entries >= self.max_backfill_messages:
                break
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                channel_id = str(data.get("channel_id") or path.stem)
                entries = data.get("messages", [])
                docs.extend(self._docs_from_history(channel_id, entries))
                total_entries += len(entries)
            except Exception as e:
                logger.debug(f"Discord RAG could not read {path}: {e}")
        if not docs:
            return {"success": True, "indexed": 0, "skipped": 0, "message": "No messages to index"}

        ids = [doc["doc_id"] for doc in docs]
        existing = self._existing_ids(ids)
        to_index = [doc for doc in docs if doc["doc_id"] not in existing]
        indexed = 0
        failed = 0
        for i in range(0, len(to_index), self.backfill_batch_size):
            batch = to_index[i:i + self.backfill_batch_size]
            embeddings = self.generate_embeddings_batch([doc["search_text"] for doc in batch])
            good_docs = []
            good_embeddings = []
            for doc, embedding in zip(batch, embeddings):
                if embedding is None:
                    failed += 1
                    continue
                good_docs.append(doc)
                good_embeddings.append(embedding)
            if not good_docs:
                continue
            if self.provider == "gemini":
                self._upsert_mongo_many(good_docs, good_embeddings)
            else:
                self._upsert_chroma(good_docs, good_embeddings)
            indexed += len(good_docs)
            if self.provider == "gemini" and i + self.backfill_batch_size < len(to_index):
                time.sleep(1.0)
        result = {"success": True, "indexed": indexed, "skipped": len(existing), "failed": failed, "total_docs": len(docs)}
        logger.info(f"Discord RAG backfill complete: {result}")
        return result

    def _upsert_mongo_many(self, docs: list[dict[str, Any]], embeddings: list[list[float]]):
        if self._collection is None or UpdateOne is None:
            return
        now = _utcnow()
        ops = []
        for doc, embedding in zip(docs, embeddings):
            update = dict(doc)
            update["created_at"] = _parse_datetime(doc.get("created_at")) or now
            update["updated_at"] = now
            update["embedding"] = embedding
            ops.append(UpdateOne({"doc_id": doc["doc_id"]}, {"$set": update}, upsert=True))
        if ops:
            self._collection.bulk_write(ops, ordered=False)

    def _docs_from_history(self, channel_id: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        docs = []
        normalized = []
        for idx, entry in enumerate(entries):
            role = entry.get("role", "user")
            if role not in ("user", "assistant"):
                continue
            content = _clean_text(entry.get("content", ""))
            attachments = [att.get("filename", "attachment") for att in entry.get("attachments", []) if isinstance(att, dict)]
            if not content and not attachments:
                continue
            created_at = entry.get("timestamp") or _utcnow().isoformat()
            username = entry.get("username") or ("AI" if role == "assistant" else "unknown")
            synthetic_id = f"hist_{idx}_{_hash_text(f'{channel_id}:{created_at}:{username}:{content}')}"
            doc = self._build_document(
                doc_id=f"hist:{channel_id}:{synthetic_id}",
                chunk_type="message",
                role=role,
                content=content,
                channel_id=str(channel_id),
                channel_name=f"Discord channel {channel_id}",
                author_ids=["self" if role == "assistant" else username],
                author_names=[username],
                message_ids=[synthetic_id],
                created_at=created_at,
                attachments=attachments,
            )
            docs.append(doc)
            normalized.append({"role": role, "username": username, "content": content, "created_at": created_at, "id": synthetic_id})

        if self.window_size > 1 and normalized:
            stride = max(1, self.window_stride)
            for start in range(0, len(normalized), stride):
                window = normalized[start:start + self.window_size]
                if len(window) < 2:
                    continue
                lines = []
                authors = []
                message_ids = []
                for item in window:
                    label = "AI" if item["role"] == "assistant" else item["username"]
                    lines.append(f"{label}: {item['content']}")
                    authors.append(item["username"])
                    message_ids.append(item["id"])
                content = "\n".join(lines)
                created_at = window[0]["created_at"]
                doc_id = f"window:{channel_id}:{start}:{_hash_text(content)}"
                docs.append(self._build_document(
                    doc_id=doc_id,
                    chunk_type="window",
                    role="mixed",
                    content=content,
                    channel_id=str(channel_id),
                    channel_name=f"Discord channel {channel_id}",
                    author_ids=list(dict.fromkeys(authors)),
                    author_names=list(dict.fromkeys(authors)),
                    message_ids=message_ids,
                    created_at=created_at,
                    attachments=[],
                ))
        return docs

    async def auto_context(self, query: str, channel_id: str, current_message_ids: set[str] | None = None) -> str:
        if not self.ready or not self.auto_inject_enabled:
            return ""
        results = await asyncio.to_thread(
            self.search,
            query,
            self.auto_inject_limit * 4,
            str(channel_id) if self.channel_scope_default else None,
            None,
            self.vector_min_score,
            current_message_ids or set(),
            self.exclude_recent_seconds,
        )
        if not results.get("success") or not results.get("results"):
            return ""
        selected = results["results"][:self.auto_inject_limit]
        return self.format_context(selected, self.auto_inject_max_chars)

    def search(
        self,
        query: str,
        limit: int = 8,
        channel_id: str | None = None,
        author: str | None = None,
        min_score: float | None = None,
        exclude_message_ids: set[str] | None = None,
        exclude_recent_seconds: float = 0,
    ) -> dict[str, Any]:
        if not self.ready:
            return {"success": False, "message": "Discord RAG is not ready"}
        query = _clean_text(query, 2000)
        if not query:
            return {"success": False, "message": "query required"}
        embedding = self.generate_embedding(query)
        if embedding is None:
            return {"success": False, "message": "Could not generate query embedding"}
        raw = self._search_mongo(embedding, limit * 4, channel_id) if self.provider == "gemini" else self._search_chroma(embedding, limit * 4, channel_id)
        threshold = self.vector_min_score if min_score is None else float(min_score)
        exclude_ids = exclude_message_ids or set()
        recent_cutoff = time.time() - exclude_recent_seconds if exclude_recent_seconds else 0
        author_text = str(author or "").lower().strip()
        filtered = []
        for item in raw:
            if item.get("score", 0) < threshold:
                continue
            if exclude_ids and any(mid in exclude_ids for mid in item.get("message_ids", [])):
                continue
            if recent_cutoff and item.get("created_ts", 0) >= recent_cutoff:
                continue
            if author_text:
                haystack = " ".join(item.get("author_ids", []) + item.get("author_names", [])).lower()
                if author_text not in haystack:
                    continue
            filtered.append(item)
            if len(filtered) >= limit:
                break
        return {"success": True, "provider": self.provider, "count": len(filtered), "results": filtered}

    def _search_mongo(self, embedding: list[float], limit: int, channel_id: str | None) -> list[dict[str, Any]]:
        if self._collection is None:
            return []
        vector_search = {
            "index": self.vector_index_name,
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": max(limit * 10, 50),
            "limit": limit,
        }
        if channel_id:
            vector_search["filter"] = {"channel_id": str(channel_id)}
        pipeline = [
            {"$vectorSearch": vector_search},
            {"$project": {
                "doc_id": 1,
                "chunk_type": 1,
                "role": 1,
                "content": 1,
                "channel_id": 1,
                "channel_name": 1,
                "author_ids": 1,
                "author_names": 1,
                "message_ids": 1,
                "created_at": 1,
                "created_ts": 1,
                "attachments": 1,
                "score": {"$meta": "vectorSearchScore"},
            }},
        ]
        try:
            return [self._public_result(doc) for doc in self._collection.aggregate(pipeline)]
        except Exception as e:
            logger.debug(f"Discord RAG Mongo search failed: {e}")
            return []

    def _search_chroma(self, embedding: list[float], limit: int, channel_id: str | None) -> list[dict[str, Any]]:
        if self._chroma_collection is None:
            return []
        where = {"channel_id": str(channel_id)} if channel_id else None
        try:
            results = self._chroma_collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for doc_id, document, metadata, distance in zip(ids, docs, metas, distances):
                item = {
                    "doc_id": doc_id,
                    "chunk_type": metadata.get("chunk_type", "message"),
                    "role": metadata.get("role", "user"),
                    "content": metadata.get("content") or document,
                    "channel_id": metadata.get("channel_id", ""),
                    "channel_name": metadata.get("channel_name", ""),
                    "author_ids": _load_json_list(metadata.get("author_ids_json")),
                    "author_names": _load_json_list(metadata.get("author_names_json")),
                    "message_ids": _load_json_list(metadata.get("message_ids_json")),
                    "created_at": metadata.get("created_at", ""),
                    "created_ts": float(metadata.get("created_ts", 0)),
                    "attachments": _load_json_list(metadata.get("attachments_json")),
                    "score": round(1.0 - float(distance), 4),
                }
                output.append(item)
            return output
        except Exception as e:
            logger.debug(f"Discord RAG Chroma search failed: {e}")
            return []

    def _public_result(self, doc: dict[str, Any]) -> dict[str, Any]:
        created = doc.get("created_at")
        created_at = created.isoformat() if isinstance(created, datetime) else str(created or "")
        return {
            "doc_id": doc.get("doc_id"),
            "chunk_type": doc.get("chunk_type", "message"),
            "role": doc.get("role", "user"),
            "content": doc.get("content", ""),
            "channel_id": str(doc.get("channel_id", "")),
            "channel_name": doc.get("channel_name", ""),
            "author_ids": [str(v) for v in doc.get("author_ids", [])],
            "author_names": doc.get("author_names", []),
            "message_ids": [str(v) for v in doc.get("message_ids", [])],
            "created_at": created_at,
            "created_ts": float(doc.get("created_ts") or _as_timestamp(created_at)),
            "attachments": doc.get("attachments", []),
            "score": round(float(doc.get("score", 0)), 4),
        }

    def format_context(self, results: list[dict[str, Any]], max_chars: int) -> str:
        if not results:
            return ""
        lines = [
            "Relevant older Discord history follows. Use it only when it directly helps the reply, and do not say you searched history unless asked."
        ]
        used = len(lines[0])
        for idx, item in enumerate(results, 1):
            authors = ", ".join(item.get("author_names") or item.get("author_ids") or ["unknown"])
            content = _clean_text(item.get("content", ""), 420)
            line = f"{idx}. score {item.get('score')}: {item.get('created_at')} | {authors} | {content}"
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        if not self.enabled:
            return {"success": True, "enabled": False, "provider": self.provider, "ready": False, "count": 0}
        count = 0
        try:
            if self.provider == "gemini" and self._collection is not None:
                count = self._collection.count_documents({})
            elif self.provider == "local" and self._chroma_collection is not None:
                count = self._chroma_collection.count()
        except Exception as e:
            return {"success": False, "message": str(e), "enabled": self.enabled, "provider": self.provider, "ready": self.ready}
        return {
            "success": True,
            "enabled": self.enabled,
            "provider": self.provider,
            "ready": self.ready,
            "count": count,
            "auto_inject": self.auto_inject_enabled,
            "vector_min_score": self.vector_min_score,
        }

    def close(self):
        try:
            if self._httpx_client:
                self._httpx_client.close()
        except Exception:
            pass
        try:
            if self._mongo_client:
                self._mongo_client.close()
        except Exception:
            pass
