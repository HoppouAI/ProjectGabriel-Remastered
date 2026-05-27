"""Document construction + upsert paths for both backends."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from .helpers import (
    _as_timestamp,
    _clean_text,
    _hash_text,
    _parse_datetime,
    _safe_json,
    _utcnow,
    logger,
    time,
)


class DocumentsMixin:
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
