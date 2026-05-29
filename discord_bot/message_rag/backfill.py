"""Bulk import of existing conversation history into the RAG index."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .helpers import (
    UpdateOne,
    _clean_text,
    _hash_text,
    _parse_datetime,
    _utcnow,
    logger,
)


class BackfillMixin:
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
        logger.debug(f"Discord RAG backfill complete: {result}")
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
