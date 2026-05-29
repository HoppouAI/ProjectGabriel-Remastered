"""Vector + keyword search, result merging, public formatting."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from typing import Any

from .helpers import (
    DESCENDING,
    _RECALL_HINT_PATTERN,
    _as_timestamp,
    _clean_text,
    _extract_keyword_terms,
    _load_json_list,
    _semantic_query_text,
    logger,
)


class SearchMixin:
    async def auto_context(self, query: str, channel_id: str, current_message_ids: set[str] | None = None) -> str:
        if not self.ready or not self.auto_inject_enabled:
            return ""
        scopes: list[str | None] = []
        if self.channel_scope_default and channel_id:
            scopes.append(str(channel_id))
        if self.should_search_all_channels(query) or not scopes:
            scopes.append(None)

        combined: list[dict[str, Any]] = []
        seen_scopes = set()
        for scope in scopes:
            scope_key = scope or "all"
            if scope_key in seen_scopes:
                continue
            seen_scopes.add(scope_key)
            results = await asyncio.to_thread(
                self.search,
                query,
                self.auto_inject_limit * 4,
                scope,
                None,
                self.vector_min_score,
                current_message_ids or set(),
                self.exclude_recent_seconds,
            )
            if results.get("success") and results.get("results"):
                combined.extend(results["results"])

        merged = self._merge_results(combined)
        if not merged:
            return ""
        selected = merged[:self.auto_inject_limit]
        return self.format_context(selected, self.auto_inject_max_chars)

    def should_search_all_channels(self, query: str) -> bool:
        if not self.auto_cross_channel_search:
            return False
        semantic_query = _semantic_query_text(query)
        if _extract_keyword_terms(semantic_query):
            return True
        return bool(_RECALL_HINT_PATTERN.search(semantic_query))

    @staticmethod
    def _merge_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in results:
            doc_id = str(item.get("doc_id") or "")
            if not doc_id:
                continue
            existing = merged.get(doc_id)
            if existing is None or item.get("score", 0) > existing.get("score", 0):
                merged[doc_id] = item
        return sorted(merged.values(), key=lambda item: item.get("score", 0), reverse=True)

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
        original_query = _clean_text(query, 2000)
        semantic_query = _semantic_query_text(original_query)
        if not semantic_query:
            return {"success": False, "message": "query required"}
        embedding = self.generate_embedding(semantic_query)
        raw = []
        if embedding is not None:
            raw = self._search_mongo(embedding, limit * 4, channel_id) if self.provider == "gemini" else self._search_chroma(embedding, limit * 4, channel_id)
        keyword_hits = self._keyword_search(original_query, limit * 4, channel_id)
        raw = self._merge_results(raw + keyword_hits)
        if embedding is None and not raw:
            return {"success": False, "message": "Could not generate query embedding"}
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

    def _keyword_search(self, query: str, limit: int, channel_id: str | None) -> list[dict[str, Any]]:
        if not self.keyword_fallback_enabled:
            return []
        terms = _extract_keyword_terms(query)
        if not terms:
            return []
        if self.provider == "gemini":
            return self._keyword_search_mongo(terms, limit, channel_id)
        return self._keyword_search_chroma(terms, limit, channel_id)

    def _keyword_search_mongo(self, terms: list[str], limit: int, channel_id: str | None) -> list[dict[str, Any]]:
        if self._collection is None:
            return []
        output = []
        seen = set()
        for term in terms:
            safe = re.escape(term)
            filters: list[dict[str, Any]] = [
                {"content": {"$regex": safe, "$options": "i"}},
                {"channel_name": {"$regex": safe, "$options": "i"}},
                {"author_names": {"$regex": safe, "$options": "i"}},
                {"author_ids": {"$regex": safe, "$options": "i"}},
            ]
            query: dict[str, Any] = {"$or": filters}
            if channel_id:
                query = {"$and": [{"channel_id": str(channel_id)}, query]}
            try:
                for doc in self._collection.find(query).sort("created_at", DESCENDING).limit(limit):
                    doc_id = str(doc.get("doc_id") or "")
                    if not doc_id or doc_id in seen:
                        continue
                    seen.add(doc_id)
                    item = self._public_result(doc)
                    item["score"] = self._keyword_score(item, terms)
                    item["match_type"] = "keyword"
                    output.append(item)
                    if len(output) >= limit:
                        return output
            except Exception as e:
                logger.debug(f"Discord RAG Mongo keyword search failed: {e}")
        return output

    def _keyword_search_chroma(self, terms: list[str], limit: int, channel_id: str | None) -> list[dict[str, Any]]:
        if self._chroma_collection is None:
            return []
        output = []
        seen = set()
        for term in terms:
            try:
                results = self._chroma_collection.get(
                    where={"channel_id": str(channel_id)} if channel_id else None,
                    where_document={"$contains": term},
                    limit=limit,
                    include=["documents", "metadatas"],
                )
            except Exception as e:
                logger.debug(f"Discord RAG Chroma keyword search failed: {e}")
                continue
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            for doc_id, document, metadata in zip(ids, docs, metas):
                doc_id = str(doc_id or "")
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                item = self._chroma_item(doc_id, document, metadata)
                item["score"] = self._keyword_score(item, terms)
                item["match_type"] = "keyword"
                output.append(item)
                if len(output) >= limit:
                    return output
        return output

    @staticmethod
    def _keyword_score(item: dict[str, Any], terms: list[str]) -> float:
        haystack = " ".join(
            [
                item.get("content", ""),
                item.get("channel_name", ""),
                " ".join(item.get("author_names", [])),
                " ".join(item.get("author_ids", [])),
            ]
        ).lower()
        score = 0.86
        for term in terms:
            lowered = term.lower()
            if lowered in haystack:
                score += 0.04
            if lowered in " ".join(item.get("author_names", [])).lower():
                score += 0.06
        if item.get("chunk_type") == "window":
            score += 0.02
        return round(min(score, 0.99), 4)

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
                item = self._chroma_item(doc_id, document, metadata)
                item["score"] = round(1.0 - float(distance), 4)
                item["match_type"] = "semantic"
                output.append(item)
            return output
        except Exception as e:
            logger.debug(f"Discord RAG Chroma search failed: {e}")
            return []

    @staticmethod
    def _chroma_item(doc_id: str, document: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
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
            "score": 0.0,
        }

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
            channel = item.get("channel_name") or f"Discord channel {item.get('channel_id', 'unknown')}"
            content = _clean_text(item.get("content", ""), 420)
            line = f"{idx}. score {item.get('score')}: {item.get('created_at')} | {channel} | {authors} | {content}"
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
