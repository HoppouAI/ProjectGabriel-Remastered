"""Shared helpers + optional-dep shims for the Discord RAG package."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime
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

logger = logging.getLogger("discord_bot.message_rag")


def _import_chromadb() -> bool:
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


_ENTITY_STOPWORDS = {
    "channel",
    "discord",
    "dm",
    "group",
    "message",
    "someone",
    "about",
    "recently",
    "remember",
    "named",
    "called",
    "username",
    "user",
}

_RECALL_HINT_PATTERN = re.compile(
    r"\b(remember|recall|know|knew|known|ever|recently|before|earlier|previous|past|old|"
    r"history|dm|dming|message|messaged|messaging|said|asked|named|called|username|"
    r"come\s+to\s+mind|who|what)\b",
    re.IGNORECASE,
)


def _semantic_query_text(query: str) -> str:
    text = str(query or "")
    text = re.sub(r"\[CHANNEL:[^\]]*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<@!?\d+>", " ", text)
    text = re.sub(r"\(ID:\d+\)\s*:", ":", text, flags=re.IGNORECASE)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[^:\n]{1,80}:\s*", "", line).strip()
        if line:
            lines.append(line)
    return _clean_text(" ".join(lines), 2000)


def _extract_keyword_terms(query: str, limit: int = 6) -> list[str]:
    text = _semantic_query_text(query)
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str):
        term = re.sub(r"[^A-Za-z0-9_.-]", "", str(term or "")).strip("._-")
        if len(term) < 3:
            return
        lowered = term.lower()
        if lowered in _ENTITY_STOPWORDS or lowered in seen:
            return
        seen.add(lowered)
        terms.append(term)

    for match in re.finditer(r"[`\"']([A-Za-z0-9_.-]{3,40})[`\"']", text):
        add(match.group(1))
    for match in re.finditer(r"\b(?:named|called|username|user)\s+([A-Za-z0-9_.-]{3,40})\b", text, re.IGNORECASE):
        add(match.group(1))
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_.-]{2,40}\b", text):
        add(token)
    for token in re.findall(r"\b[A-Za-z0-9]+_[A-Za-z0-9_.-]*\b", text):
        add(token)

    return terms[:limit]
