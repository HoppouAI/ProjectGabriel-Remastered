"""Module-level constants, config loader, hashing, generic-subject filter.

Lives at the bottom of the import graph so every mixin can pull from here
without worrying about cycles.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict

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

# chromadb imported lazily in _init_local_rag() to save ~1.5s startup time
chromadb = None
CHROMA_AVAILABLE = False


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


try:
    import httpx as _httpx
    HTTPX_AVAILABLE = True
except ImportError:
    _httpx = None
    HTTPX_AVAILABLE = False


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


_GENERIC_SUBJECT_RE = re.compile(
    r"(?<![a-zA-Z])"            # not preceded by a letter (avoids matching inside words)
    r"(?:the\s+user|(?:a|the)\s+person|someone|somebody)"
    r"(?![a-zA-Z])",            # not followed by a letter
    re.IGNORECASE,
)

# patterns that are ok even if they contain "user" (usernames, etc.)
_USERNAME_CONTEXT_RE = re.compile(
    r"(?:username|user\s*(?:name|id|handle)|(?:their|my|the)\s+user(?:name|id))",
    re.IGNORECASE,
)


def _has_generic_subject(content: str) -> bool:
    """Check if content uses generic subjects instead of actual names.
    Returns True if the memory should be rejected."""
    # skip if just referencing a username field
    if _USERNAME_CONTEXT_RE.search(content):
        return False
    # check if content starts with or heavily uses generic phrasing
    lower = content.lower().strip()
    # "User likes X" or "The user said" at the start is the main offender
    if lower.startswith(("the user ", "user ", "a user ", "the person ", "a person ")):
        return True
    # also catch "User's" at the start
    if lower.startswith(("the user's ", "user's ")):
        return True
    return False
