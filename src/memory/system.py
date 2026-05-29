"""MemorySystem class + lazy-init global proxy."""

from __future__ import annotations

from ._base import MemorySystemBase
from .crud import CRUDMixin
from .rag import RAGMixin


class MemorySystem(CRUDMixin, RAGMixin, MemorySystemBase):
    """Unified memory storage with MongoDB and SQLite backends."""


class _LazyMemorySystem:
    """Lazy proxy that defers MemorySystem() creation until first attribute access."""

    def __init__(self):
        self._instance = None

    def _ensure(self):
        if self._instance is None:
            self._instance = MemorySystem()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._ensure(), name)

    def __bool__(self):
        return self._instance is not None and bool(self._instance)


# Global instance (lazy -- connects on first use, not at import time)
memory_system = _LazyMemorySystem()
