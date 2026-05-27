"""Public state: ready flag, stats dict, close()."""

from __future__ import annotations

from typing import Any


class LifecycleMixin:
    @property
    def ready(self) -> bool:
        return self.enabled and self._ready

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
            "auto_cross_channel_search": self.auto_cross_channel_search,
            "keyword_fallback": self.keyword_fallback_enabled,
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
