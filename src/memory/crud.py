"""save/read/update/delete/list/search/stats + duplicate + recent-for-prompt."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ._helpers import (
    DESCENDING,
    MEMORY_TYPE_LONG_TERM,
    MEMORY_TYPE_QUICK_NOTE,
    MEMORY_TYPE_SHORT_TERM,
    ReturnDocument,
    _has_generic_subject,
    _hash_content,
)

logger = logging.getLogger(__name__)


class CRUDMixin:
    """Read/write paths for both SQLite and MongoDB backends."""

    def save(
        self,
        key: str,
        content: str,
        category: str = "general",
        memory_type: str = MEMORY_TYPE_LONG_TERM,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Save a memory."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        valid_types = [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_QUICK_NOTE]
        if memory_type not in valid_types:
            return {"success": False, "message": f"Invalid memory type: {memory_type}"}

        # reject generic "the user" / "user" content, the AI should use actual names
        if _has_generic_subject(content) and memory_type != MEMORY_TYPE_QUICK_NOTE:
            return {
                "success": False,
                "message": "Memory rejected: use the person's actual name or username instead of 'the user' or 'user'. Re-save with their real name.",
            }

        tags_list = list(tags) if tags else []
        content_hash = _hash_content(content)
        now = datetime.utcnow()

        try:
            if self.backend == "sqlite":
                tags_json = json.dumps(tags_list, ensure_ascii=False)
                now_iso = now.isoformat()

                with self._sqlite_lock:
                    self.sqlite_conn.execute("""
                        INSERT INTO memories (key, content, category, memory_type, tags_json, content_hash, created_at, updated_at, access_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                        ON CONFLICT(key) DO UPDATE SET
                            content = excluded.content,
                            category = excluded.category,
                            memory_type = excluded.memory_type,
                            tags_json = excluded.tags_json,
                            content_hash = excluded.content_hash,
                            updated_at = excluded.updated_at
                    """, (key, content, category, memory_type, tags_json, content_hash, now_iso, now_iso))
                    self.sqlite_conn.commit()
            else:
                update_fields = {
                    "content": content,
                    "category": category,
                    "memory_type": memory_type,
                    "tags": tags_list,
                    "content_hash": content_hash,
                    "updated_at": now,
                }
                # Generate embedding for RAG (only when enabled)
                if self.rag_enabled:
                    embedding = self.generate_embedding(f"{category}: {content}")
                    if embedding is not None:
                        update_fields["embedding"] = embedding

                self.collection.find_one_and_update(
                    {"key": key},
                    {
                        "$set": update_fields,
                        "$setOnInsert": {"created_at": now, "access_count": 0},
                    },
                    upsert=True,
                    return_document=ReturnDocument.AFTER,
                )

            logger.info(f"Memory saved: {key} ({memory_type})")

            # sync to ChromaDB if using local RAG
            if self.rag_enabled and self.rag_provider == "local":
                self._upsert_chroma(key, content, category, memory_type, tags_list, now.isoformat())

            return {"success": True, "key": key, "memory_type": memory_type}

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return {"success": False, "message": str(e)}

    def read(self, key: str) -> Dict[str, Any]:
        """Read a memory by key."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    row = self.sqlite_conn.execute(
                        "SELECT * FROM memories WHERE key = ?", (key,)
                    ).fetchone()

                    if not row:
                        return {"success": False, "message": f"Memory '{key}' not found"}

                    self.sqlite_conn.execute(
                        "UPDATE memories SET access_count = access_count + 1 WHERE key = ?", (key,)
                    )
                    self.sqlite_conn.commit()

                    tags = json.loads(row["tags_json"]) if row["tags_json"] else []
                    memory = {
                        "key": row["key"],
                        "content": row["content"],
                        "category": row["category"],
                        "memory_type": row["memory_type"],
                        "tags": tags,
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "access_count": row["access_count"] + 1,
                    }
            else:
                doc = self.collection.find_one_and_update(
                    {"key": key},
                    {"$inc": {"access_count": 1}},
                    projection={"embedding": 0},
                    return_document=ReturnDocument.AFTER,
                )
                if not doc:
                    return {"success": False, "message": f"Memory '{key}' not found"}

                memory = self._format_doc(doc)

            logger.info(f"Memory read: {key}")
            return {"success": True, "memory": memory}

        except Exception as e:
            logger.error(f"Read failed: {e}")
            return {"success": False, "message": str(e)}

    def update(
        self,
        key: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        if memory_type and memory_type not in [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_QUICK_NOTE]:
            return {"success": False, "message": f"Invalid memory type: {memory_type}"}

        try:
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    row = self.sqlite_conn.execute("SELECT id FROM memories WHERE key = ?", (key,)).fetchone()
                    if not row:
                        return {"success": False, "message": f"Memory '{key}' not found"}

                    updates = []
                    values = []
                    if content is not None:
                        updates.extend(["content = ?", "content_hash = ?"])
                        values.extend([content, _hash_content(content)])
                    if category is not None:
                        updates.append("category = ?")
                        values.append(category)
                    if memory_type is not None:
                        updates.append("memory_type = ?")
                        values.append(memory_type)
                    if tags is not None:
                        updates.append("tags_json = ?")
                        values.append(json.dumps(tags, ensure_ascii=False))

                    if updates:
                        updates.append("updated_at = ?")
                        values.append(datetime.utcnow().isoformat())
                        values.append(key)

                        self.sqlite_conn.execute(
                            f"UPDATE memories SET {', '.join(updates)} WHERE key = ?", values
                        )
                        self.sqlite_conn.commit()
            else:
                if not self.collection.find_one({"key": key}):
                    return {"success": False, "message": f"Memory '{key}' not found"}

                updates = {"updated_at": datetime.utcnow()}
                if content is not None:
                    updates["content"] = content
                    updates["content_hash"] = _hash_content(content)
                if category is not None:
                    updates["category"] = category
                if memory_type is not None:
                    updates["memory_type"] = memory_type
                if tags is not None:
                    updates["tags"] = tags

                self.collection.update_one({"key": key}, {"$set": updates})

            # Re-embed in ChromaDB if content changed
            if content is not None and self._chroma_collection is not None:
                existing = self.read(key)
                mem = existing.get("memory", {}) if existing.get("success") else {}
                cat = category if category is not None else mem.get("category", "general")
                mem_type = memory_type if memory_type is not None else mem.get("memory_type", MEMORY_TYPE_LONG_TERM)
                tag_list = tags if tags is not None else mem.get("tags", [])
                self._upsert_chroma(key, content, cat, mem_type, tag_list, datetime.utcnow().isoformat())

            logger.info(f"Memory updated: {key}")
            return {"success": True, "key": key}

        except Exception as e:
            logger.error(f"Update failed: {e}")
            return {"success": False, "message": str(e)}

    def delete(self, key: str) -> Dict[str, Any]:
        """Delete a memory."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            if self.backend == "sqlite":
                with self._sqlite_lock:
                    cur = self.sqlite_conn.execute("DELETE FROM memories WHERE key = ?", (key,))
                    deleted = cur.rowcount or 0
                    self.sqlite_conn.commit()
            else:
                res = self.collection.delete_one({"key": key})
                deleted = res.deleted_count if res else 0

            if deleted:
                logger.info(f"Memory deleted: {key}")
                if self.rag_enabled and self.rag_provider == "local":
                    self._delete_chroma(key)
                return {"success": True, "key": key}
            return {"success": False, "message": f"Memory '{key}' not found"}

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return {"success": False, "message": str(e)}

    def list_memories(
        self,
        category: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """List memories with optional filters."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            memories = []

            if self.backend == "sqlite":
                where = []
                params = []
                if category:
                    where.append("category = ?")
                    params.append(category)
                if memory_type:
                    where.append("memory_type = ?")
                    params.append(memory_type)

                where_sql = f" WHERE {' AND '.join(where)}" if where else ""
                params.append(limit)

                with self._sqlite_lock:
                    rows = self.sqlite_conn.execute(
                        f"SELECT * FROM memories{where_sql} ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                        params
                    ).fetchall()

                for row in rows:
                    content = row["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": row["key"],
                        "content": content,
                        "category": row["category"],
                        "memory_type": row["memory_type"],
                        "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
                        "created_at": row["created_at"],
                        "access_count": row["access_count"],
                    })
            else:
                filters = {}
                if category:
                    filters["category"] = category
                if memory_type:
                    filters["memory_type"] = memory_type

                cursor = self.collection.find(filters, {"embedding": 0}).sort("updated_at", DESCENDING).limit(limit)
                for doc in cursor:
                    content = doc.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": doc.get("key"),
                        "content": content,
                        "category": doc.get("category", "general"),
                        "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
                        "tags": doc.get("tags", []),
                        "created_at": self._serialize_dt(doc.get("created_at")),
                        "access_count": doc.get("access_count", 0),
                    })

            return {"success": True, "memories": memories, "count": len(memories)}

        except Exception as e:
            logger.error(f"List failed: {e}")
            return {"success": False, "message": str(e)}

    def search(self, term: str, memory_type: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Search memories by content or key."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            memories = []

            if self.backend == "sqlite":
                like = f"%{term}%"
                where = ["(key LIKE ? OR content LIKE ?)"]
                params: List[Any] = [like, like]

                if memory_type:
                    where.append("memory_type = ?")
                    params.append(memory_type)

                params.append(limit)

                with self._sqlite_lock:
                    rows = self.sqlite_conn.execute(
                        f"SELECT * FROM memories WHERE {' AND '.join(where)} ORDER BY access_count DESC LIMIT ?",
                        params
                    ).fetchall()

                for row in rows:
                    content = row["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": row["key"],
                        "content": content,
                        "category": row["category"],
                        "memory_type": row["memory_type"],
                        "created_at": row["created_at"],
                    })
            else:
                import re
                regex = {"$regex": re.escape(term), "$options": "i"}
                query: Dict[str, Any] = {"$or": [{"key": regex}, {"content": regex}]}
                if memory_type:
                    query["memory_type"] = memory_type

                cursor = self.collection.find(query, {"embedding": 0}).sort("access_count", DESCENDING).limit(limit)
                for doc in cursor:
                    content = doc.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": doc.get("key"),
                        "content": content,
                        "category": doc.get("category", "general"),
                        "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
                        "created_at": self._serialize_dt(doc.get("created_at")),
                    })

            return {"success": True, "memories": memories, "count": len(memories), "term": term}

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"success": False, "message": str(e)}

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.is_available():
            return {"success": False, "message": "Memory storage unavailable"}

        try:
            counts = {"total": 0, "long_term": 0, "short_term": 0, "quick_note": 0}

            if self.backend == "sqlite":
                with self._sqlite_lock:
                    rows = self.sqlite_conn.execute(
                        "SELECT memory_type, COUNT(1) as c FROM memories GROUP BY memory_type"
                    ).fetchall()
                for row in rows:
                    mt = row[0] or "unknown"
                    c = row[1] or 0
                    counts[mt] = c
                    counts["total"] += c
            else:
                pipeline = [{"$group": {"_id": "$memory_type", "count": {"$sum": 1}}}]
                for row in self.collection.aggregate(pipeline):
                    mt = row.get("_id") or "unknown"
                    c = row.get("count", 0)
                    counts[mt] = c
                    counts["total"] += c

            return {"success": True, "stats": counts}

        except Exception as e:
            logger.error(f"Stats failed: {e}")
            return {"success": False, "message": str(e)}

    def has_recent_duplicate(self, content_hash: str, window_seconds: float, types_list: Optional[List[str]] = None) -> bool:
        """Check if a similar memory was saved recently."""
        if not self.is_available():
            return False

        try:
            since = datetime.utcnow() - timedelta(seconds=window_seconds)

            if self.backend == "sqlite":
                with self._sqlite_lock:
                    if types_list:
                        placeholders = ",".join(["?"] * len(types_list))
                        row = self.sqlite_conn.execute(
                            f"SELECT id FROM memories WHERE content_hash = ? AND created_at > ? AND memory_type IN ({placeholders}) LIMIT 1",
                            (content_hash, since.isoformat(), *types_list)
                        ).fetchone()
                    else:
                        row = self.sqlite_conn.execute(
                            "SELECT id FROM memories WHERE content_hash = ? AND created_at > ? LIMIT 1",
                            (content_hash, since.isoformat())
                        ).fetchone()
                return row is not None
            else:
                query: Dict[str, Any] = {"content_hash": content_hash, "created_at": {"$gt": since}}
                if types_list:
                    query["memory_type"] = {"$in": types_list}
                doc = self.collection.find_one(query, {"_id": 1})
                return doc is not None

        except Exception:
            return False

    def get_recent_for_prompt(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent memories for system prompt, pinned first."""
        if not self.is_available():
            return []

        try:
            memories = []

            if self.backend == "sqlite":
                with self._sqlite_lock:
                    rows = self.sqlite_conn.execute(
                        "SELECT key, content, category, created_at, tags_json, access_count FROM memories "
                        "WHERE memory_type IN (?, ?) "
                        "ORDER BY "
                        "  CASE WHEN tags_json LIKE '%\"pinned\"%' THEN 0 ELSE 1 END, "
                        "  created_at DESC "
                        "LIMIT ?",
                        (MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, count)
                    ).fetchall()

                for row in rows:
                    memories.append({
                        "key": row["key"],
                        "content": row["content"],
                        "category": row["category"],
                        "created_at": row["created_at"],
                        "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
                    })
            else:
                # MongoDB: two-phase fetch - pinned first, then most recent
                pinned = list(self.collection.find(
                    {"memory_type": {"$in": [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM]}, "tags": "pinned"},
                    {"key": 1, "content": 1, "category": 1, "created_at": 1, "tags": 1, "access_count": 1}
                ).sort("created_at", DESCENDING).limit(count))

                remaining = count - len(pinned)
                pinned_keys = {doc["key"] for doc in pinned}
                others = []
                if remaining > 0:
                    others = list(self.collection.find(
                        {
                            "memory_type": {"$in": [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM]},
                            "key": {"$nin": list(pinned_keys)},
                        },
                        {"key": 1, "content": 1, "category": 1, "created_at": 1, "tags": 1, "access_count": 1}
                    ).sort("created_at", DESCENDING).limit(remaining))

                for doc in pinned + others:
                    memories.append({
                        "key": doc.get("key"),
                        "content": doc.get("content"),
                        "category": doc.get("category", "general"),
                        "created_at": self._serialize_dt(doc.get("created_at")),
                        "tags": doc.get("tags", []),
                    })

            return memories

        except Exception as e:
            logger.error(f"Get recent failed: {e}")
            return []
