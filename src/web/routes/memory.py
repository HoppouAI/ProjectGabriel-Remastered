"""Memory CRUD endpoints."""
from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException

from ..models import MemoryCreateInput, MemoryPinInput, MemoryUpdateInput
from ..shared import shared_state

router = APIRouter()


def _get_memory_mgr():
    mgr = shared_state.get("memory_mgr")
    if not mgr or not mgr.is_available():
        raise HTTPException(status_code=503, detail="Memory system unavailable")
    return mgr


@router.get("/api/memories")
async def list_memories(
    memory_type: str | None = None,
    category: str | None = None,
    search: str | None = None,
    limit: int = 50,
):
    mgr = _get_memory_mgr()
    if search:
        res = await asyncio.to_thread(mgr.search, search, memory_type, limit)
    else:
        res = await asyncio.to_thread(mgr.list_memories, category, memory_type, limit)
    if not res.get("success"):
        raise HTTPException(status_code=500, detail=res.get("message", "Unknown error"))
    return {"memories": res.get("memories", []), "count": res.get("count", 0)}


@router.post("/api/memories")
async def create_memory(body: MemoryCreateInput):
    mgr = _get_memory_mgr()
    key = body.key or f"webui_{int(datetime.utcnow().timestamp() * 1000)}"
    res = await asyncio.to_thread(
        mgr.save,
        key, body.content, body.category, body.memory_type, body.tags
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message", "Create failed"))
    return {"result": "ok", "key": res.get("key")}


@router.get("/api/memories/stats")
async def memory_stats():
    mgr = _get_memory_mgr()
    res = await asyncio.to_thread(mgr.stats)
    if not res.get("success"):
        raise HTTPException(status_code=500, detail=res.get("message", "Unknown error"))
    return res["stats"]


@router.get("/api/memories/{key}")
async def read_memory(key: str):
    mgr = _get_memory_mgr()
    res = await asyncio.to_thread(mgr.read, key)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("message", "Not found"))
    return res["memory"]


@router.put("/api/memories/{key}")
async def update_memory(key: str, body: MemoryUpdateInput):
    mgr = _get_memory_mgr()
    res = await asyncio.to_thread(
        mgr.update,
        key, body.content, body.category, body.memory_type, body.tags
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message", "Update failed"))
    return {"result": "ok"}


@router.delete("/api/memories/{key}")
async def delete_memory(key: str):
    mgr = _get_memory_mgr()
    res = await asyncio.to_thread(mgr.delete, key)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("message", "Not found"))
    return {"result": "ok"}


@router.post("/api/memories/{key}/pin")
async def pin_memory(key: str, body: MemoryPinInput):
    mgr = _get_memory_mgr()
    read_res = await asyncio.to_thread(mgr.read, key)
    if not read_res.get("success"):
        raise HTTPException(status_code=404, detail=read_res.get("message", "Not found"))
    mem = read_res["memory"]
    tags = mem.get("tags", [])
    if body.pin and "pinned" not in tags:
        tags.append("pinned")
    elif not body.pin and "pinned" in tags:
        tags = [t for t in tags if t != "pinned"]
    res = await asyncio.to_thread(mgr.update, key, None, None, None, tags)
    if not res.get("success"):
        raise HTTPException(status_code=500, detail=res.get("message", "Pin failed"))
    return {"result": "ok", "pinned": body.pin}
