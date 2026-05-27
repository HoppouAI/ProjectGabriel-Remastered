"""Spatial mapping + waypoints endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from ..models import (
    CellEditIn,
    CellsBulkEditIn,
    CleanupJumpArtifactsIn,
    CleanupStraysIn,
    GotoIn,
    MappingExploreInput,
    MappingManualInput,
    MappingSettingsIn,
    MappingStartInput,
    PathfindInput,
    WaypointCreateInput,
)
from ..shared import shared_state

router = APIRouter()


def _get_mapping():
    ms = shared_state.get("mapping_service")
    if ms is None:
        raise HTTPException(status_code=503, detail="mapping service not available")
    return ms


@router.get("/api/mapping/state")
async def mapping_state():
    return _get_mapping().get_state()


@router.get("/api/mapping/world")
async def mapping_world():
    return _get_mapping().get_world_cells()


@router.post("/api/mapping/start")
async def mapping_start(body: MappingStartInput):
    ms = _get_mapping()
    state = await asyncio.to_thread(ms.start, explore=body.explore)
    if not state.get("running") and state.get("last_error"):
        raise HTTPException(status_code=400, detail=state["last_error"])
    return state


@router.post("/api/mapping/stop")
async def mapping_stop():
    return await asyncio.to_thread(_get_mapping().stop)


@router.post("/api/mapping/explore")
async def mapping_explore(body: MappingExploreInput):
    return _get_mapping().set_explore(body.enabled)


@router.post("/api/mapping/manual")
async def mapping_manual(body: MappingManualInput):
    return _get_mapping().set_manual_mapping(body.enabled)


@router.post("/api/mapping/pathfind")
async def mapping_pathfind(body: PathfindInput):
    ms = _get_mapping()
    if body.waypoint:
        return ms.pathfind_to_waypoint(body.waypoint)
    return ms.pathfind_to(body.x, body.y, body.z)


@router.get("/api/waypoints")
async def waypoints_list():
    ms = _get_mapping()
    return {"waypoints": ms.list_waypoints(), "world": ms.get_state().get("world")}


@router.post("/api/waypoints")
async def waypoints_create(body: WaypointCreateInput):
    ms = _get_mapping()
    try:
        return ms.add_waypoint(body.name, body.note)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/api/waypoints/{name}")
async def waypoints_delete(name: str):
    ms = _get_mapping()
    if not ms.remove_waypoint(name):
        raise HTTPException(status_code=404, detail="waypoint not found")
    return {"result": "ok"}


@router.get("/api/mapping/worlds")
async def mapping_worlds():
    return {"worlds": _get_mapping().list_worlds()}


@router.post("/api/mapping/settings")
async def mapping_settings(payload: MappingSettingsIn):
    ms = _get_mapping()
    settings = await asyncio.to_thread(
        ms.update_settings,
        tick_hz=payload.tick_hz,
        force_run=payload.force_run,
        manual_wall_distance=payload.manual_wall_distance,
        manual_wall_ratio=payload.manual_wall_ratio,
    )
    return {"settings": settings, "state": ms.get_state()}


@router.post("/api/mapping/goto")
async def mapping_goto(payload: GotoIn):
    ms = _get_mapping()
    if payload.waypoint:
        return await asyncio.to_thread(ms.goto_waypoint, payload.waypoint)
    if payload.x is None or payload.y is None or payload.z is None:
        raise HTTPException(status_code=400, detail="provide 'waypoint' or 'x','y','z'")
    return await asyncio.to_thread(ms.goto_xyz, payload.x, payload.y, payload.z)


@router.post("/api/mapping/cancel_goto")
async def mapping_cancel_goto():
    return await asyncio.to_thread(_get_mapping().cancel_goto)


@router.post("/api/mapping/cell")
async def mapping_cell_edit(payload: CellEditIn):
    ms = _get_mapping()
    try:
        return await asyncio.to_thread(ms.edit_cell, payload.sx, payload.sy, payload.sz, payload.kind)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/mapping/cells/bulk")
async def mapping_cells_bulk_edit(payload: CellsBulkEditIn):
    ms = _get_mapping()
    if not payload.cells:
        return {"result": "ok", "kind": payload.kind, "applied": 0, "total": 0}
    try:
        cells = [(int(c[0]), int(c[1]), int(c[2])) for c in payload.cells if len(c) >= 3]
        return await asyncio.to_thread(ms.edit_cells_bulk, cells, payload.kind)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/mapping/world")
async def mapping_delete_world(world: str | None = None):
    ms = _get_mapping()
    try:
        return await asyncio.to_thread(ms.delete_world, world)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mapping/cleanup_strays")
async def mapping_cleanup_strays(payload: CleanupStraysIn):
    ms = _get_mapping()
    return await asyncio.to_thread(
        ms.cleanup_strays,
        min_component_size=int(payload.min_component_size),
        dry_run=bool(payload.dry_run),
    )


@router.post("/api/mapping/cleanup_jump_artifacts")
async def mapping_cleanup_jump_artifacts(payload: CleanupJumpArtifactsIn):
    ms = _get_mapping()
    return await asyncio.to_thread(ms.cleanup_jump_artifacts, dry_run=bool(payload.dry_run))
