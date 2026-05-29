"""WebSocket endpoint + background music/state broadcast loops."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..shared import (
    broadcast_log,
    broadcast_state,
    get_full_state,
    shared_state,
    websocket_clients,
)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websocket_clients.append(ws)
    try:
        state = get_full_state()
        await ws.send_text(json.dumps({"type": "state", "data": state}))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in websocket_clients:
            websocket_clients.remove(ws)


_music_broadcast_task = None
_last_music_state = None
_state_broadcast_task = None
_last_state_payload: str | None = None


async def _music_broadcast_loop():
    global _last_music_state
    while True:
        await asyncio.sleep(1)
        audio_mgr = shared_state.get("audio_mgr")
        if not audio_mgr or not hasattr(audio_mgr, "get_music_progress"):
            if _last_music_state is not None:
                _last_music_state = None
                await broadcast_log({"type": "music_update", "content": "", "extra": {"playing": False}})
            continue

        prog = audio_mgr.get_music_progress()
        if prog:
            lyric = audio_mgr.get_current_lyric() if hasattr(audio_mgr, "get_current_lyric") else None
            is_playing = audio_mgr.is_music_playing() if hasattr(audio_mgr, "is_music_playing") else True
            await broadcast_log({"type": "music_update", "content": prog.get("song_name", ""), "extra": {
                "playing": is_playing,
                "song_name": prog.get("song_name", "Unknown"),
                "position": round(prog.get("position", 0), 1),
                "duration": round(prog.get("duration", 0), 1),
                "progress": round(prog.get("progress", 0), 3),
                "lyric": lyric,
            }})
            _last_music_state = True
        elif _last_music_state is not None:
            _last_music_state = None
            await broadcast_log({"type": "music_update", "content": "", "extra": {"playing": False}})


async def _state_broadcast_loop():
    """Push the full UI state whenever it actually changes."""
    global _last_state_payload
    while True:
        await asyncio.sleep(1)
        try:
            state = await asyncio.to_thread(get_full_state)
            payload = json.dumps(state, sort_keys=True)
            if payload != _last_state_payload:
                _last_state_payload = payload
                await broadcast_state(state)
        except Exception:
            pass


def start_music_broadcast():
    global _music_broadcast_task
    if _music_broadcast_task is None or _music_broadcast_task.done():
        _music_broadcast_task = asyncio.ensure_future(_music_broadcast_loop())


def start_state_broadcast():
    global _state_broadcast_task
    if _state_broadcast_task is None or _state_broadcast_task.done():
        _state_broadcast_task = asyncio.ensure_future(_state_broadcast_loop())
