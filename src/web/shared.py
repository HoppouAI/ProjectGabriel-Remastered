"""Shared state, broadcast helpers and computed UI state for the control panel."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import WebSocket

SESSION_HANDLE_FILE = Path("session_handle.txt")
MUSIC_DIR = Path("sfx/music")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
SUBTITLE_EXTENSIONS = {".srt", ".lrc"}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | SUBTITLE_EXTENSIONS
ARCHIVE_EXTENSIONS = {".zip", ".7z", ".rar"}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

MUSIC_DIR.mkdir(parents=True, exist_ok=True)

# all the singletons get bolted on here at startup. anyone with a reference to
# this dict can read/write the live session, audio manager etc.
shared_state: dict[str, Any] = {
    "session": None,
    "usage_metadata": None,
    "is_connected": False,
    "mic_muted": False,
    "last_activity": None,
    "personality_mgr": None,
    "audio_mgr": None,
    "memory_mgr": None,
    "get_emotion_fn": None,
    "config": None,
}

console_logs: deque = deque(maxlen=100)
websocket_clients: list[WebSocket] = []

# set by routes/websocket.py at startup
_state_broadcast_task = None
_last_state_payload: str | None = None


def find_7zip():
    candidates = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    sz = shutil.which("7z")
    if sz:
        return sz
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


SEVEN_ZIP_PATH = find_7zip()


def add_console_log(log_type: str, content: str, extra: dict | None = None) -> dict:
    entry = {
        "type": log_type,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "extra": extra or {},
    }
    console_logs.append(entry)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_log(entry))
    except RuntimeError:
        pass
    return entry


def get_session_handle_info() -> dict:
    if not SESSION_HANDLE_FILE.exists():
        return {"exists": False}
    try:
        data = json.loads(SESSION_HANDLE_FILE.read_text())
        saved_at = datetime.fromisoformat(data.get("saved_at", ""))
        age = (datetime.now() - saved_at).total_seconds() / 60
        return {
            "exists": True,
            "handle": (data.get("handle", "")[:20] + "...") if data.get("handle") else None,
            "saved_at": data.get("saved_at"),
            "age_minutes": round(age, 1),
        }
    except Exception:
        return {"exists": True, "error": "Could not parse handle file"}


async def broadcast_state(state: dict | None = None):
    if state is None:
        state = await asyncio.to_thread(get_full_state)
    msg = json.dumps({"type": "state", "data": state})
    disconnected = []
    for ws in websocket_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in websocket_clients:
            websocket_clients.remove(ws)


async def broadcast_log(entry: dict):
    msg = json.dumps({"type": "log", "data": entry})
    disconnected = []
    for ws in websocket_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in websocket_clients:
            websocket_clients.remove(ws)


def get_full_state() -> dict:
    session = shared_state.get("session")
    is_connected = shared_state.get("is_connected", False)
    mic_muted = shared_state.get("mic_muted", False)
    usage_metadata = shared_state.get("usage_metadata")

    if session:
        is_connected = getattr(session, "_session", None) is not None
        mic_muted = getattr(session, "_mic_muted", False)
        um = getattr(session, "_usage_metadata", None)
        if um and isinstance(um, dict):
            usage_metadata = {
                "prompt_tokens": um.get("prompt_tokens"),
                "response_tokens": um.get("response_tokens"),
                "total_tokens": um.get("total_tokens"),
                "tool_calls": um.get("tool_calls"),
            }

    personalities = []
    current_personality = None
    personality_mgr = shared_state.get("personality_mgr")
    if personality_mgr:
        try:
            pl = personality_mgr.list_personalities()
            personalities = pl.get("personalities", [])
        except Exception:
            pass
        try:
            cp = personality_mgr.get_current()
            current_personality = cp.get("id")
        except Exception:
            pass

    music_progress = None
    audio_mgr = shared_state.get("audio_mgr")
    if audio_mgr and hasattr(audio_mgr, "get_music_progress"):
        prog = audio_mgr.get_music_progress()
        if prog:
            music_progress = {
                "is_playing": audio_mgr.is_music_playing() if hasattr(audio_mgr, "is_music_playing") else True,
                "song_name": prog.get("song_name"),
                "position": prog.get("position", 0),
                "duration": prog.get("duration", 0),
            }

    recent_memories = None
    memory_mgr = shared_state.get("memory_mgr")
    if memory_mgr and hasattr(memory_mgr, "list_memories"):
        try:
            recent_memories = memory_mgr.list_memories(limit=10)
        except Exception:
            pass

    cfg = shared_state.get("config")
    app_name = cfg.app_name if cfg else "Gabriel"

    vrchat_info = None
    im = shared_state.get("instance_monitor")
    if im:
        location = im.current_location
        players = im.get_players()
        vrchat_info = {
            "is_in_world": im.is_in_world,
            "location": location or None,
            "player_count": len(players),
            "players": [{"name": p.get("name", "Unknown"), "id": p.get("id", "")} for p in players],
        }

    return {
        "app_name": app_name,
        "is_connected": is_connected,
        "mic_muted": mic_muted,
        "usage_metadata": usage_metadata,
        "last_activity": shared_state.get("last_activity"),
        "session_handle": get_session_handle_info(),
        "personalities": personalities,
        "current_personality": current_personality,
        "music_progress": music_progress,
        "recent_memories": recent_memories,
        "vrchat": vrchat_info,
    }
