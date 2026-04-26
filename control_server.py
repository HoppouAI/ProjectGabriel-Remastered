"""Gabriel Control Panel - combined WebUI for session control and music management."""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

SESSION_HANDLE_FILE = Path("session_handle.txt")
MUSIC_DIR = Path("sfx/music")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
SUBTITLE_EXTENSIONS = {".srt", ".lrc"}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | SUBTITLE_EXTENSIONS
ARCHIVE_EXTENSIONS = {".zip", ".7z", ".rar"}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
STATIC_DIR = Path(__file__).parent / "webui"

MUSIC_DIR.mkdir(parents=True, exist_ok=True)

shared_state = {
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

console_logs = deque(maxlen=100)
websocket_clients: list[WebSocket] = []
_state_broadcast_task = None
_last_state_payload = None

app = FastAPI(title="Control Panel")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Helpers ---

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


def add_console_log(log_type: str, content: str, extra: dict = None) -> dict:
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


async def _state_broadcast_loop():
    """Broadcast full UI state whenever it changes, even if no route explicitly triggers it."""
    global _last_state_payload
    while True:
        await asyncio.sleep(1)
        try:
            # run in thread so blocking db/sync calls dont stall the event loop
            state = await asyncio.to_thread(get_full_state)
            payload = json.dumps(state, sort_keys=True)
            if payload != _last_state_payload:
                _last_state_payload = payload
                await broadcast_state(state)  # pass pre-computed state, avoid double get_full_state
        except Exception:
            pass


# --- Pydantic Models ---

class TextInput(BaseModel):
    text: str

class MusicInput(BaseModel):
    filename: str

class VolumeInput(BaseModel):
    volume: float

class PersonalityInput(BaseModel):
    personality: str

class EmotionInput(BaseModel):
    emotion: str

class MemoryCreateInput(BaseModel):
    key: str | None = None
    content: str
    category: str = "general"
    memory_type: str = "long_term"
    tags: list[str] | None = None

class MemoryUpdateInput(BaseModel):
    content: str | None = None
    category: str | None = None
    memory_type: str | None = None
    tags: list[str] | None = None

class MemoryPinInput(BaseModel):
    pin: bool = True


# --- Routes ---

@app.get("/")
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/overlay")
async def overlay():
    cfg = shared_state.get("config")
    if not cfg or not cfg.obs_enabled:
        raise HTTPException(status_code=404, detail="OBS overlay is disabled (obs.enabled: false)")
    html_path = STATIC_DIR / "overlay.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/overlay/config")
async def overlay_config():
    cfg = shared_state.get("config")
    if not cfg or not cfg.obs_enabled:
        raise HTTPException(status_code=404, detail="OBS overlay is disabled (obs.enabled: false)")
    html_path = STATIC_DIR / "overlay_config.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/overlay/music")
async def overlay_music():
    cfg = shared_state.get("config")
    if not cfg or not cfg.obs_enabled:
        raise HTTPException(status_code=404, detail="OBS overlay is disabled (obs.enabled: false)")
    html_path = STATIC_DIR / "overlay_music.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/state")
async def get_state():
    return get_full_state()


@app.get("/api/console-logs")
async def get_console_logs():
    return list(console_logs)


@app.get("/api/sevenzip-status")
async def sevenzip_status():
    return {"available": SEVEN_ZIP_PATH is not None, "path": SEVEN_ZIP_PATH}


@app.post("/api/reconnect")
async def reconnect():
    session = shared_state.get("session")
    if session and hasattr(session, "request_reconnect"):
        session.request_reconnect()
        return {"message": "Reconnect requested"}
    return {"message": "No active session to reconnect"}


@app.post("/api/clear-session")
async def clear_session():
    if SESSION_HANDLE_FILE.exists():
        SESSION_HANDLE_FILE.unlink()
    session = shared_state.get("session")
    if session:
        if hasattr(session, "_session_handle"):
            session._session_handle = None
        session.request_reconnect()
        return {"message": "Session cleared and reconnect requested"}
    return {"message": "Session handle cleared"}


@app.post("/api/toggle-mute")
async def toggle_mute():
    session = shared_state.get("session")
    if session and hasattr(session, "_mic_muted"):
        session._mic_muted = not session._mic_muted
        shared_state["mic_muted"] = session._mic_muted
        if hasattr(session, "osc") and session.osc:
            session.osc.toggle_voice()
    else:
        shared_state["mic_muted"] = not shared_state.get("mic_muted", False)
    await broadcast_state()
    return {"muted": shared_state["mic_muted"]}


@app.post("/api/send-text")
async def send_text(data: TextInput):
    session = shared_state.get("session")
    if not session:
        raise HTTPException(status_code=400, detail="No active session")
    if hasattr(session, "send_text"):
        await session.send_text(data.text)
        return {"message": "Text sent"}
    raise HTTPException(status_code=400, detail="Session does not support text input")


@app.post("/api/send-system-instruction")
async def send_system_instruction(data: TextInput):
    session = shared_state.get("session")
    if not session:
        raise HTTPException(status_code=400, detail="No active session")
    if not hasattr(session, "_session") or not session._session:
        raise HTTPException(status_code=400, detail="No active live session")
    try:
        from google.genai import types
        await session.send_client_content_safe(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"System instruction update - {data.text}")],
            ),
            turn_complete=True,
        )
        add_console_log("info", f"System instruction sent: {data.text[:80]}")
        return {"message": "System instruction sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/switch-personality")
async def switch_personality(data: PersonalityInput):
    personality_mgr = shared_state.get("personality_mgr")
    session = shared_state.get("session")
    if not personality_mgr:
        raise HTTPException(status_code=400, detail="Personality manager not available")
    try:
        result = personality_mgr.switch(data.personality)
        if "personality_prompt" in result and session and hasattr(session, "_session") and session._session:
            from google.genai import types
            prompt_text = result["personality_prompt"]
            await session.send_client_content_safe(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)],
                ),
                turn_complete=True,
            )
        await broadcast_state()
        return {k: v for k, v in result.items() if k != "personality_prompt"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/trigger-emotion")
async def trigger_emotion(data: EmotionInput):
    get_emotion_fn = shared_state.get("get_emotion_fn")
    emotion_mgr = None
    if get_emotion_fn:
        emotion_mgr = get_emotion_fn()
    if not emotion_mgr:
        raise HTTPException(status_code=400, detail="Emotion manager not available")
    try:
        result = emotion_mgr.play_emotion(data.emotion)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Music Endpoints ---

@app.get("/api/music-list")
async def music_list():
    files = []
    for f in sorted(MUSIC_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({"name": str(f.relative_to(MUSIC_DIR)), "path": str(f)})
    return {"files": files}


@app.post("/api/play-music")
async def play_music(data: MusicInput):
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    try:
        audio_mgr.play_music(data.filename)
        await broadcast_state()
        return {"success": True, "filename": data.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pause-music")
async def pause_music():
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    audio_mgr.pause_music()
    await broadcast_state()
    return {"success": True}


@app.post("/api/resume-music")
async def resume_music():
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    audio_mgr.resume_music()
    await broadcast_state()
    return {"success": True}


@app.post("/api/stop-music")
async def stop_music():
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    audio_mgr.stop_music()
    await broadcast_state()
    return {"success": True}


@app.post("/api/set-volume")
async def set_volume(data: VolumeInput):
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    volume_int = max(0, min(100, int(data.volume * 100)))
    audio_mgr.set_music_volume(volume_int)
    return {"success": True, "volume": data.volume}


@app.post("/api/open-music-folder")
async def open_music_folder():
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    path = str(MUSIC_DIR.resolve())
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return {"success": True, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Music File Management ---

@app.get("/api/music-files")
async def list_music_files():
    files = []
    for f in sorted(MUSIC_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            rel = f.relative_to(MUSIC_DIR)
            stat = f.stat()
            files.append({
                "name": str(rel),
                "display_name": f.name,
                "folder": str(rel.parent) if str(rel.parent) != "." else "",
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return files


@app.delete("/api/music-files/{file_path:path}")
async def delete_music_file(file_path: str):
    target = MUSIC_DIR / file_path
    if not target.resolve().is_relative_to(MUSIC_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    parent = target.parent
    while parent != MUSIC_DIR and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent
    return {"message": f"Deleted {file_path}"}


@app.get("/api/music-folders")
async def list_music_folders():
    folders = [""]
    for d in sorted(MUSIC_DIR.rglob("*")):
        if d.is_dir():
            folders.append(str(d.relative_to(MUSIC_DIR)))
    return folders


@app.post("/api/music-upload")
async def upload_music(files: list[UploadFile] = File(...), folder: str = ""):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    safe_folder = Path(folder).as_posix().strip("/")
    target_dir = MUSIC_DIR / safe_folder if safe_folder else MUSIC_DIR

    if not target_dir.resolve().is_relative_to(MUSIC_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid folder path")

    target_dir.mkdir(parents=True, exist_ok=True)

    uploaded: list[str] = []
    extracted: list[str] = []
    errors: list[str] = []

    for file in files:
        if file.size and file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"{file.filename}: file too large (max 2GB)")

        ext = Path(file.filename).suffix.lower()
        if ext in ARCHIVE_EXTENSIONS:
            result = await _extract_archive(file, target_dir)
            extracted.extend(result.get("files", []))
            errors.extend(result.get("errors", []))
            continue

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

        dest = target_dir / file.filename
        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        uploaded.append(str(dest.relative_to(MUSIC_DIR)))

    total_saved = len(uploaded) + len(extracted)
    return {
        "message": f"Uploaded {total_saved} file(s)",
        "uploaded": uploaded,
        "extracted": extracted,
        "errors": errors,
    }


async def _extract_archive(file: UploadFile, target_dir: Path):
    if not SEVEN_ZIP_PATH:
        raise HTTPException(status_code=400, detail="7-Zip not found, cannot extract archives")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [SEVEN_ZIP_PATH, "x", tmp_path, f"-o{target_dir}", "-y", "-aoa"],
            capture_output=True, text=True, timeout=300,
        )

        extracted = []
        errors = []
        for f in target_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                extracted.append(str(f.relative_to(MUSIC_DIR)))

        if result.returncode != 0:
            errors.append(f"7-Zip warning: {result.stderr[:200]}")

        return {
            "message": f"Extracted {len(extracted)} files from {file.filename}",
            "extracted_count": len(extracted),
            "files": extracted,
            "errors": errors,
        }
    finally:
        os.unlink(tmp_path)


# --- Memory Management API ---

def _get_memory_mgr():
    mgr = shared_state.get("memory_mgr")
    if not mgr or not mgr.is_available():
        raise HTTPException(status_code=503, detail="Memory system unavailable")
    return mgr


@app.get("/api/memories")
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


@app.post("/api/memories")
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


@app.get("/api/memories/stats")
async def memory_stats():
    mgr = _get_memory_mgr()
    res = await asyncio.to_thread(mgr.stats)
    if not res.get("success"):
        raise HTTPException(status_code=500, detail=res.get("message", "Unknown error"))
    return res["stats"]


@app.get("/api/memories/{key}")
async def read_memory(key: str):
    mgr = _get_memory_mgr()
    res = await asyncio.to_thread(mgr.read, key)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("message", "Not found"))
    return res["memory"]


@app.put("/api/memories/{key}")
async def update_memory(key: str, body: MemoryUpdateInput):
    mgr = _get_memory_mgr()
    res = await asyncio.to_thread(
        mgr.update,
        key, body.content, body.category, body.memory_type, body.tags
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message", "Update failed"))
    return {"result": "ok"}


@app.delete("/api/memories/{key}")
async def delete_memory(key: str):
    mgr = _get_memory_mgr()
    res = await asyncio.to_thread(mgr.delete, key)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("message", "Not found"))
    return {"result": "ok"}


@app.post("/api/memories/{key}/pin")
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


# --- VRChat Proxy Endpoints ---

VRCHAT_BASE = "https://api.vrchat.cloud/api/1"
_VRCAPI_HEADERS = {"User-Agent": "ProjectGabriel/1.0", "Content-Type": "application/json"}


def _get_vrchat_cookie_header() -> str:
    if COOKIE_FILE.exists():
        try:
            data = json.loads(COOKIE_FILE.read_text(encoding="utf-8"))
            parts = []
            if data.get("auth"):
                parts.append(f"auth={data['auth']}")
            if data.get("twoFactorAuth"):
                parts.append(f"twoFactorAuth={data['twoFactorAuth']}")
            return "; ".join(parts)
        except Exception:
            pass
    return ""


COOKIE_FILE = Path("data/vrchat_cookies.json")


@app.get("/api/vrchat/user/{user_id}")
async def vrchat_get_user(user_id: str):
    import aiohttp
    cookie = _get_vrchat_cookie_header()
    if not cookie:
        raise HTTPException(status_code=503, detail="No VRChat auth cookie available")
    url = f"{VRCHAT_BASE}/users/{user_id}"
    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers={**_VRCAPI_HEADERS, "Cookie": cookie}) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=str(data))
            return {
                "id": data.get("id"),
                "displayName": data.get("displayName"),
                "bio": data.get("bio"),
                "status": data.get("status"),
                "statusDescription": data.get("statusDescription"),
                "currentAvatarThumbnailImageUrl": data.get("currentAvatarThumbnailImageUrl"),
                "profilePicOverride": data.get("profilePicOverride"),
                "isFriend": data.get("isFriend"),
                "last_platform": data.get("last_platform"),
            }


class ModerationInput(BaseModel):
    moderated: str
    type: str


@app.get("/api/vrchat/moderations/{user_id}")
async def vrchat_get_moderations(user_id: str):
    import aiohttp
    import logging
    logger = logging.getLogger(__name__)
    
    cookie = _get_vrchat_cookie_header()
    if not cookie:
        raise HTTPException(status_code=503, detail="No VRChat auth cookie available")
    # fetch all our moderations then filter in python - querystring targetUserId causes 403
    url = f"{VRCHAT_BASE}/auth/user/playermoderations"
    logger.info(f"[GET_MODS] Fetching all moderations to filter by {user_id}")
    
    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers={**_VRCAPI_HEADERS, "Cookie": cookie}) as resp:
            data = await resp.json(content_type=None)
            logger.info(f"[GET_MODS] VRChat response status: {resp.status}")
            logger.info(f"[GET_MODS] Total moderations in response: {len(data) if isinstance(data, list) else 'not a list'}")
            
            if resp.status != 200:
                logger.error(f"[GET_MODS] Failed with status {resp.status}: {data}")
                raise HTTPException(status_code=resp.status, detail=str(data))
            
            active = [
                entry.get("type") for entry in data
                if isinstance(entry, dict) and entry.get("targetUserId") == user_id
            ]
            logger.info(f"[GET_MODS] Active mod types for {user_id}: {active}")
            matching = [e for e in data if isinstance(e, dict) and e.get('targetUserId') == user_id]
            if matching:
                logger.info(f"[GET_MODS] Moderation entries for {user_id}: {matching}")
            return {"active": active}


@app.post("/api/vrchat/moderate")
async def vrchat_moderate(body: ModerationInput):
    import aiohttp
    import logging
    logger = logging.getLogger(__name__)
    
    valid_types = {"block", "hideAvatar", "interactOff", "interactOn", "mute", "muteChat", "showAvatar", "unmute", "unmuteChat"}
    if body.type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid moderation type: {body.type}")
    cookie = _get_vrchat_cookie_header()
    if not cookie:
        raise HTTPException(status_code=503, detail="No VRChat auth cookie available")
    
    url = f"{VRCHAT_BASE}/auth/user/playermoderations"
    payload = {"moderated": body.moderated, "type": body.type}
    
    logger.info(f"[MODERATE] Applying {body.type} to {body.moderated}")
    logger.info(f"[MODERATE] Payload: {payload}")
    
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=payload, headers={**_VRCAPI_HEADERS, "Cookie": cookie}) as resp:
            data = await resp.json(content_type=None)
            logger.info(f"[MODERATE] VRChat response status: {resp.status}")
            logger.info(f"[MODERATE] VRChat response data: {data}")
            
            if resp.status != 200:
                logger.error(f"[MODERATE] Failed with status {resp.status}: {data}")
                raise HTTPException(status_code=resp.status, detail=str(data))
            
            logger.info(f"[MODERATE] Success - applied {body.type}")
            return {"result": "ok", "type": body.type}


@app.put("/api/vrchat/unmoderate")
async def vrchat_unmoderate(body: ModerationInput):
    import aiohttp
    import logging
    logger = logging.getLogger(__name__)
    
    valid_types = {"block", "hideAvatar", "interactOff", "interactOn", "mute", "muteChat", "showAvatar", "unmute", "unmuteChat"}
    if body.type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid moderation type: {body.type}")
    cookie = _get_vrchat_cookie_header()
    if not cookie:
        raise HTTPException(status_code=503, detail="No VRChat auth cookie available")
    
    url = f"{VRCHAT_BASE}/auth/user/unplayermoderate"
    payload = {"moderated": body.moderated, "type": body.type}
    
    logger.info(f"[UNMODERATE] Removing {body.type} from {body.moderated}")
    logger.info(f"[UNMODERATE] Payload: {payload}")
    
    async with aiohttp.ClientSession() as s:
        async with s.put(url, json=payload, headers={**_VRCAPI_HEADERS, "Cookie": cookie}) as resp:
            data = await resp.json(content_type=None)
            logger.info(f"[UNMODERATE] VRChat response status: {resp.status}")
            logger.info(f"[UNMODERATE] VRChat response data: {data}")
            
            if resp.status != 200:
                logger.error(f"[UNMODERATE] Failed with status {resp.status}: {data}")
                raise HTTPException(status_code=resp.status, detail=str(data))
            
            logger.info(f"[UNMODERATE] Success - removed {body.type}")
            return {"result": "ok", "type": body.type}


# --- WebSocket ---

@app.websocket("/ws")
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


# --- Music broadcast loop ---

_music_broadcast_task = None
_last_music_state = None  # Track previous state to detect changes


async def _music_broadcast_loop():
    """Broadcast music_update events to WebSocket clients (not stored in console log)."""
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
            lyric = None
            if hasattr(audio_mgr, "get_current_lyric"):
                lyric = audio_mgr.get_current_lyric()
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


def start_music_broadcast():
    global _music_broadcast_task
    if _music_broadcast_task is None or _music_broadcast_task.done():
        _music_broadcast_task = asyncio.ensure_future(_music_broadcast_loop())


def start_state_broadcast():
    global _state_broadcast_task
    if _state_broadcast_task is None or _state_broadcast_task.done():
        _state_broadcast_task = asyncio.ensure_future(_state_broadcast_loop())


@app.on_event("startup")
async def _on_startup():
    start_music_broadcast()
    start_state_broadcast()


# --- Run ---

def run_control_server(host: str = "0.0.0.0", port: int = 8766):
    cfg = shared_state.get("config")
    name = cfg.app_name if cfg else "Control Panel"
    print(f"{name} Control Panel")
    print(f"Open http://localhost:{port} in your browser")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_control_server()
