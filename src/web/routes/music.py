"""Music playback + file management endpoints."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..models import MusicInput, VolumeInput
from ..shared import (
    ALLOWED_EXTENSIONS,
    ARCHIVE_EXTENSIONS,
    MAX_FILE_SIZE,
    MUSIC_DIR,
    SEVEN_ZIP_PATH,
    broadcast_state,
    shared_state,
)

router = APIRouter()


@router.get("/api/music-list")
async def music_list():
    files = []
    for f in sorted(MUSIC_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({"name": str(f.relative_to(MUSIC_DIR)), "path": str(f)})
    return {"files": files}


@router.post("/api/play-music")
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


@router.post("/api/pause-music")
async def pause_music():
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    audio_mgr.pause_music()
    await broadcast_state()
    return {"success": True}


@router.post("/api/resume-music")
async def resume_music():
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    audio_mgr.resume_music()
    await broadcast_state()
    return {"success": True}


@router.post("/api/stop-music")
async def stop_music():
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    audio_mgr.stop_music()
    await broadcast_state()
    return {"success": True}


@router.post("/api/set-volume")
async def set_volume(data: VolumeInput):
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    volume_int = max(0, min(100, int(data.volume * 100)))
    audio_mgr.set_music_volume(volume_int)
    return {"success": True, "volume": data.volume}


@router.post("/api/open-music-folder")
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


@router.get("/api/music-files")
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


@router.delete("/api/music-files/{file_path:path}")
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


@router.get("/api/music-folders")
async def list_music_folders():
    folders = [""]
    for d in sorted(MUSIC_DIR.rglob("*")):
        if d.is_dir():
            folders.append(str(d.relative_to(MUSIC_DIR)))
    return folders


@router.post("/api/music-upload")
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
