"""Static page routes: / index, /overlay/*."""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..shared import shared_state

STATIC_DIR = Path(__file__).parent.parent.parent.parent / "webui"

router = APIRouter()


def _require_obs():
    cfg = shared_state.get("config")
    if not cfg or not cfg.obs_enabled:
        raise HTTPException(status_code=404, detail="OBS overlay is disabled (obs.enabled: false)")


@router.get("/")
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/overlay")
async def overlay():
    _require_obs()
    return HTMLResponse((STATIC_DIR / "overlay.html").read_text(encoding="utf-8"))


@router.get("/overlay/config")
async def overlay_config():
    _require_obs()
    return HTMLResponse((STATIC_DIR / "overlay_config.html").read_text(encoding="utf-8"))


@router.get("/overlay/music")
async def overlay_music():
    _require_obs()
    return HTMLResponse((STATIC_DIR / "overlay_music.html").read_text(encoding="utf-8"))
