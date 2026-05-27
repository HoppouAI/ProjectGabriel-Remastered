"""VRChat REST proxy endpoints used by the WebUI players panel."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..models import ModerationInput

router = APIRouter()
logger = logging.getLogger(__name__)

VRCHAT_BASE = "https://api.vrchat.cloud/api/1"
_VRCAPI_HEADERS = {"User-Agent": "ProjectGabriel/1.0", "Content-Type": "application/json"}
COOKIE_FILE = Path("data/vrchat_cookies.json")


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


@router.get("/api/vrchat/user/{user_id}")
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


@router.get("/api/vrchat/moderations/{user_id}")
async def vrchat_get_moderations(user_id: str):
    import aiohttp
    cookie = _get_vrchat_cookie_header()
    if not cookie:
        raise HTTPException(status_code=503, detail="No VRChat auth cookie available")
    url = f"{VRCHAT_BASE}/auth/user/playermoderations"
    logger.info(f"[GET_MODS] Fetching all moderations to filter by {user_id}")

    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers={**_VRCAPI_HEADERS, "Cookie": cookie}) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                logger.error(f"[GET_MODS] Failed with status {resp.status}: {data}")
                raise HTTPException(status_code=resp.status, detail=str(data))
            active = [
                entry.get("type") for entry in data
                if isinstance(entry, dict) and entry.get("targetUserId") == user_id
            ]
            logger.info(f"[GET_MODS] Active mod types for {user_id}: {active}")
            return {"active": active}


_VALID_MOD_TYPES = {
    "block", "hideAvatar", "interactOff", "interactOn",
    "mute", "muteChat", "showAvatar", "unmute", "unmuteChat",
}


@router.post("/api/vrchat/moderate")
async def vrchat_moderate(body: ModerationInput):
    import aiohttp
    if body.type not in _VALID_MOD_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid moderation type: {body.type}")
    cookie = _get_vrchat_cookie_header()
    if not cookie:
        raise HTTPException(status_code=503, detail="No VRChat auth cookie available")

    url = f"{VRCHAT_BASE}/auth/user/playermoderations"
    payload = {"moderated": body.moderated, "type": body.type}
    logger.info(f"[MODERATE] Applying {body.type} to {body.moderated}")

    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=payload, headers={**_VRCAPI_HEADERS, "Cookie": cookie}) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                logger.error(f"[MODERATE] Failed with status {resp.status}: {data}")
                raise HTTPException(status_code=resp.status, detail=str(data))
            return {"result": "ok", "type": body.type}


@router.put("/api/vrchat/unmoderate")
async def vrchat_unmoderate(body: ModerationInput):
    import aiohttp
    if body.type not in _VALID_MOD_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid moderation type: {body.type}")
    cookie = _get_vrchat_cookie_header()
    if not cookie:
        raise HTTPException(status_code=503, detail="No VRChat auth cookie available")

    url = f"{VRCHAT_BASE}/auth/user/unplayermoderate"
    payload = {"moderated": body.moderated, "type": body.type}
    logger.info(f"[UNMODERATE] Removing {body.type} from {body.moderated}")

    async with aiohttp.ClientSession() as s:
        async with s.put(url, json=payload, headers={**_VRCAPI_HEADERS, "Cookie": cookie}) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                logger.error(f"[UNMODERATE] Failed with status {resp.status}: {data}")
                raise HTTPException(status_code=resp.status, detail=str(data))
            return {"result": "ok", "type": body.type}
