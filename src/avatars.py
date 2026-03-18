import json
import uuid
import logging
import asyncio
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

AVATARS_CACHE_FILE = Path("data/avatars.json")
VRCX_ID_FILE = Path("data/vrcx_id.txt")
VRCX_SEARCH_URL = "https://api.avtrdb.com/v3/avatar/search/vrcx"

# Maps disambiguated display names (e.g. "Gabriel-2") to avatar IDs from the last search
_last_search_map: dict[str, str] = {}


def _get_vrcx_id() -> str:
    VRCX_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    if VRCX_ID_FILE.exists():
        stored = VRCX_ID_FILE.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    new_id = str(uuid.uuid4())
    VRCX_ID_FILE.write_text(new_id, encoding="utf-8")
    logger.info(f"Generated new VRCX-ID: {new_id}")
    return new_id


def _load_cache() -> dict:
    if AVATARS_CACHE_FILE.exists():
        try:
            return json.loads(AVATARS_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load avatar cache: {e}")
    return {}


def _save_cache(cache: dict):
    AVATARS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AVATARS_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _add_to_cache(avatars: list[dict]):
    cache = _load_cache()
    for av in avatars:
        aid = av.get("id", "")
        if aid:
            cache[aid] = {
                "id": aid,
                "name": av.get("name", "Unknown"),
                "authorName": av.get("authorName", ""),
            }
    _save_cache(cache)


def lookup_cached_avatar(name_or_id: str) -> dict | None:
    cache = _load_cache()
    if name_or_id in cache:
        return cache[name_or_id]
    name_lower = name_or_id.lower()
    for aid, av in cache.items():
        if av.get("name", "").lower() == name_lower:
            return av
    for aid, av in cache.items():
        if name_lower in av.get("name", "").lower():
            return av
    return None


async def search_avatars(query: str, max_results: int = 25) -> list[dict]:
    vrcx_id = _get_vrcx_id()
    headers = {
        "Referer": "https://vrcx.app",
        "VRCX-ID": vrcx_id,
        "User-Agent": "VRCX",
    }
    params = {"search": query, "n": "5000"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                VRCX_SEARCH_URL, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.error(f"VRCX avatar search failed: HTTP {resp.status}")
                    return []
                data = await resp.json()
    except Exception as e:
        logger.error(f"VRCX avatar search error: {e}")
        return []

    if not isinstance(data, list):
        logger.warning(f"VRCX search returned non-list: {type(data)}")
        return []

    # Filter: must have both PC and Android ratings (cross-platform only)
    crossplatform = []
    for av in data:
        perf = av.get("performance") or {}
        if perf.get("pc_rating") and perf.get("android_rating"):
            crossplatform.append(av)

    _add_to_cache(crossplatform)

    # Build results with disambiguated names for duplicates
    global _last_search_map
    _last_search_map = {}
    name_counts: dict[str, int] = {}
    name_totals: dict[str, int] = {}

    # First pass: count how many times each name appears
    for av in crossplatform[:max_results]:
        n = av.get("name", "Unknown")
        name_totals[n] = name_totals.get(n, 0) + 1

    results = []
    for av in crossplatform[:max_results]:
        raw_name = av.get("name", "Unknown")
        aid = av.get("id", "")
        author = av.get("authorName", "")

        if name_totals.get(raw_name, 1) > 1:
            name_counts[raw_name] = name_counts.get(raw_name, 0) + 1
            display_name = f"{raw_name}-{name_counts[raw_name]}"
        else:
            display_name = raw_name

        _last_search_map[display_name.lower()] = aid
        results.append({
            "id": aid,
            "name": display_name,
            "authorName": author,
        })

    logger.info(f"Avatar search '{query}': {len(data)} total, {len(crossplatform)} cross-platform, returning {len(results)}")
    return results


async def switch_avatar(vrchat_api, name_or_id: str) -> dict:
    avatar_id = name_or_id
    avatar_name = name_or_id

    if not name_or_id.startswith("avtr_"):
        # Check disambiguation map from last search first
        mapped_id = _last_search_map.get(name_or_id.lower())
        if mapped_id:
            avatar_id = mapped_id
            logger.info(f"Resolved '{name_or_id}' to ID '{avatar_id}' from search map")
        else:
            cached = lookup_cached_avatar(name_or_id)
            if cached:
                avatar_id = cached["id"]
                avatar_name = cached.get("name", name_or_id)
                logger.info(f"Resolved avatar name '{name_or_id}' to ID '{avatar_id}' from cache")
            else:
                results = await search_avatars(name_or_id, max_results=1)
                if results:
                    avatar_id = results[0]["id"]
                    avatar_name = results[0]["name"]
                    logger.info(f"Resolved avatar name '{name_or_id}' to ID '{avatar_id}' via search")
                else:
                    return {"result": "error", "message": f"No avatar found matching '{name_or_id}'"}

    result = await vrchat_api.select_avatar(avatar_id)
    if result.get("error"):
        return {"result": "error", "message": result["error"]}

    return {"result": "ok", "avatar_id": avatar_id, "avatar_name": avatar_name}
