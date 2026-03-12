import aiohttp
import json
import logging
import os
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)

API_URL = "https://myinstants.barricade.dev/search?q={}"

# Cache dir for downloaded sounds
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sfx", "myinstants")
os.makedirs(_CACHE_DIR, exist_ok=True)

_CACHE_FILE = os.path.join(_CACHE_DIR, "soundeffects.json")

# Accumulated sound registry: ID -> {title, mp3} (persists across sessions via JSON)
_known_sounds: dict[str, dict] = {}


def _load_cache():
    global _known_sounds
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _known_sounds = json.load(f)
            logger.info(f"Loaded {len(_known_sounds)} sounds from cache")
        except Exception as e:
            logger.warning(f"Failed to load sound cache: {e}")
            _known_sounds = {}


def _save_cache():
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_known_sounds, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save sound cache: {e}")


# Load cache on module import
_load_cache()


async def search_sounds(query: str, limit: int = 10) -> list[dict] | None:
    url = API_URL.format(quote(query))
    logger.debug(f"MyInstants API search URL: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                logger.debug(f"MyInstants API response status: {resp.status}")
                if resp.status != 200:
                    logger.warning(f"MyInstants API returned non-200 status: {resp.status}")
                    return None
                data = await resp.json()

        results = data.get("data", [])
        logger.debug(f"MyInstants API returned {len(results)} results")
        if not results:
            logger.warning(f"MyInstants API: no results for '{query}'")
            return None

        # Accumulate results for play-by-ID lookup (never cleared)
        sounds = []
        for item in results[:limit]:
            sid = item.get("id", "")
            title = item.get("title", "")
            mp3 = item.get("mp3", "")
            if sid and mp3:
                _known_sounds[sid] = {"title": title, "mp3": mp3}
                sounds.append({"id": sid, "title": title})

        _save_cache()
        logger.info(f"MyInstants search '{query}': {len(sounds)} results cached")
        return sounds
    except Exception as e:
        logger.error(f"MyInstants search failed: {e}")
        return None


def get_sound_url(sound_id: str) -> dict | None:
    # Check in-memory registry first
    entry = _known_sounds.get(sound_id)
    if entry:
        logger.debug(f"Found known sound ID '{sound_id}': {entry['title']}")
        return entry

    # Fallback: check if a cached file on disk matches the ID
    for fname in os.listdir(_CACHE_DIR):
        # Match if filename starts with the ID slug (e.g. 'laughing-track' from 'laughing-track-17221')
        name_no_ext = os.path.splitext(fname)[0]
        # The ID is like 'laughing-track-17221', the file might be 'laughing-track.mp3'
        id_slug = re.sub(r'-\d+$', '', sound_id)  # strip trailing number from ID
        if name_no_ext == id_slug or name_no_ext.replace('_', '-') == id_slug:
            filepath = os.path.join(_CACHE_DIR, fname)
            logger.info(f"Found cached file for ID '{sound_id}' on disk: {filepath}")
            return {"title": sound_id, "mp3": filepath, "_local": True}

    logger.warning(f"Sound ID '{sound_id}' not found in known sounds or cache")
    return None


def _url_to_filename(url: str) -> str:
    # Extract filename from URL, sanitize for filesystem
    name = url.rsplit("/", 1)[-1]
    name = re.sub(r'[^\w\-.]', '_', name)
    if not name.endswith(".mp3"):
        name += ".mp3"
    return name


async def download_sound(url: str) -> str | None:
    # Check cache: see if file already exists in sfx/myinstants/
    filename = _url_to_filename(url)
    cached_path = os.path.join(_CACHE_DIR, filename)
    if os.path.exists(cached_path):
        logger.debug(f"Sound cache hit: {url} -> {cached_path}")
        return cached_path

    logger.debug(f"Downloading sound from: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                logger.debug(f"Download response status: {resp.status}, content-type: {resp.headers.get('Content-Type', 'unknown')}")
                if resp.status != 200:
                    logger.warning(f"Sound download got status {resp.status} for {url}")
                    return None
                data = await resp.read()
                logger.debug(f"Downloaded {len(data)} bytes")
        with open(cached_path, "wb") as f:
            f.write(data)
        logger.debug(f"Sound saved and cached: {cached_path}")
        return cached_path
    except Exception as e:
        logger.error(f"Sound download failed: {e}")
        return None
