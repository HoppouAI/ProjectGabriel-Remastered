import aiohttp
import logging
import re
import tempfile

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.myinstants.com/en/search/?name={}"
BASE_URL = "https://www.myinstants.com"


async def search_sound(query: str) -> dict | None:
    url = SEARCH_URL.format(query.replace(" ", "+"))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()

        sound_pattern = r"onclick=\"play\('(/media/sounds/[^']+)'\)\""
        matches = re.findall(sound_pattern, html)
        if not matches:
            return None

        name_pattern = r'class="instant-link"[^>]*>([^<]+)</a>'
        names = re.findall(name_pattern, html)

        sound_url = BASE_URL + matches[0]
        sound_name = names[0].strip() if names else query
        return {"name": sound_name, "url": sound_url}
    except Exception as e:
        logger.error(f"MyInstants search failed: {e}")
        return None


async def download_sound(url: str) -> str | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(data)
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.error(f"Sound download failed: {e}")
        return None
