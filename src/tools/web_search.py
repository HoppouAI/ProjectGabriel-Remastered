import logging
import aiohttp
from urllib.parse import quote
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)

JINA_READER_URL = "https://r.jina.ai/"
DDG_SEARCH_URL = "https://html.duckduckgo.com/html/?q="
MAX_SEARCH_CONTENT = 4000
MAX_READ_CONTENT = 8000


@register_tool
class WebSearchTools(BaseTool):
    tool_key = "web_search"

    def declarations(self, config=None):
        if config and not config.get("web_search", "enabled", default=False):
            return []
        # Only expose web search on 3.1 models (2.5 models have built-in google_search)
        if config and not config.is_31_model:
            return []
        return [
            types.FunctionDeclaration(
                name="webSearch",
                description=(
                    "Search the web for current information. Returns top results with "
                    "titles, URLs, and content snippets.\n"
                    "**Invocation Condition:** Call when you need current/live information "
                    "that is beyond your knowledge, when asked to search or look something up, "
                    "or when you need to verify facts with up-to-date sources."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "The search query to look up on the web",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="readWebpage",
                description=(
                    "Read and extract the main content from a specific URL/webpage. "
                    "Returns the page content in clean readable text.\n"
                    "**Invocation Condition:** Call when you need to read a specific webpage "
                    "URL, or when someone shares a link and asks you to read/summarize it."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "url": {
                            "type": "STRING",
                            "description": "The full URL to read (must start with http:// or https://)",
                        },
                    },
                    "required": ["url"],
                },
            ),
        ]

    async def handle(self, name, args):
        config = self.handler.config
        api_key = config.get("web_search", "jina_api_key", default=None)

        if name == "webSearch":
            query = args.get("query", "")
            if not query:
                return {"result": "error", "message": "Query is required"}
            return await self._search(query, api_key)
        elif name == "readWebpage":
            url = args.get("url", "")
            if not url:
                return {"result": "error", "message": "URL is required"}
            if not url.startswith(("http://", "https://")):
                return {"result": "error", "message": "URL must start with http:// or https://"}
            return await self._read_url(url, api_key)
        return None

    async def _search(self, query, api_key=None):
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Use Jina Reader to fetch DuckDuckGo search results page
        ddg_url = f"{DDG_SEARCH_URL}{quote(query)}"
        url = f"{JINA_READER_URL}{ddg_url}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Web search HTTP {resp.status}: {body[:200]}")
                        return {"result": "error", "message": f"Search failed (HTTP {resp.status})"}
                    data = await resp.json()
                    page_data = data.get("data", {})
                    content = page_data.get("content", "")
                    if len(content) > MAX_SEARCH_CONTENT:
                        content = content[:MAX_SEARCH_CONTENT] + "..."
                    logger.info(f"Web search: '{query}' ({len(content)} chars)")
                    return {"result": "ok", "query": query, "content": content}
        except TimeoutError:
            logger.error(f"Web search timed out for: {query}")
            return {"result": "error", "message": "Search timed out"}
        except aiohttp.ContentTypeError:
            logger.error(f"Web search: response was not JSON for: {query}")
            return {"result": "error", "message": "Invalid response format"}
        except Exception as e:
            logger.error(f"Web search error: {type(e).__name__}: {e}")
            return {"result": "error", "message": str(e) or type(e).__name__}

    async def _read_url(self, target_url, api_key=None):
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{JINA_READER_URL}{target_url}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Read URL HTTP {resp.status}: {body[:200]}")
                        return {"result": "error", "message": f"Failed to read URL (HTTP {resp.status})"}
                    data = await resp.json()
                    page_data = data.get("data", {})
                    content = page_data.get("content", "")
                    if len(content) > MAX_READ_CONTENT:
                        content = content[:MAX_READ_CONTENT] + "..."
                    logger.info(f"Read URL: {target_url} ({len(content)} chars)")
                    return {
                        "result": "ok",
                        "title": page_data.get("title", ""),
                        "url": target_url,
                        "content": content,
                    }
        except TimeoutError:
            logger.error(f"Read URL timed out: {target_url}")
            return {"result": "error", "message": "Page load timed out"}
        except aiohttp.ContentTypeError:
            logger.error(f"Read URL: response was not JSON for: {target_url}")
            return {"result": "error", "message": "Invalid response format"}
        except Exception as e:
            logger.error(f"Read URL error: {type(e).__name__}: {e}")
            return {"result": "error", "message": str(e) or type(e).__name__}
