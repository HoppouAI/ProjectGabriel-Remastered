import json
import logging
import re
import time
from pathlib import Path

import aiohttp
from google.genai import types

logger = logging.getLogger(__name__)

KLIPY_BASE_URL = "https://api.klipy.com"
STARRED_GIFS_FILE = Path(__file__).parent.parent / "data" / "starred_gifs.json"
CTRL_TOKEN_PATTERN = re.compile(r"<\s*ctrl\s*\d+\s*>?", re.IGNORECASE)


class DiscordGifTool:
    """KLIPY GIF tools for the Discord bot."""

    def __init__(self, handler):
        self.handler = handler

    def declarations(self):
        return [
            types.FunctionDeclaration(
                name="searchDiscordGifs",
                description=(
                    "Search KLIPY for Discord-ready GIFs and return numbered choices with titles, tags, previews, and send URLs. "
                    "Review the choices and then call sendDiscordGif with the best choice number, slug, or starred name. "
                    "Do not send the first result blindly when the vibe matters."
                    "\n**Invocation Condition:** Call when you want to find a reaction GIF, meme GIF, mood GIF, or any GIF to send on Discord."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Search phrase, mood, reaction, meme, or keyword"},
                        "limit": {"type": "INTEGER", "description": "How many choices to return, 1 to 20, default 8"},
                        "page": {"type": "INTEGER", "description": "Result page to fetch, default 1"},
                        "content_filter": {"type": "STRING", "description": "Safety filter: off, low, medium, or high. Default comes from config"},
                        "locale": {"type": "STRING", "description": "Locale or country hint, such as us, uk, or en_US. Default comes from config"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="getTrendingDiscordGifs",
                description=(
                    "Fetch trending KLIPY GIFs for Discord and return numbered choices with previews and send URLs. "
                    "Use this when a popular or broadly funny GIF fits better than a specific keyword search. "
                    "After comparing choices, call sendDiscordGif with a choice number, slug, or starred name.\n"
                    "**Invocation Condition:** Call when you want a trending GIF, viral GIF, or a general reaction GIF without a specific search term."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "limit": {"type": "INTEGER", "description": "How many choices to return, 1 to 20, default 8"},
                        "page": {"type": "INTEGER", "description": "Result page to fetch, default 1"},
                        "locale": {"type": "STRING", "description": "Locale or country hint, such as us, uk, or en_US. Default comes from config"},
                    },
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="sendDiscordGif",
                description=(
                    "Send a GIF to Discord using a KLIPY choice number from the latest search, a slug, an ID, a direct URL, or a starred GIF name. "
                    "You can add a short caption, but keep it natural and avoid explaining the tool call. "
                    "If choosing from search results, pick the best match by title, tags, and preview URL first.\n"
                    "**Invocation Condition:** Call after selecting the GIF you actually want to post on Discord."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "gif_ref": {"type": "STRING", "description": "Choice number, KLIPY slug, KLIPY ID, direct GIF URL, or starred GIF name"},
                        "caption": {"type": "STRING", "description": "Optional short message to send above the GIF"},
                        "target": {"type": "STRING", "description": "Optional channel ID, user ID, or username. Defaults to the current Discord channel"},
                    },
                    "required": ["gif_ref"],
                },
            ),
            types.FunctionDeclaration(
                name="starDiscordGif",
                description=(
                    "Save a GIF as a local starred favorite so it can be reused quickly later by name. "
                    "Use a memorable short name like smug, facepalm, panic, or victory.\n"
                    "**Invocation Condition:** Call when you find a reusable GIF you want to keep for fast future access."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "gif_ref": {"type": "STRING", "description": "Choice number, KLIPY slug, KLIPY ID, direct GIF URL, or existing starred GIF name"},
                        "name": {"type": "STRING", "description": "Short friendly name for the starred GIF"},
                        "note": {"type": "STRING", "description": "Optional note about when this GIF fits"},
                    },
                    "required": ["gif_ref"],
                },
            ),
            types.FunctionDeclaration(
                name="listStarredGifs",
                description=(
                    "List locally starred GIFs with their names, titles, notes, and send hints. "
                    "Use the returned starred name with sendDiscordGif to post one quickly.\n"
                    "**Invocation Condition:** Call when you want to reuse a saved GIF, review favorites, or pick from starred GIFs."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Optional filter for name, title, note, slug, or tags"},
                        "limit": {"type": "INTEGER", "description": "Maximum starred GIFs to return, default 20"},
                    },
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="unstarDiscordGif",
                description=(
                    "Remove a GIF from the local starred favorites list by starred name, slug, or ID.\n"
                    "**Invocation Condition:** Call when you no longer want a GIF saved as a favorite."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "gif_ref": {"type": "STRING", "description": "Starred name, KLIPY slug, or KLIPY ID to remove"},
                    },
                    "required": ["gif_ref"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "searchDiscordGifs":
            return await self._search(args)
        if name == "getTrendingDiscordGifs":
            return await self._trending(args)
        if name == "sendDiscordGif":
            return await self._send(args)
        if name == "starDiscordGif":
            return await self._star(args)
        if name == "listStarredGifs":
            return await self._list_starred(args)
        if name == "unstarDiscordGif":
            return await self._unstar(args)
        return None

    async def _search(self, args):
        ready = self._api_ready()
        if ready:
            return ready

        query = str(args.get("query", "")).strip()
        if not query:
            return {"result": "error", "message": "query required"}

        limit = self._clamp_int(args.get("limit"), 8, 1, 20)
        page = self._clamp_int(args.get("page"), 1, 1, 100)
        params = {
            "page": page,
            "per_page": max(limit, 8),
            "q": query,
            "customer_id": self._customer_id(),
            "locale": self._locale(args),
            "content_filter": args.get("content_filter") or self.handler.config.klipy_content_filter,
        }
        if self.handler.config.klipy_format_filter:
            params["format_filter"] = self.handler.config.klipy_format_filter

        data = await self._api_get("gifs/search", params)
        choices = self._normalize_response(data, limit, source="search", source_query=query)
        self._store_last_choices(choices)

        return {
            "result": "ok",
            "source": "search",
            "query": query,
            "count": len(choices),
            "page": self._page_info(data).get("current_page", page),
            "has_next": self._page_info(data).get("has_next", False),
            "choices": choices,
            "selection_guidance": "Choose by title, tags, preview_url, and the conversation vibe, then call sendDiscordGif with choice 1, choice 2, a slug, or a starred name.",
        }

    async def _trending(self, args):
        ready = self._api_ready()
        if ready:
            return ready

        limit = self._clamp_int(args.get("limit"), 8, 1, 20)
        page = self._clamp_int(args.get("page"), 1, 1, 100)
        params = {
            "page": page,
            "per_page": limit,
            "customer_id": self._customer_id(),
            "locale": self._locale(args),
        }
        if self.handler.config.klipy_format_filter:
            params["format_filter"] = self.handler.config.klipy_format_filter

        data = await self._api_get("gifs/trending", params)
        choices = self._normalize_response(data, limit, source="trending", source_query="")
        self._store_last_choices(choices)

        return {
            "result": "ok",
            "source": "trending",
            "count": len(choices),
            "page": self._page_info(data).get("current_page", page),
            "has_next": self._page_info(data).get("has_next", False),
            "choices": choices,
            "selection_guidance": "Pick the most fitting numbered choice, then call sendDiscordGif with that choice number.",
        }

    async def _send(self, args):
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        gif_ref = str(args.get("gif_ref", "")).strip()
        if not gif_ref:
            return {"result": "error", "message": "gif_ref required"}

        gif, source = await self._resolve_gif_ref(gif_ref)
        if not gif:
            return {"result": "error", "message": f"Could not resolve GIF: {gif_ref}"}

        url = gif.get("url")
        if not url:
            return {"result": "error", "message": "Resolved GIF has no sendable URL"}

        destination, destination_label = await self._resolve_destination(args.get("target"))
        if not destination:
            return {"result": "error", "message": destination_label}

        caption = self._sanitize_text(args.get("caption", ""))
        content = self._build_discord_message(caption, url)

        try:
            await destination.send(content)
            self._log_sent_message(destination, content)
            self.handler._tool_sent_message = True
            await self._track_share(gif)
            if source and source.get("star_key"):
                self._mark_star_used(source["star_key"])
            return {
                "result": "ok",
                "sent_to": destination_label,
                "gif": self._public_gif(gif),
                "used_ref": gif_ref,
            }
        except Exception as e:
            logger.error(f"Failed to send GIF: {e}")
            return {"result": "error", "message": str(e)}

    async def _star(self, args):
        gif_ref = str(args.get("gif_ref", "")).strip()
        if not gif_ref:
            return {"result": "error", "message": "gif_ref required"}

        gif, _source = await self._resolve_gif_ref(gif_ref)
        if not gif:
            return {"result": "error", "message": f"Could not resolve GIF: {gif_ref}"}

        requested_name = str(args.get("name", "")).strip()
        name = self._star_key(requested_name or gif.get("title") or gif.get("slug") or gif_ref)
        starred = self._load_starred()
        now = int(time.time())
        starred[name] = {
            "name": name,
            "title": gif.get("title") or name,
            "id": gif.get("id"),
            "slug": gif.get("slug"),
            "url": gif.get("url"),
            "preview_url": gif.get("preview_url"),
            "tags": gif.get("tags", []),
            "note": str(args.get("note", "")).strip(),
            "source_query": gif.get("source_query", ""),
            "created_at": now,
            "last_used_at": None,
            "use_count": 0,
        }
        self._save_starred(starred)
        return {
            "result": "ok",
            "starred": self._public_star(name, starred[name]),
            "send_hint": f"sendDiscordGif gif_ref={name}",
        }

    async def _list_starred(self, args):
        starred = self._load_starred()
        query = str(args.get("query", "")).strip().lower()
        limit = self._clamp_int(args.get("limit"), 20, 1, 50)
        items = []
        for name, gif in starred.items():
            haystack = " ".join([
                name,
                str(gif.get("title", "")),
                str(gif.get("slug", "")),
                str(gif.get("note", "")),
                " ".join(gif.get("tags", []) or []),
            ]).lower()
            if query and query not in haystack:
                continue
            items.append((name, gif))

        items.sort(key=lambda item: (item[1].get("last_used_at") or item[1].get("created_at") or 0), reverse=True)
        results = [self._public_star(name, gif) for name, gif in items[:limit]]
        return {"result": "ok", "count": len(results), "starred_gifs": results}

    async def _unstar(self, args):
        gif_ref = str(args.get("gif_ref", "")).strip()
        if not gif_ref:
            return {"result": "error", "message": "gif_ref required"}

        starred = self._load_starred()
        match = self._find_star_key(starred, gif_ref)
        if not match:
            return {"result": "error", "message": f"No starred GIF found for {gif_ref}"}

        removed = starred.pop(match)
        self._save_starred(starred)
        return {"result": "ok", "removed": self._public_star(match, removed)}

    async def _api_get(self, endpoint, params):
        timeout = aiohttp.ClientTimeout(total=12)
        url = f"{KLIPY_BASE_URL}/api/v1/{self.handler.config.klipy_app_key}/{endpoint}"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"KLIPY API returned {resp.status}: {text[:200]}")
                return json.loads(text) if text else {}

    async def _api_post(self, endpoint, payload):
        timeout = aiohttp.ClientTimeout(total=8)
        url = f"{KLIPY_BASE_URL}/api/v1/{self.handler.config.klipy_app_key}/{endpoint}"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"KLIPY API returned {resp.status}: {text[:200]}")
                return json.loads(text) if text else {}

    def _api_ready(self):
        if not self.handler.config.klipy_enabled:
            return {"result": "error", "message": "KLIPY GIF tools are disabled in discord_bot/config.yml"}
        if not self.handler.config.klipy_app_key:
            return {"result": "error", "message": "KLIPY app_key missing. Add klipy.app_key to discord_bot/config.yml"}
        return None

    def _locale(self, args):
        return args.get("locale") or self.handler.config.klipy_locale

    def _customer_id(self):
        configured = self.handler.config.klipy_customer_id
        if configured:
            return str(configured)
        client = self.handler._discord_client
        if client and client.user:
            return f"discord_{client.user.id}"
        return "projectgabriel_discord_bot"

    def _normalize_response(self, data, limit, source, source_query):
        payload = data.get("data", {}) if isinstance(data, dict) else {}
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        choices = []
        for item in rows:
            if item.get("type") == "ad":
                continue
            normalized = self._normalize_item(item, source=source, source_query=source_query)
            if not normalized:
                continue
            normalized["choice"] = len(choices) + 1
            choices.append(normalized)
            if len(choices) >= limit:
                break
        return choices

    def _normalize_item(self, item, source="", source_query=""):
        media = self._pick_media(item.get("file") or {})
        if not media:
            return None
        preview = self._pick_media(item.get("file") or {}, preview=True) or media
        tags = item.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return {
            "id": str(item.get("id", "")) if item.get("id") is not None else "",
            "slug": item.get("slug", ""),
            "title": item.get("title") or item.get("slug") or "Untitled GIF",
            "tags": tags[:8],
            "url": media["url"],
            "format": media.get("format"),
            "size": media.get("size"),
            "width": media.get("width"),
            "height": media.get("height"),
            "preview_url": preview["url"],
            "source": source,
            "source_query": source_query,
            "send_hint": f"sendDiscordGif gif_ref={item.get('slug') or item.get('id')}",
        }

    def _pick_media(self, files, preview=False):
        size_order = [self.handler.config.klipy_preferred_size, "md", "sm", "hd", "xs"]
        if preview:
            size_order = ["sm", "xs", self.handler.config.klipy_preferred_size, "md", "hd"]
        format_order = [self.handler.config.klipy_preferred_format, "gif", "webp", "mp4", "webm", "jpg"]
        if preview:
            format_order = ["gif", "webp", "jpg", "mp4", "webm"]

        for size in self._unique(size_order):
            size_block = files.get(size) or {}
            for fmt in self._unique(format_order):
                entry = size_block.get(fmt)
                if entry and entry.get("url"):
                    return {
                        "url": entry["url"],
                        "format": fmt,
                        "size": size,
                        "width": entry.get("width"),
                        "height": entry.get("height"),
                        "bytes": entry.get("size"),
                    }
        return None

    async def _resolve_gif_ref(self, gif_ref):
        if re.match(r"^https?://", gif_ref, re.IGNORECASE):
            return {"title": "Direct GIF", "url": gif_ref, "preview_url": gif_ref, "tags": []}, None

        last_choice = self._resolve_last_choice(gif_ref)
        if last_choice:
            return dict(last_choice), None

        starred = self._load_starred()
        star_key = self._find_star_key(starred, gif_ref)
        if star_key:
            gif = dict(starred[star_key])
            gif["starred_name"] = star_key
            return gif, {"star_key": star_key}

        ready = self._api_ready()
        if ready:
            return None, None

        fetched = await self._fetch_item(gif_ref)
        return fetched, None

    def _resolve_last_choice(self, gif_ref):
        match = re.search(r"\d+", gif_ref)
        if not match:
            return None
        index = int(match.group(0)) - 1
        choices = self._get_last_choices()
        if 0 <= index < len(choices):
            return choices[index]
        return None

    async def _fetch_item(self, gif_ref):
        key = gif_ref.strip()
        params = {"ids": key} if key.isdigit() else {"slugs": key}
        data = await self._api_get("gifs/items", params)
        choices = self._normalize_response(data, 1, source="items", source_query="")
        return choices[0] if choices else None

    async def _resolve_destination(self, target):
        client = self.handler._discord_client
        target = str(target or "").strip()

        if not target:
            channel = getattr(self.handler, "_current_channel", None)
            if channel:
                return channel, f"channel:{channel.id}"
            return None, "No active channel context and no target provided"

        try:
            raw_id = int(target)
            channel = client.get_channel(raw_id)
            if channel:
                return channel, f"channel:{raw_id}"
            user = await client.fetch_user(raw_id)
            dm = await user.create_dm()
            return dm, f"user:{user.id}"
        except (ValueError, Exception):
            pass

        lowered = target.lower()
        for guild in client.guilds:
            for member in guild.members:
                if member.name.lower() == lowered or str(member).lower() == lowered:
                    dm = await member.create_dm()
                    return dm, f"user:{member.id}"
        return None, f"Could not find Discord channel or user: {target}"

    def _build_discord_message(self, caption, url):
        caption = caption.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
        lines = []
        if caption:
            lines.append(caption)
        lines.append(url)
        if self.handler.config.klipy_attribution:
            lines.append("-# Powered by KLIPY")

        content = "\n".join(lines)
        max_len = self.handler.config.max_message_length
        if len(content) <= max_len:
            return content

        room = max_len - len(url) - 25
        trimmed_caption = caption[:max(room, 0)].rstrip()
        lines = [trimmed_caption, url] if trimmed_caption else [url]
        if self.handler.config.klipy_attribution and len("\n".join(lines)) + 18 <= max_len:
            lines.append("-# Powered by KLIPY")
        return "\n".join(lines)

    async def _track_share(self, gif):
        if not self.handler.config.klipy_enabled or not self.handler.config.klipy_app_key:
            return
        slug = gif.get("slug") or gif.get("id")
        if not slug:
            return
        payload = {
            "customer_id": self._customer_id(),
            "q": gif.get("source_query", "") or "",
        }
        try:
            await self._api_post(f"gifs/share/{slug}", payload)
        except Exception as e:
            logger.debug(f"KLIPY share tracking failed: {e}")

    def _store_last_choices(self, choices):
        if not hasattr(self.handler, "_gif_search_results"):
            self.handler._gif_search_results = {}
        key = self._channel_key()
        self.handler._gif_search_results[key] = choices
        self.handler._gif_search_results["global"] = choices

    def _get_last_choices(self):
        results = getattr(self.handler, "_gif_search_results", {})
        return results.get(self._channel_key()) or results.get("global") or []

    def _channel_key(self):
        channel = getattr(self.handler, "_current_channel", None)
        return str(channel.id) if channel else "global"

    def _load_starred(self):
        if not STARRED_GIFS_FILE.exists():
            return {}
        try:
            data = json.loads(STARRED_GIFS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_starred(self, starred):
        STARRED_GIFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STARRED_GIFS_FILE.write_text(json.dumps(starred, indent=2), encoding="utf-8")

    def _find_star_key(self, starred, gif_ref):
        key = gif_ref.strip().lower()
        key_slug = self._star_key(key)
        for name, gif in starred.items():
            candidates = {
                name.lower(),
                self._star_key(name),
                str(gif.get("slug", "")).lower(),
                str(gif.get("id", "")).lower(),
                self._star_key(str(gif.get("title", ""))),
            }
            if key in candidates or key_slug in candidates:
                return name
        return None

    def _mark_star_used(self, star_key):
        starred = self._load_starred()
        if star_key not in starred:
            return
        starred[star_key]["last_used_at"] = int(time.time())
        starred[star_key]["use_count"] = int(starred[star_key].get("use_count") or 0) + 1
        self._save_starred(starred)

    @staticmethod
    def _star_key(value):
        key = re.sub(r"[^a-z0-9_-]+", "_", str(value).strip().lower()).strip("_")
        return key[:60] or "gif"

    @staticmethod
    def _sanitize_text(text):
        cleaned = CTRL_TOKEN_PATTERN.sub(" ", str(text or ""))
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _clamp_int(value, default, minimum, maximum):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _unique(values):
        seen = []
        for value in values:
            if value and value not in seen:
                seen.append(value)
        return seen

    @staticmethod
    def _page_info(data):
        payload = data.get("data", {}) if isinstance(data, dict) else {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _public_gif(gif):
        return {
            "title": gif.get("title"),
            "id": gif.get("id"),
            "slug": gif.get("slug"),
            "url": gif.get("url"),
            "preview_url": gif.get("preview_url"),
            "tags": gif.get("tags", []),
        }

    def _public_star(self, name, gif):
        return {
            "name": name,
            "title": gif.get("title"),
            "id": gif.get("id"),
            "slug": gif.get("slug"),
            "preview_url": gif.get("preview_url") or gif.get("url"),
            "tags": gif.get("tags", []),
            "note": gif.get("note", ""),
            "use_count": gif.get("use_count", 0),
            "send_hint": f"sendDiscordGif gif_ref={name}",
        }

    def _log_sent_message(self, destination, content):
        conversations = getattr(self.handler, "_conversations", None)
        if conversations and hasattr(destination, "id"):
            conversations.add_message(str(destination.id), "assistant", content)
