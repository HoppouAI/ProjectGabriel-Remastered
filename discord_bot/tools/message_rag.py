import logging

from google.genai import types

logger = logging.getLogger(__name__)


class DiscordMessageRagTool:
    def __init__(self, handler):
        self.handler = handler

    def declarations(self):
        return [
            types.FunctionDeclaration(
                name="searchDiscordHistory",
                description=(
                    "Semantic search over older Discord messages and conversation chunks stored in the Discord message RAG index. "
                    "Use this to recall facts, past conversations, links, plans, preferences, or things people said when the recent chat context is not enough. "
                    "Results include message content, approximate date, author names, channel IDs, message IDs when available, and similarity scores.\n"
                    "**Invocation Condition:** Call when a user asks about something from earlier Discord history, asks what someone said before, asks to find an old message/link/topic, or when you need older context before answering."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "Natural language search query describing what to find in Discord history.",
                        },
                        "limit": {
                            "type": "INTEGER",
                            "description": "Maximum results to return. Defaults to 8. Use 3-5 for quick recall, up to 15 for deeper searches.",
                        },
                        "channel_id": {
                            "type": "STRING",
                            "description": "Optional channel ID to search. If omitted, searches the current channel unless include_all_channels is true or the query clearly asks about a person/older cross-channel memory.",
                        },
                        "include_all_channels": {
                            "type": "BOOLEAN",
                            "description": "Set true when the user asks to search broadly across Discord history, asks about another DM/channel, or asks whether a named person/topic ever came up.",
                        },
                        "author": {
                            "type": "STRING",
                            "description": "Optional author user ID or display name fragment to filter results after semantic search.",
                        },
                        "min_score": {
                            "type": "NUMBER",
                            "description": "Optional minimum similarity score. Leave unset for the configured provider threshold.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="getDiscordMessagesAround",
                description=(
                    "Fetches the nearby live Discord messages around a known message ID in a channel. "
                    "Use this after semantic search when you need exact surrounding conversation flow, reply context, or message IDs before taking an action.\n"
                    "**Invocation Condition:** Call after searchDiscordHistory returns a useful real message_id, or when the user provides a channel/message ID and asks what was said around it."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "message_id": {
                            "type": "STRING",
                            "description": "Discord message ID to center the context window on.",
                        },
                        "channel_id": {
                            "type": "STRING",
                            "description": "Discord channel ID. If omitted, uses the current channel.",
                        },
                        "before": {
                            "type": "INTEGER",
                            "description": "How many messages before the target to fetch. Defaults to 5, max 15.",
                        },
                        "after": {
                            "type": "INTEGER",
                            "description": "How many messages after the target to fetch. Defaults to 5, max 15.",
                        },
                    },
                    "required": ["message_id"],
                },
            ),
            types.FunctionDeclaration(
                name="backfillDiscordHistoryRag",
                description=(
                    "Indexes saved Discord conversation JSON logs into the Discord message RAG database. "
                    "This is useful after enabling RAG, changing providers, or importing older conversation logs. "
                    "It skips documents that are already indexed.\n"
                    "**Invocation Condition:** Call only when the user asks to sync, index, rebuild, backfill, or refresh the Discord history RAG database."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {},
                },
            ),
            types.FunctionDeclaration(
                name="getDiscordRagStats",
                description=(
                    "Shows Discord message RAG status, selected provider, document count, configured auto-injection state, and score threshold.\n"
                    "**Invocation Condition:** Call when the user asks whether Discord RAG is enabled, how much history is indexed, which RAG provider is active, or whether message recall is working."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {},
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "searchDiscordHistory":
            return await self._search(args)
        if name == "getDiscordMessagesAround":
            return await self._messages_around(args)
        if name == "backfillDiscordHistoryRag":
            return await self._backfill()
        if name == "getDiscordRagStats":
            return await self._stats()
        return None

    @property
    def _rag(self):
        return getattr(self.handler, "_message_rag", None)

    async def _search(self, args):
        rag = self._rag
        if not rag or not rag.ready:
            return {"success": False, "message": "Discord message RAG is not enabled or not ready"}
        query = str(args.get("query", "")).strip()
        if not query:
            return {"success": False, "message": "query is required"}
        limit = self._clamp_int(args.get("limit"), default=8, minimum=1, maximum=15)
        include_all = bool(args.get("include_all_channels", False)) or rag.should_search_all_channels(query)
        current_channel = getattr(self.handler, "_current_channel", None)
        channel_id = args.get("channel_id")
        if not channel_id and current_channel is not None and not include_all:
            channel_id = str(current_channel.id)
        result = rag.search(
            query=query,
            limit=limit,
            channel_id=str(channel_id) if channel_id else None,
            author=args.get("author"),
            min_score=args.get("min_score"),
        )
        if result.get("success"):
            result["search_scope"] = "all_channels" if not channel_id else f"channel:{channel_id}"
            result["usage_hint"] = "Use message_ids from results when you need to reference or delete a specific message."
        return result

    async def _messages_around(self, args):
        client = getattr(self.handler, "_discord_client", None)
        if not client:
            return {"success": False, "message": "Discord client is not available"}
        message_id = str(args.get("message_id", "")).strip()
        if not message_id:
            return {"success": False, "message": "message_id is required"}
        current_channel = getattr(self.handler, "_current_channel", None)
        channel_id = str(args.get("channel_id") or (current_channel.id if current_channel else "")).strip()
        if not channel_id:
            return {"success": False, "message": "channel_id is required outside the current channel"}
        before_limit = self._clamp_int(args.get("before"), default=5, minimum=0, maximum=15)
        after_limit = self._clamp_int(args.get("after"), default=5, minimum=0, maximum=15)
        try:
            channel = client.get_channel(int(channel_id)) or await client.fetch_channel(int(channel_id))
            target = await channel.fetch_message(int(message_id))
            before_msgs = []
            async for msg in channel.history(limit=before_limit, before=target):
                before_msgs.append(msg)
            after_msgs = []
            async for msg in channel.history(limit=after_limit, after=target):
                after_msgs.append(msg)
            ordered = list(reversed(before_msgs)) + [target] + after_msgs
            return {
                "success": True,
                "channel_id": str(channel.id),
                "target_message_id": str(target.id),
                "messages": [self._format_message(msg, target.id) for msg in ordered],
            }
        except Exception as e:
            logger.warning(f"getDiscordMessagesAround failed: {e}")
            return {"success": False, "message": str(e)}

    async def _backfill(self):
        rag = self._rag
        if not rag or not rag.ready:
            return {"success": False, "message": "Discord message RAG is not enabled or not ready"}
        store = getattr(self.handler, "_conversation_store", None)
        if store is None:
            return {"success": False, "message": "Conversation store is not available"}
        return await rag.backfill_from_conversations(store)

    async def _stats(self):
        rag = self._rag
        if not rag:
            return {"success": True, "enabled": False, "ready": False, "message": "Discord message RAG service is not attached"}
        return rag.stats()

    @staticmethod
    def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(maximum, number))

    @staticmethod
    def _format_message(message, target_id):
        content = message.clean_content or message.content or ""
        if len(content) > 1000:
            content = content[:997] + "..."
        return {
            "message_id": str(message.id),
            "is_target": message.id == target_id,
            "author_id": str(message.author.id),
            "author_name": getattr(message.author, "display_name", None) or message.author.name,
            "created_at": message.created_at.isoformat() if message.created_at else None,
            "content": content,
            "attachments": [att.filename for att in message.attachments],
        }
