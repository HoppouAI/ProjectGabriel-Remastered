import json
import logging
import time as _time
from pathlib import Path
from google.genai import types

logger = logging.getLogger(__name__)

MUTES_FILE = Path(__file__).parent.parent / "data" / "mutes.json"
USER_CACHE_FILE = Path(__file__).parent.parent / "data" / "user_cache.json"


class DiscordActionsTool:
    """Discord-specific actions the bot can perform."""

    def __init__(self, handler):
        self.handler = handler
        self._user_cache = self._load_user_cache()

    @staticmethod
    def _load_user_cache():
        """Load cached user ID -> name mappings from disk."""
        if USER_CACHE_FILE.exists():
            try:
                return json.loads(USER_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_user_cache(self):
        USER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USER_CACHE_FILE.write_text(json.dumps(self._user_cache, indent=2), encoding="utf-8")

    def _cache_user(self, user):
        """Cache a resolved user's ID and names."""
        uid = str(user.id)
        self._user_cache[uid] = {
            "name": user.name,
            "display_name": getattr(user, "display_name", None) or user.name,
        }
        self._save_user_cache()

    def declarations(self):
        return [
            types.FunctionDeclaration(
                name="sendDiscordMessage",
                description="Send a message to a Discord user or channel.\n**Invocation Condition:** Call when you need to proactively message someone on Discord.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "target": {"type": "STRING", "description": "Username, user ID, or channel ID to message"},
                        "message": {"type": "STRING", "description": "The message to send"},
                    },
                    "required": ["target", "message"],
                },
            ),
            types.FunctionDeclaration(
                name="addReaction",
                description="Add a reaction emoji to a message.\n**Invocation Condition:** Call when you want to react to a message with an emoji.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The channel ID where the message is"},
                        "message_id": {"type": "STRING", "description": "The message ID to react to"},
                        "emoji": {"type": "STRING", "description": "The emoji to react with (Unicode or custom format)"},
                    },
                    "required": ["channel_id", "message_id", "emoji"],
                },
            ),
            types.FunctionDeclaration(
                name="setDiscordStatus",
                description="Set your Discord status/activity.\n**Invocation Condition:** Call when asked to change your Discord status.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "status": {"type": "STRING", "description": "online, idle, dnd, or invisible"},
                        "activity_text": {"type": "STRING", "description": "Custom status text"},
                    },
                    "required": ["status"],
                },
            ),
            types.FunctionDeclaration(
                name="getChannelMembers",
                description="Get the list of members in the current Discord channel, group DM, or server.\n**Invocation Condition:** Call when someone asks who is in the chat, group, channel, or server.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="muteChannel",
                description=(
                    "Temporarily mute a channel/DM so you stop responding there for a period of time. "
                    "After the duration expires, you will automatically re-engage with a comeback message.\n"
                    "**Invocation Condition:** Call when you want to mute, ignore, or take a break from a "
                    "channel, group DM, or DM conversation. Use when people are being annoying and you want "
                    "to stop responding for a while."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "duration_minutes": {"type": "NUMBER", "description": "How many minutes to mute for (1-60, default 5)"},
                        "comeback_message": {"type": "STRING", "description": "Optional message to send when you unmute and come back"},
                    },
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="createGroupChat",
                description=(
                    "Create a new group DM with specified users. Requires at least 2 users. "
                    "All users must be friends with this account. Use their user IDs (from message metadata) whenever possible.\n"
                    "**Invocation Condition:** Call when asked to create or start a new group chat with people."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_ids": {"type": "STRING", "description": "Comma-separated user IDs or usernames of ALL users to add (minimum 2)"},
                    },
                    "required": ["user_ids"],
                },
            ),
            types.FunctionDeclaration(
                name="addToGroup",
                description=(
                    "Add a user to a group DM.\n"
                    "**Invocation Condition:** Call when asked to add someone to a group chat."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_id": {"type": "STRING", "description": "The user ID or username to add"},
                        "group_id": {"type": "STRING", "description": "The group channel ID (optional, defaults to current channel)"},
                    },
                    "required": ["user_id"],
                },
            ),
            types.FunctionDeclaration(
                name="removeFromGroup",
                description=(
                    "Remove a user from a group DM. You must be the group owner to remove people.\n"
                    "**Invocation Condition:** Call when asked to remove or kick someone from a group chat."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_id": {"type": "STRING", "description": "The user ID or username to remove"},
                        "group_id": {"type": "STRING", "description": "The group channel ID (optional, defaults to current channel)"},
                    },
                    "required": ["user_id"],
                },
            ),
            types.FunctionDeclaration(
                name="viewProfile",
                description=(
                    "View a Discord user's profile including bio, badges, account creation date, "
                    "mutual servers, and connected accounts.\n"
                    "**Invocation Condition:** Call when asked about someone's Discord profile, bio, "
                    "badges, join date, or when you want to learn about a user."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_id": {"type": "STRING", "description": "The user ID or username to look up"},
                    },
                    "required": ["user_id"],
                },
            ),
            types.FunctionDeclaration(
                name="listGroupChats",
                description=(
                    "List your active group DMs with their IDs and members.\n"
                    "**Invocation Condition:** Call when you need to find a group chat ID, "
                    "or when asked about your group DMs."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="getFriendsList",
                description=(
                    "Get your Discord friends list with usernames and user IDs.\n"
                    "**Invocation Condition:** Call when asked about your Discord friends, "
                    "who your friends are, or when you need to look up a friend's user ID."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="transferGroupOwnership",
                description=(
                    "Transfer ownership of a group DM to another member. You must be the current owner.\n"
                    "**Invocation Condition:** Call when asked to transfer or give group ownership to someone."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_id": {"type": "STRING", "description": "The user ID or username to transfer ownership to"},
                        "group_id": {"type": "STRING", "description": "The group channel ID (optional, defaults to current channel)"},
                    },
                    "required": ["user_id"],
                },
            ),
            types.FunctionDeclaration(
                name="deleteMessage",
                description=(
                    "Delete one of YOUR OWN messages by its message ID. You can only delete messages you sent.\n"
                    "**Invocation Condition:** Call when asked to delete, unsend, or remove one of your messages."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The channel ID where the message is"},
                        "message_id": {"type": "STRING", "description": "The message ID to delete"},
                    },
                    "required": ["channel_id", "message_id"],
                },
            ),
            types.FunctionDeclaration(
                name="blockUser",
                description=(
                    "Block a Discord user. They will no longer be able to message you or see your online status.\n"
                    "**Invocation Condition:** Call when asked to block someone on Discord."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_id": {"type": "STRING", "description": "The user ID or username to block"},
                    },
                    "required": ["user_id"],
                },
            ),
            types.FunctionDeclaration(
                name="unblockUser",
                description=(
                    "Unblock a previously blocked Discord user.\n"
                    "**Invocation Condition:** Call when asked to unblock someone on Discord."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "user_id": {"type": "STRING", "description": "The user ID or username to unblock"},
                    },
                    "required": ["user_id"],
                },
            ),
            types.FunctionDeclaration(
                name="pinMessage",
                description=(
                    "Pin a message in a channel.\n"
                    "**Invocation Condition:** Call when asked to pin a message."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The channel ID where the message is"},
                        "message_id": {"type": "STRING", "description": "The message ID to pin"},
                    },
                    "required": ["channel_id", "message_id"],
                },
            ),
            types.FunctionDeclaration(
                name="unpinMessage",
                description=(
                    "Unpin a previously pinned message in a channel.\n"
                    "**Invocation Condition:** Call when asked to unpin a message."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "channel_id": {"type": "STRING", "description": "The channel ID where the message is"},
                        "message_id": {"type": "STRING", "description": "The message ID to unpin"},
                    },
                    "required": ["channel_id", "message_id"],
                },
            ),
            types.FunctionDeclaration(
                name="renameGroupChat",
                description=(
                    "Rename a group DM. You must be the group owner.\n"
                    "**Invocation Condition:** Call when asked to rename or change the name of a group chat."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING", "description": "The new name for the group chat"},
                        "group_id": {"type": "STRING", "description": "The group channel ID (optional, defaults to current channel)"},
                    },
                    "required": ["name"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "sendDiscordMessage":
            return await self._send_message(args)
        elif name == "addReaction":
            return await self._add_reaction(args)
        elif name == "setDiscordStatus":
            return await self._set_status(args)
        elif name == "getChannelMembers":
            return await self._get_channel_members(args)
        elif name == "muteChannel":
            return await self._mute_channel(args)
        elif name == "createGroupChat":
            return await self._create_group_chat(args)
        elif name == "addToGroup":
            return await self._add_to_group(args)
        elif name == "removeFromGroup":
            return await self._remove_from_group(args)
        elif name == "viewProfile":
            return await self._view_profile(args)
        elif name == "listGroupChats":
            return await self._list_group_chats(args)
        elif name == "getFriendsList":
            return await self._get_friends_list(args)
        elif name == "transferGroupOwnership":
            return await self._transfer_ownership(args)
        elif name == "deleteMessage":
            return await self._delete_message(args)
        elif name == "blockUser":
            return await self._block_user(args)
        elif name == "unblockUser":
            return await self._unblock_user(args)
        elif name == "pinMessage":
            return await self._pin_message(args)
        elif name == "unpinMessage":
            return await self._unpin_message(args)
        elif name == "renameGroupChat":
            return await self._rename_group_chat(args)
        return None

    async def _send_message(self, args):
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        target = args.get("target", "")
        message = args.get("message", "")
        if not target or not message:
            return {"result": "error", "message": "target and message required"}

        # Strip mass pings
        message = message.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")

        try:
            # Try as user ID first
            user = None
            try:
                uid = int(target)
                user = await client.fetch_user(uid)
            except (ValueError, Exception):
                pass

            # Try as channel ID
            if not user:
                try:
                    cid = int(target)
                    channel = client.get_channel(cid)
                    if channel:
                        await channel.send(message)
                        self._log_sent_message(str(cid), message)
                        self.handler._tool_sent_message = True
                        return {"result": "ok", "sent_to": f"channel:{cid}"}
                except (ValueError, Exception):
                    pass

            # Try as username
            if not user:
                for guild in client.guilds:
                    for member in guild.members:
                        if member.name == target or str(member) == target:
                            user = member
                            break
                    if user:
                        break

            if user:
                dm = await user.create_dm()
                await dm.send(message)
                self._log_sent_message(str(dm.id), message)
                self.handler._tool_sent_message = True
                return {"result": "ok", "sent_to": str(user)}

            return {"result": "error", "message": f"Could not find user or channel: {target}"}
        except Exception as e:
            logger.error(f"Send message failed: {e}")
            return {"result": "error", "message": str(e)}

    def _log_sent_message(self, channel_id, message):
        conversations = getattr(self.handler, "_conversations", None)
        if conversations:
            conversations.add_message(channel_id, "assistant", message)

    async def _add_reaction(self, args):
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        try:
            channel = client.get_channel(int(args["channel_id"]))
            if not channel:
                return {"result": "error", "message": "Channel not found"}
            message = await channel.fetch_message(int(args["message_id"]))
            await message.add_reaction(args["emoji"])
            return {"result": "ok"}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _set_status(self, args):
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        import discord
        status_map = {
            "online": discord.Status.online,
            "idle": discord.Status.idle,
            "dnd": discord.Status.dnd,
            "invisible": discord.Status.invisible,
        }
        status = status_map.get(args.get("status", "online"), discord.Status.online)
        activity_text = args.get("activity_text")
        activity = discord.CustomActivity(name=activity_text) if activity_text else None

        try:
            await client.change_presence(status=status, activity=activity)
            return {"result": "ok", "status": args.get("status")}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _get_channel_members(self, args):
        import discord
        channel = getattr(self.handler, "_current_channel", None)
        if not channel:
            return {"result": "error", "message": "No active channel context"}

        members = []
        if isinstance(channel, discord.DMChannel):
            if channel.recipient:
                members.append(channel.recipient.display_name or channel.recipient.name)
            return {"result": "ok", "type": "DM", "members": members, "count": len(members)}
        elif isinstance(channel, discord.GroupChannel):
            for user in channel.recipients:
                members.append(user.display_name or user.name)
            return {"result": "ok", "type": "Group DM", "name": channel.name or "unnamed", "members": members, "count": len(members)}
        elif hasattr(channel, "members"):
            for member in channel.members:
                members.append(member.display_name or member.name)
            return {"result": "ok", "type": "channel", "name": channel.name, "members": members, "count": len(members)}

        return {"result": "error", "message": "Cannot get members for this channel type"}

    async def _mute_channel(self, args):
        import asyncio
        channel = getattr(self.handler, "_current_channel", None)
        if not channel:
            return {"result": "error", "message": "No active channel context"}

        channel_id = str(channel.id)
        duration = min(max(int(args.get("duration_minutes", 5)), 1), 60)
        comeback = args.get("comeback_message", "")

        # Add to muted set on the bot
        muted = getattr(self.handler, "_muted_channels", None)
        if muted is None:
            self.handler._muted_channels = set()
            muted = self.handler._muted_channels
        muted.add(channel_id)

        # Persist to file
        expires_at = _time.time() + duration * 60
        self._save_mute(channel_id, expires_at, comeback)

        logger.info(f"Muted channel {channel_id} for {duration} minutes")

        # Schedule unmute
        async def _unmute():
            await asyncio.sleep(duration * 60)
            muted.discard(channel_id)
            self._remove_mute(channel_id)
            logger.info(f"Unmuted channel {channel_id}")
            client = self.handler._discord_client
            if client and comeback:
                try:
                    ch = client.get_channel(int(channel_id))
                    if ch:
                        await ch.send(comeback)
                except Exception as e:
                    logger.warning(f"Failed to send comeback message: {e}")

        asyncio.create_task(_unmute())
        return {"result": "ok", "muted": True, "channel_id": channel_id, "duration_minutes": duration}

    async def _resolve_user(self, identifier):
        """Resolve a user by ID, username, or partial match. Returns (user, error_msg)."""
        client = self.handler._discord_client
        identifier = identifier.strip()

        # Try as user ID
        try:
            uid = int(identifier)
            user = await client.fetch_user(uid)
            self._cache_user(user)
            return user, None
        except (ValueError, Exception):
            pass

        lower = identifier.lower()

        # Check persistent cache for name matches
        for uid, info in self._user_cache.items():
            cached_name = info.get("name", "").lower()
            cached_display = info.get("display_name", "").lower()
            if lower == cached_name or lower == cached_display or lower in cached_name or lower in cached_display:
                try:
                    user = await client.fetch_user(int(uid))
                    self._cache_user(user)
                    return user, None
                except Exception:
                    pass

        # Exact match first
        for guild in client.guilds:
            for member in guild.members:
                if member.name.lower() == lower or (member.display_name and member.display_name.lower() == lower):
                    self._cache_user(member)
                    return member, None

        # Partial/contains match
        for guild in client.guilds:
            for member in guild.members:
                if lower in member.name.lower() or (member.display_name and lower in member.display_name.lower()):
                    self._cache_user(member)
                    return member, None

        # Also check DM channels and friends
        for channel in client.private_channels:
            if hasattr(channel, "recipients"):
                for user in channel.recipients:
                    if user.name.lower() == lower or lower in user.name.lower():
                        self._cache_user(user)
                        return user, None
                    if user.display_name and (user.display_name.lower() == lower or lower in user.display_name.lower()):
                        self._cache_user(user)
                        return user, None
            elif hasattr(channel, "recipient") and channel.recipient:
                user = channel.recipient
                if user.name.lower() == lower or lower in user.name.lower():
                    self._cache_user(user)
                    return user, None
                if user.display_name and (user.display_name.lower() == lower or lower in user.display_name.lower()):
                    self._cache_user(user)
                    return user, None

        # Query guild member lists via API (handles uncached members)
        for guild in client.guilds:
            try:
                results = await guild.query_members(query=identifier, limit=5)
                if results:
                    self._cache_user(results[0])
                    return results[0], None
            except Exception:
                pass

        return None, f"Could not find user: {identifier}. Use getChannelMembers to find exact user IDs."

    async def _create_group_chat(self, args):
        import discord
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        raw_ids = args.get("user_ids", "")
        identifiers = [uid.strip() for uid in raw_ids.split(",") if uid.strip()]
        if len(identifiers) < 2:
            return {"result": "error", "message": "Need at least 2 users to create a group chat. Provide comma-separated user IDs or usernames."}

        users = []
        for ident in identifiers:
            user, err = await self._resolve_user(ident)
            if err:
                return {"result": "error", "message": err}
            users.append(user)

        try:
            group = await client.create_group(*users)
            return {"result": "ok", "group_id": str(group.id), "members": [u.name for u in users]}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _add_to_group(self, args):
        import discord
        channel = None
        group_id = args.get("group_id", "").strip()
        if group_id:
            channel = self.handler._discord_client.get_channel(int(group_id))
        if not channel:
            channel = getattr(self.handler, "_current_channel", None)
        if not channel or not isinstance(channel, discord.GroupChannel):
            return {"result": "error", "message": "Target is not a group DM. Provide a valid group_id."}

        ident = args.get("user_id", "").strip()
        if not ident:
            return {"result": "error", "message": "No user provided"}

        user, err = await self._resolve_user(ident)
        if err:
            return {"result": "error", "message": err}

        try:
            await channel.add_recipients(user)
            return {"result": "ok", "added": user.name, "group_id": str(channel.id)}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _remove_from_group(self, args):
        import discord
        channel = None
        group_id = args.get("group_id", "").strip()
        if group_id:
            channel = self.handler._discord_client.get_channel(int(group_id))
        if not channel:
            channel = getattr(self.handler, "_current_channel", None)
        if not channel or not isinstance(channel, discord.GroupChannel):
            return {"result": "error", "message": "Target is not a group DM. Provide a valid group_id."}

        ident = args.get("user_id", "").strip()
        if not ident:
            return {"result": "error", "message": "No user provided"}

        user, err = await self._resolve_user(ident)
        if err:
            return {"result": "error", "message": err}

        try:
            await channel.remove_recipients(user)
            return {"result": "ok", "removed": user.name, "group_id": str(channel.id)}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _view_profile(self, args):
        ident = args.get("user_id", "").strip()
        if not ident:
            return {"result": "error", "message": "No user provided"}

        user, err = await self._resolve_user(ident)
        if err:
            return {"result": "error", "message": err}

        try:
            profile = await user.profile(with_mutual_guilds=True, with_mutual_friends=True)
            info = {
                "result": "ok",
                "username": user.name,
                "display_name": user.display_name,
                "id": str(user.id),
                "created_at": user.created_at.strftime("%B %d, %Y"),
                "bot": user.bot,
            }
            if profile.bio:
                info["bio"] = profile.bio
            if profile.badges:
                info["badges"] = [str(b) for b in profile.badges]
            if profile.premium_since:
                info["nitro_since"] = profile.premium_since.strftime("%B %d, %Y")
            if profile.mutual_guilds:
                info["mutual_servers"] = [g.guild.name if hasattr(g, "guild") else str(g) for g in profile.mutual_guilds]
            if profile.mutual_friends:
                info["mutual_friends"] = [f.name for f in profile.mutual_friends]
            if profile.connections:
                info["connections"] = [{"type": c.type, "name": c.name} for c in profile.connections]
            return info
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _get_friends_list(self, args):
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        try:
            relationships = client.friends  # List[Relationship], not List[User]
            friend_list = []
            for rel in relationships:
                user = rel.user
                entry = {
                    "username": user.name,
                    "display_name": user.display_name or user.name,
                    "id": str(user.id),
                }
                if rel.nick:
                    entry["nickname"] = rel.nick
                friend_list.append(entry)
                self._cache_user(user)
            friend_list.sort(key=lambda f: f["username"].lower())
            return {"result": "ok", "count": len(friend_list), "friends": friend_list}
        except Exception as e:
            logger.error(f"getFriendsList failed: {e}", exc_info=True)
            return {"result": "error", "message": str(e)}

    async def _list_group_chats(self, args):
        import discord
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        groups = []
        for ch in client.private_channels:
            if isinstance(ch, discord.GroupChannel):
                members = [u.display_name or u.name for u in ch.recipients]
                groups.append({
                    "id": str(ch.id),
                    "name": ch.name or "unnamed",
                    "members": members,
                    "owner_id": str(ch.owner_id) if ch.owner_id else None,
                })

        return {"result": "ok", "groups": groups, "count": len(groups)}

    async def _transfer_ownership(self, args):
        import discord
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        channel = None
        group_id = args.get("group_id", "").strip()
        if group_id:
            try:
                channel = client.get_channel(int(group_id))
            except ValueError:
                return {"result": "error", "message": f"Invalid group ID: {group_id}"}
        if not channel:
            channel = getattr(self.handler, "_current_channel", None)
        if not channel or not isinstance(channel, discord.GroupChannel):
            return {"result": "error", "message": "Target is not a group DM. Provide a valid group_id."}

        ident = args.get("user_id", "").strip()
        if not ident:
            return {"result": "error", "message": "No user provided"}

        user, err = await self._resolve_user(ident)
        if err:
            return {"result": "error", "message": err}

        try:
            await channel.edit(owner=user)
            return {"result": "ok", "new_owner": user.name, "group_id": str(channel.id)}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _delete_message(self, args):
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        try:
            channel = client.get_channel(int(args["channel_id"]))
            if not channel:
                return {"result": "error", "message": "Channel not found"}
            message = await channel.fetch_message(int(args["message_id"]))
            if message.author.id != client.user.id:
                return {"result": "error", "message": "Can only delete your own messages"}
            await message.delete()
            return {"result": "ok", "deleted": args["message_id"]}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _block_user(self, args):
        ident = args.get("user_id", "").strip()
        if not ident:
            return {"result": "error", "message": "No user provided"}

        user, err = await self._resolve_user(ident)
        if err:
            return {"result": "error", "message": err}

        try:
            await user.block()
            return {"result": "ok", "blocked": user.name, "id": str(user.id)}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _unblock_user(self, args):
        ident = args.get("user_id", "").strip()
        if not ident:
            return {"result": "error", "message": "No user provided"}

        user, err = await self._resolve_user(ident)
        if err:
            return {"result": "error", "message": err}

        try:
            await user.unblock()
            return {"result": "ok", "unblocked": user.name, "id": str(user.id)}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _pin_message(self, args):
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        try:
            channel = client.get_channel(int(args["channel_id"]))
            if not channel:
                return {"result": "error", "message": "Channel not found"}
            message = await channel.fetch_message(int(args["message_id"]))
            await message.pin()
            return {"result": "ok", "pinned": args["message_id"]}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _unpin_message(self, args):
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        try:
            channel = client.get_channel(int(args["channel_id"]))
            if not channel:
                return {"result": "error", "message": "Channel not found"}
            message = await channel.fetch_message(int(args["message_id"]))
            await message.unpin()
            return {"result": "ok", "unpinned": args["message_id"]}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    async def _rename_group_chat(self, args):
        import discord
        client = self.handler._discord_client
        if not client:
            return {"result": "error", "message": "Discord client not connected"}

        channel = None
        group_id = args.get("group_id", "").strip()
        if group_id:
            try:
                channel = client.get_channel(int(group_id))
            except ValueError:
                return {"result": "error", "message": f"Invalid group ID: {group_id}"}
        if not channel:
            channel = getattr(self.handler, "_current_channel", None)
        if not channel or not isinstance(channel, discord.GroupChannel):
            return {"result": "error", "message": "Target is not a group DM. Provide a valid group_id."}

        new_name = args.get("name", "").strip()
        if not new_name:
            return {"result": "error", "message": "No name provided"}

        try:
            await channel.edit(name=new_name)
            return {"result": "ok", "renamed": new_name, "group_id": str(channel.id)}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    @staticmethod
    def _save_mute(channel_id, expires_at, comeback):
        MUTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        mutes = {}
        if MUTES_FILE.exists():
            try:
                mutes = json.loads(MUTES_FILE.read_text())
            except Exception:
                pass
        mutes[channel_id] = {"expires_at": expires_at, "comeback": comeback}
        MUTES_FILE.write_text(json.dumps(mutes))

    @staticmethod
    def _remove_mute(channel_id):
        if not MUTES_FILE.exists():
            return
        try:
            mutes = json.loads(MUTES_FILE.read_text())
            mutes.pop(channel_id, None)
            MUTES_FILE.write_text(json.dumps(mutes))
        except Exception:
            pass

    @staticmethod
    def load_persisted_mutes():
        """Load mutes from file, returning {channel_id: {expires_at, comeback}} for active mutes."""
        if not MUTES_FILE.exists():
            return {}
        try:
            mutes = json.loads(MUTES_FILE.read_text())
            now = _time.time()
            active = {k: v for k, v in mutes.items() if v["expires_at"] > now}
            # Clean up expired entries
            if len(active) != len(mutes):
                MUTES_FILE.write_text(json.dumps(active))
            return active
        except Exception:
            return {}
