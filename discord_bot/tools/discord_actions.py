import logging
from google.genai import types

logger = logging.getLogger(__name__)


class DiscordActionsTool:
    """Discord-specific actions the bot can perform."""

    def __init__(self, handler):
        self.handler = handler

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
                return {"result": "ok", "sent_to": str(user)}

            return {"result": "error", "message": f"Could not find user or channel: {target}"}
        except Exception as e:
            logger.error(f"Send message failed: {e}")
            return {"result": "error", "message": str(e)}

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
