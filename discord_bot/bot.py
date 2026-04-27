import asyncio
import io
import logging
import random
import re
import time
from datetime import datetime

import aiohttp
import discord

from discord_bot.config import BotConfig
from discord_bot.gemini_session import GeminiTextSession
from discord_bot.tools.handler import DiscordToolHandler
from discord_bot.conversation_store import ConversationStore
from src.personalities import PersonalityManager

logger = logging.getLogger(__name__)

_CTRL_TOKEN_PATTERN = re.compile(r"<\s*ctrl\s*\d+\s*>?", re.IGNORECASE)


class DiscordBot:
    """Discord selfbot powered by a Gemini Live text session.

    Listens for DMs, mentions, and messages in configured channels.
    Routes messages through Gemini Live for AI-generated responses.
    Supports image viewing, memory tools, and relay to the main VRChat session.
    """

    def __init__(self, config=None, relay_callback=None, instance_monitor=None):
        self.config = config or BotConfig()
        self._relay_callback = relay_callback
        self._instance_monitor = instance_monitor
        self._client = None
        self._gemini = None
        self._tool_handler = None
        self._personality = PersonalityManager(
            personalities_file="discord_bot/prompts/personalities.yml"
        )
        self._conversations = ConversationStore(self.config.conversations_dir)
        self._cooldowns = {}  # channel_id -> last_response_time
        self._batch_queues = {}  # channel_id -> list of (message, images) tuples
        self._batch_tasks = {}  # channel_id -> asyncio.Task (debounce timer)
        self._response_tasks = {}  # channel_id -> asyncio.Task (current response)
        self._response_lock = asyncio.Lock()  # Serialize Gemini interactions across channels
        self._followup_tasks = {}  # channel_id -> asyncio.Task (left-on-read timers)
        self._disabled = False  # Global disable toggle
        self._running = False
        self._start_time = datetime.now()

    async def start(self):
        """Start the Discord bot and Gemini session."""
        self._running = True

        # Set up tool handler
        self._tool_handler = DiscordToolHandler(
            self.config,
            relay_callback=self._relay_callback,
            personality_mgr=self._personality,
        )

        # Set up Gemini session (AUDIO modality with transcription)
        self._gemini = GeminiTextSession(self.config, self._tool_handler, self._personality)

        # Set up Discord client
        self._client = discord.Client()
        self._tool_handler.set_discord_client(self._client)
        self._tool_handler._conversations = self._conversations

        # Register event handlers
        self._register_events()

        # Start Gemini session in background
        gemini_task = asyncio.create_task(self._gemini.run_forever())

        # Start Discord client
        try:
            await self._client.start(self.config.discord_token)
        except Exception as e:
            logger.error(f"Discord client error: {e}")
        finally:
            self._running = False
            gemini_task.cancel()
            await self._gemini.disconnect()

    async def stop(self):
        """Stop the bot gracefully."""
        self._running = False
        if self._gemini:
            await self._gemini.disconnect()
        if self._client and not self._client.is_closed():
            await self._client.close()

    async def send_message_to_user(self, username_or_id, message):
        """Send a message to a Discord user. Called from main session tools."""
        if not self._client or not self._client.is_ready():
            return {"result": "error", "message": "Discord bot not connected"}

        user = None

        # Try as user ID
        try:
            uid = int(username_or_id)
            user = await self._client.fetch_user(uid)
        except (ValueError, Exception):
            pass

        # Try as username across guilds
        if not user:
            for guild in self._client.guilds:
                for member in guild.members:
                    if member.name == username_or_id or str(member) == username_or_id:
                        user = member
                        break
                if user:
                    break

        if not user:
            return {"result": "error", "message": f"User not found: {username_or_id}"}

        try:
            message = self._sanitize_generated_text(message)
            if not message:
                return {"result": "error", "message": "Message is empty after sanitization"}
            dm = await user.create_dm()
            await dm.send(message)
            self._conversations.add_message(str(dm.id), "assistant", message)
            return {"result": "ok", "sent_to": str(user)}
        except Exception as e:
            return {"result": "error", "message": str(e)}

    @staticmethod
    def _sanitize_generated_text(text):
        """Remove Gemini Live control artifacts from generated text."""
        if not text:
            return ""
        cleaned = _CTRL_TOKEN_PATTERN.sub(" ", text)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
        return cleaned.strip()

    async def receive_relay(self, text):
        """Receive a relay message from the VRChat session into the Gemini session."""
        if self._gemini and self._gemini._connected.is_set():
            await self._gemini.inject_context(text)

    def _restore_mutes(self):
        from discord_bot.tools.discord_actions import DiscordActionsTool
        active = DiscordActionsTool.load_persisted_mutes()
        if not active:
            return
        if not hasattr(self._tool_handler, "_muted_channels"):
            self._tool_handler._muted_channels = set()
        muted = self._tool_handler._muted_channels
        now = time.time()
        for channel_id, data in active.items():
            muted.add(channel_id)
            remaining = data["expires_at"] - now
            comeback = data.get("comeback", "")

            async def _unmute(cid=channel_id, msg=comeback, secs=remaining):
                await asyncio.sleep(max(secs, 0))
                muted.discard(cid)
                DiscordActionsTool._remove_mute(cid)
                logger.info(f"Unmuted restored channel {cid}")
                if self._client and msg:
                    try:
                        ch = self._client.get_channel(int(cid))
                        if ch:
                            await ch.send(msg)
                    except Exception as e:
                        logger.warning(f"Failed to send comeback message: {e}")

            asyncio.create_task(_unmute())
        logger.info(f"Restored {len(active)} persisted mute(s)")

    async def _send_embed(self, destination, title, description=None, fields=None, color=0x5865F2):
        """Send a rich embed to a channel via webhook forwarding.

        Posts an embed to a webhook channel, then forwards the message to the
        destination. Falls back to plain text if no webhook is configured.

        Args:
            destination: Channel to send the embed to
            title: Embed title
            description: Optional description text
            fields: Optional list of (name, value, inline) tuples
            color: Embed color (default: Discord blurple)

        Returns:
            The sent/forwarded message, or None on failure
        """
        webhook_url = self.config.embed_webhook_url
        if not webhook_url:
            # Fallback to plain text
            lines = [f"**{title}**"]
            if description:
                lines.append(description)
            if fields:
                for name, value, _ in fields:
                    lines.append(f"**{name}:** {value}")
            return await destination.send("\n".join(lines))

        # Build embed payload for webhook
        embed_data = {"title": title, "color": color}
        if description:
            embed_data["description"] = description
        if fields:
            embed_data["fields"] = [
                {"name": name, "value": value, "inline": inline}
                for name, value, inline in fields
            ]

        payload = {"embeds": [embed_data]}

        try:
            async with aiohttp.ClientSession() as session:
                # Post embed via webhook (wait=true returns the message)
                async with session.post(f"{webhook_url}?wait=true", json=payload) as resp:
                    if resp.status != 200:
                        logger.warning(f"Webhook post failed: {resp.status}")
                        # Fallback to plain text
                        lines = [f"**{title}**"]
                        if description:
                            lines.append(description)
                        if fields:
                            for name, value, _ in fields:
                                lines.append(f"**{name}:** {value}")
                        return await destination.send("\n".join(lines))
                    data = await resp.json()

            # Fetch the webhook message from Discord and forward it
            webhook_msg_id = int(data["id"])
            webhook_channel_id = int(data["channel_id"])
            webhook_channel = self._client.get_channel(webhook_channel_id)
            if webhook_channel:
                webhook_msg = await webhook_channel.fetch_message(webhook_msg_id)
                forwarded = await webhook_msg.forward(destination)
                # Delete the original webhook message to keep the channel clean
                try:
                    await webhook_msg.delete()
                except Exception:
                    pass
                return forwarded
        except Exception as e:
            logger.warning(f"Embed forwarding failed: {e}")
            # Final fallback
            lines = [f"**{title}**"]
            if description:
                lines.append(description)
            if fields:
                for name, value, _ in fields:
                    lines.append(f"**{name}:** {value}")
            return await destination.send("\n".join(lines))

    def _register_events(self):
        @self._client.event
        async def on_ready():
            logger.info(f"Discord bot logged in as {self._client.user}")
            if self._gemini:
                self._gemini.discord_username = self._client.user.name
            self._restore_mutes()

        @self._client.event
        async def on_message(message):
            # Ignore own messages
            if message.author == self._client.user:
                return

            should_respond = False
            channel_id = str(message.channel.id)

            # Check for admin commands in DMs/Group DMs
            if isinstance(message.channel, (discord.DMChannel, discord.GroupChannel)):
                if str(message.author.id) in self.config.authorized_users:
                    handled = await self._handle_command(message)
                    if handled:
                        return

            # Check if mentioned or replied to (works in all channel types)
            if self._client.user in message.mentions:
                should_respond = True
            elif (
                message.reference
                and message.reference.resolved
                and hasattr(message.reference.resolved, "author")
                and message.reference.resolved.author == self._client.user
            ):
                should_respond = True
            # Auto-respond in DMs/Group DMs (no mention needed)
            elif isinstance(message.channel, (discord.DMChannel, discord.GroupChannel)):
                if self.config.auto_respond_dms:
                    should_respond = True

            if not should_respond:
                return

            # Global disable check (admin commands still work above)
            if self._disabled:
                return

            # Check if channel is muted
            muted = getattr(self._tool_handler, "_muted_channels", None)
            if muted and channel_id in muted:
                return

            # Cooldown check (skip if there's an active response to interrupt)
            active_response = self._response_tasks.get(channel_id)
            if not (active_response and not active_response.done()):
                now = time.time()
                last = self._cooldowns.get(channel_id, 0)
                if now - last < self.config.response_cooldown:
                    return

            # Collect first image from this message (Live API 1-image limit)
            images = []
            attachment_info = []
            for att in message.attachments:
                if att.content_type and att.content_type.startswith("image/"):
                    try:
                        img_data = await att.read()
                        mime = att.content_type
                        # Extract first frame from GIFs
                        if "gif" in mime:
                            img_data, mime = self._gif_to_png(img_data)
                        images.append((img_data, mime))
                        attachment_info.append({"filename": att.filename, "type": att.content_type})
                        break
                    except Exception as e:
                        logger.warning(f"Failed to read attachment {att.filename}: {e}")

            # Check embeds for Tenor/Giphy GIFs if no attachment image found
            if not images:
                for embed in message.embeds:
                    url = None
                    if embed.type == "gifv" and embed.thumbnail and embed.thumbnail.url:
                        url = embed.thumbnail.url
                    elif embed.type == "image" and embed.image and embed.image.url:
                        url = embed.image.url
                    elif embed.thumbnail and embed.thumbnail.url:
                        url = embed.thumbnail.url
                    if url:
                        try:
                            img_data, mime = await self._download_image(url)
                            if "gif" in mime:
                                img_data, mime = self._gif_to_png(img_data)
                            images.append((img_data, mime))
                            attachment_info.append({"filename": "embed_image", "type": mime})
                            break
                        except Exception as e:
                            logger.warning(f"Failed to download embed image: {e}")

            # Queue this message for batching
            if channel_id not in self._batch_queues:
                self._batch_queues[channel_id] = []
            self._batch_queues[channel_id].append({
                "message": message,
                "images": images,
                "attachment_info": attachment_info,
            })

            # Cancel pending left-on-read follow-up for this channel
            followup = self._followup_tasks.pop(channel_id, None)
            if followup and not followup.done():
                followup.cancel()

            # Cancel existing debounce timer and start a new one
            if channel_id in self._batch_tasks:
                self._batch_tasks[channel_id].cancel()
            self._batch_tasks[channel_id] = asyncio.create_task(
                self._batch_debounce(channel_id)
            )

    async def _handle_command(self, message):
        """Handle authorized user commands. Returns True if handled."""
        content = message.content.strip()
        if not content.startswith("!"):
            return False

        parts = content[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "status":
            uptime = datetime.now() - self._start_time
            h, r = divmod(int(uptime.total_seconds()), 3600)
            m, s = divmod(r, 60)
            connected = "Yes" if self._gemini and self._gemini._connected.is_set() else "No"
            muted = getattr(self._tool_handler, "_muted_channels", set())
            await self._send_embed(message.channel, "Bot Status", fields=[
                ("Uptime", f"{h}h {m}m {s}s", True),
                ("Gemini Connected", connected, True),
                ("Guilds", str(len(self._client.guilds)), True),
                ("Enabled", "No" if self._disabled else "Yes", True),
                ("Muted Channels", str(len(muted)), True),
            ], color=0x57F287 if not self._disabled else 0xED4245)
            return True

        elif cmd == "enable":
            self._disabled = False
            await message.channel.send("Bot enabled.")
            return True

        elif cmd == "disable":
            self._disabled = True
            await message.channel.send("Bot disabled. Use `!enable` to re-enable.")
            return True

        elif cmd == "say":
            if arg:
                await message.channel.send(arg)
            return True

        elif cmd == "relay":
            if arg and self._relay_callback:
                await self._relay_callback(f"DISCORD ACTIVITY: [Admin Command] {arg}")
                await message.channel.send("Relayed to VRChat session.")
            return True

        elif cmd == "mute":
            return await self._cmd_mute(message, arg)

        elif cmd == "unmute":
            return await self._cmd_unmute(message, arg)

        elif cmd == "mutes":
            return await self._cmd_list_mutes(message)

        elif cmd == "reconnect":
            if self._gemini:
                await message.channel.send("Reconnecting Gemini session...")
                await self._gemini.disconnect()
                self._gemini._closing = False
                asyncio.create_task(self._gemini.run_forever())
            return True

        elif cmd == "personality":
            if not arg:
                current = self._personality.get_current()
                await message.channel.send(f"Current personality: **{current['name']}**")
            else:
                result = self._personality.switch(arg)
                if result:
                    if self._gemini:
                        await self._gemini.inject_context(
                            f"[System] Personality switched to: {result['name']}. {result['prompt']}"
                        )
                    await message.channel.send(f"Switched to **{result['name']}**")
                else:
                    available = [p["name"] for p in self._personality.list_personalities()]
                    await message.channel.send(f"Unknown personality. Available: {', '.join(available)}")
            return True

        elif cmd == "clear":
            target = arg or str(message.channel.id)
            self._conversations._conversations.pop(target, None)
            path = self._conversations._file_for(target)
            if path.exists():
                path.unlink()
            await message.channel.send(f"Cleared conversation history for `{target}`.")
            return True

        elif cmd == "reload":
            try:
                self.config = BotConfig()
                await message.channel.send("Config reloaded.")
            except Exception as e:
                await message.channel.send(f"Reload failed: {e}")
            return True

        elif cmd == "help":
            await self._send_embed(message.channel, "Admin Commands", fields=[
                ("`!status`", "Bot status & info", False),
                ("`!enable` / `!disable`", "Toggle bot responses", False),
                ("`!say <text>`", "Send a message as the bot", False),
                ("`!relay <text>`", "Relay message to VRChat AI", False),
                ("`!mute <channel_id> [minutes|perm]`", "Mute a channel", False),
                ("`!unmute <channel_id>`", "Unmute a channel", False),
                ("`!mutes`", "List active mutes", False),
                ("`!reconnect`", "Reconnect Gemini session", False),
                ("`!personality [name]`", "View/switch personality", False),
                ("`!clear [channel_id]`", "Clear conversation history", False),
                ("`!reload`", "Reload config", False),
                ("`!help`", "This help", False),
            ], color=0x5865F2)
            return True

        return False

    async def _cmd_mute(self, message, arg):
        """Admin mute command: !mute <channel_id> [minutes|perm]"""
        parts = arg.split()
        if not parts:
            await message.channel.send("Usage: `!mute <channel_id> [minutes|perm]`")
            return True

        channel_id = parts[0]
        duration_str = parts[1] if len(parts) > 1 else "perm"

        if not hasattr(self._tool_handler, "_muted_channels"):
            self._tool_handler._muted_channels = set()
        self._tool_handler._muted_channels.add(channel_id)

        from discord_bot.tools.discord_actions import DiscordActionsTool

        if duration_str.lower() in ("perm", "permanent", "forever"):
            # Permanent mute - use a very far future expiry
            DiscordActionsTool._save_mute(channel_id, time.time() + 365 * 24 * 3600, "")
            await message.channel.send(f"Permanently muted channel `{channel_id}`.")
        else:
            try:
                minutes = int(duration_str)
            except ValueError:
                await message.channel.send("Duration must be a number of minutes or `perm`.")
                return True
            expires_at = time.time() + minutes * 60
            DiscordActionsTool._save_mute(channel_id, expires_at, "")

            async def _unmute():
                await asyncio.sleep(minutes * 60)
                self._tool_handler._muted_channels.discard(channel_id)
                DiscordActionsTool._remove_mute(channel_id)
                logger.info(f"Admin unmuted channel {channel_id}")

            asyncio.create_task(_unmute())
            await message.channel.send(f"Muted channel `{channel_id}` for {minutes} minutes.")
        return True

    async def _cmd_unmute(self, message, arg):
        """Admin unmute command: !unmute <channel_id>"""
        channel_id = arg.strip()
        if not channel_id:
            await message.channel.send("Usage: `!unmute <channel_id>`")
            return True

        muted = getattr(self._tool_handler, "_muted_channels", set())
        muted.discard(channel_id)

        from discord_bot.tools.discord_actions import DiscordActionsTool
        DiscordActionsTool._remove_mute(channel_id)
        await message.channel.send(f"Unmuted channel `{channel_id}`.")
        return True

    async def _cmd_list_mutes(self, message):
        """Admin list mutes command: !mutes"""
        from discord_bot.tools.discord_actions import DiscordActionsTool
        active = DiscordActionsTool.load_persisted_mutes()
        if not active:
            await self._send_embed(message.channel, "Active Mutes", description="No active mutes.", color=0x57F287)
            return True

        fields = []
        now = time.time()
        for cid, data in active.items():
            remaining = data["expires_at"] - now
            if remaining > 364 * 24 * 3600:
                fields.append((f"`{cid}`", "Permanent", False))
            else:
                mins = int(remaining / 60)
                fields.append((f"`{cid}`", f"{mins}m remaining", False))
        await self._send_embed(message.channel, "Active Mutes", fields=fields, color=0xFEE75C)
        return True

    async def _batch_debounce(self, channel_id):
        """Wait for the batch window, then process all queued messages together."""
        await asyncio.sleep(self.config.batch_window_ms / 1000.0)
        batch = self._batch_queues.pop(channel_id, [])
        self._batch_tasks.pop(channel_id, None)
        if not batch:
            return
        # Cancel any in-progress response for this channel (interrupt)
        existing = self._response_tasks.get(channel_id)
        if existing and not existing.done():
            existing.cancel()
            logger.info(f"Interrupted ongoing response in {channel_id}")
        task = asyncio.create_task(self._respond_to_batch(channel_id, batch))
        self._response_tasks[channel_id] = task

    async def _respond_to_batch(self, channel_id, batch):
        """Process a batch of messages through Gemini and respond."""
        if not self._gemini or not self._gemini._connected.is_set():
            logger.warning("Gemini not connected, skipping response")
            return

        # Build structured context turns from conversation history
        all_images = []
        message_lines = []
        last_message = batch[-1]["message"]

        channel_info = ""
        if isinstance(last_message.channel, discord.DMChannel):
            recipient = last_message.channel.recipient
            channel_info = f"DM with {recipient.display_name or recipient.name}" if recipient else "DM"
        elif isinstance(last_message.channel, discord.GroupChannel):
            channel_info = f"Group DM: {last_message.channel.name or 'unnamed'}"
        elif hasattr(last_message.channel, "name") and last_message.guild:
            channel_info = f"#{last_message.channel.name} in {last_message.guild.name}"

        # Only fetch context turns for fresh sessions (no session handle)
        # Resumed sessions already have context from the Gemini server
        needs_context = not self._gemini._session_resumed
        if needs_context:
            context_turns = self._conversations.get_turns(
                channel_id,
                count=self.config.context_message_count,
                channel_info=channel_info,
            )
        else:
            context_turns = None

        for entry in batch:
            msg = entry["message"]
            user_display = msg.author.display_name or msg.author.name
            image_note = ""
            if entry["images"]:
                all_images.extend(entry["images"])
                image_note = f" [attached {len(entry['images'])} image(s)]"
            message_lines.append(f"{user_display} (ID:{msg.author.id}): {msg.clean_content}{image_note}")

            # Log each user message to conversation store
            self._conversations.add_message(
                channel_id, "user", msg.clean_content,
                username=user_display,
                attachments=entry["attachment_info"] or None,
            )

        new_message = f"[CHANNEL: {channel_info}]\n" + "\n".join(message_lines)

        # Serialize Gemini interactions -- only one channel at a time
        try:
            async with self._response_lock:
                # Set current channel inside lock so tools target the right channel
                self._tool_handler._current_channel = last_message.channel
                self._tool_handler._tool_sent_message = False

                async with last_message.channel.typing():
                    delay = self.config.typing_delay_ms / 1000.0
                    if delay > 0:
                        await asyncio.sleep(delay)

                    response = None
                    try:
                        if context_turns:
                            coro = self._gemini.send_with_context(
                                context_turns,
                                new_message,
                                images=all_images if all_images else None,
                            )
                        else:
                            coro = self._gemini.send_message(
                                new_message,
                                images=all_images if all_images else None,
                            )
                        response = await asyncio.wait_for(coro, timeout=60.0)
                        # After first successful context send, mark session as having context
                        if context_turns:
                            self._gemini._session_resumed = True
                    except asyncio.TimeoutError:
                        logger.warning("Gemini response timed out")

                if not response or response.startswith("[Error:"):
                    logger.warning(f"Bad response from Gemini: {response}")
                    reconnecting = self._gemini.report_bad_response()
                    if reconnecting and self.config.show_reconnecting:
                        await last_message.channel.send("-# reconnecting...")
                    elif not reconnecting:
                        await last_message.channel.send("-# no response...")
                    return

                self._gemini.report_good_response()

                # Strip channel tag the model may echo back
                response = re.sub(r'^\[CHANNEL:[^\]]*\]\s*', '', response)
                response = self._sanitize_generated_text(response)
                if not response:
                    logger.warning("Gemini response was empty after sanitization")
                    return
                # Strip mass pings
                response = response.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")

                # Split into multiple messages for natural feel
                parts = self._split_natural(response)
                channel = last_message.channel
                use_reply = random.random() < 0.3
                last_sent = None
                for i, part in enumerate(parts):
                    if len(part) > self.config.max_message_length:
                        chunks = self._split_message(part, self.config.max_message_length)
                        for j, chunk in enumerate(chunks):
                            if i == 0 and j == 0 and use_reply:
                                last_sent = await last_message.reply(chunk, mention_author=False)
                            else:
                                last_sent = await channel.send(chunk)
                            await asyncio.sleep(0.3)
                    else:
                        if i == 0 and use_reply:
                            last_sent = await last_message.reply(part, mention_author=False)
                        else:
                            last_sent = await channel.send(part)
                    if i < len(parts) - 1:
                        next_len = len(parts[i + 1])
                        typing_delay = 0.5 + next_len * 0.065
                        async with channel.typing():
                            await asyncio.sleep(min(typing_delay, 8.0))

                self._conversations.add_message(channel_id, "assistant", response)
                self._cooldowns[channel_id] = time.time()

                # Schedule left-on-read follow-up (20% chance, skip if conversation ended naturally)
                if last_sent and random.random() < 0.2 and not self._looks_like_conversation_end(response):
                    old = self._followup_tasks.pop(channel_id, None)
                    if old and not old.done():
                        old.cancel()
                    self._followup_tasks[channel_id] = asyncio.create_task(
                        self._followup_on_read(channel_id, last_sent, last_message.author)
                    )
                else:
                    # Cancel any existing follow-up if we're not scheduling a new one
                    old = self._followup_tasks.pop(channel_id, None)
                    if old and not old.done():
                        old.cancel()

        except asyncio.CancelledError:
            logger.info(f"Response interrupted in {channel_id}")
        except discord.errors.Forbidden:
            logger.warning(f"No permission to send in {channel_id}")
        except Exception as e:
            logger.error(f"Response error: {e}")

    _CONVERSATION_END_PATTERNS = re.compile(
        r'\b(bye|goodbye|good\s*night|gn|later|cya|see\s*ya|see\s*you|'
        r'take\s*care|peace\s*out|ttyl|talk\s*to\s*you\s*later|'
        r'gotta\s*go|heading\s*out|signing\s*off|nighty?\s*night|'
        r'sweet\s*dreams|sleep\s*well|have\s*a\s*good\s*(one|night|day)|'
        r'catch\s*you\s*later|farewell|adios|sayonara)\b',
        re.IGNORECASE,
    )

    def _looks_like_conversation_end(self, response: str) -> bool:
        """Check if the AI's last response looks like a natural conversation ending."""
        # Only check the last ~200 chars to avoid matching mid-conversation goodbyes
        tail = response[-200:] if len(response) > 200 else response
        return bool(self._CONVERSATION_END_PATTERNS.search(tail))

    async def _followup_on_read(self, channel_id, bot_message, user):
        """Wait 5-15 minutes, then send a follow-up if no one responded."""
        delay = random.randint(300, 900)  # 5-15 minutes
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        # Check if still valid (not disabled, not muted)
        if self._disabled:
            return
        muted = getattr(self._tool_handler, "_muted_channels", None)
        if muted and channel_id in muted:
            return
        if not self._gemini or not self._gemini._connected.is_set():
            return

        user_display = user.display_name or user.name
        minutes = delay // 60
        prompt = (
            f"[CHANNEL: left-on-read followup]\n"
            f"{user_display} (ID:{user.id}) has not responded to your last message for "
            f"{minutes} minutes. Respond with a short follow-up complaining about being "
            f"left on read. Mention them with <@{user.id}>. Be dramatic, funny, or annoyed "
            f"about being ignored. Keep it to 1-2 sentences max. Do NOT use any tools, just respond directly."
        )

        try:
            async with self._response_lock:
                self._tool_handler._current_channel = bot_message.channel
                self._tool_handler._tool_sent_message = False
                response = await asyncio.wait_for(
                    self._gemini.send_message(prompt),
                    timeout=30.0,
                )

            # If the AI already sent a message via sendDiscordMessage tool, skip the reply
            if self._tool_handler._tool_sent_message:
                self._tool_handler._tool_sent_message = False
                logger.debug(f"Follow-up in {channel_id}: skipped reply (tool already sent message)")
                return

            if response and not response.startswith("[Error:"):
                response = re.sub(r'^\[CHANNEL:[^\]]*\]\s*', '', response)
                response = self._sanitize_generated_text(response)
                if not response:
                    return
                response = response.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
                await bot_message.reply(response, mention_author=False)
                self._conversations.add_message(channel_id, "assistant", response)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Follow-up failed in {channel_id}: {e}")
        finally:
            self._followup_tasks.pop(channel_id, None)

    @staticmethod
    def _gif_to_png(gif_bytes):
        """Extract the first frame of a GIF and return it as PNG bytes."""
        from PIL import Image
        img = Image.open(io.BytesIO(gif_bytes))
        img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"

    @staticmethod
    async def _download_image(url):
        """Download an image from a URL and return (bytes, mime_type)."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                ct = resp.content_type or "image/png"
                return data, ct

    @staticmethod
    def _split_natural(text):
        """Split text into multiple messages at natural breakpoints.
        Splits on double newlines, then on single newlines between
        sentences, keeping short fragments together."""
        import re
        # First split on explicit paragraph breaks
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs

        # Split on sentence-ending punctuation followed by space
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z<@])', text)
        if len(parts) <= 1:
            return [text]

        # Merge very short parts together (under ~60 chars)
        merged = []
        buf = ""
        for part in parts:
            if buf and len(buf) + len(part) + 1 > 120:
                merged.append(buf)
                buf = part
            else:
                buf = (buf + " " + part).strip() if buf else part
        if buf:
            merged.append(buf)
        return merged if len(merged) > 1 else [text]

    @staticmethod
    def _split_message(text, max_len):
        """Split a long message into chunks at sentence boundaries."""
        chunks = []
        while len(text) > max_len:
            # Find last sentence-ending punctuation before max_len
            split_at = max_len
            for sep in [". ", "! ", "? ", "\n", ", ", " "]:
                idx = text.rfind(sep, 0, max_len)
                if idx > max_len // 2:
                    split_at = idx + len(sep)
                    break
            chunks.append(text[:split_at].rstrip())
            text = text[split_at:].lstrip()
        if text:
            chunks.append(text)
        return chunks
