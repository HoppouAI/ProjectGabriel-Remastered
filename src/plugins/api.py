"""Plugin API surface.

Plugins subclass `Plugin` and use `PluginContext` to register stuff.
The surface is intentionally small so it can be extended later without
breaking older plugins. New hooks should bump PLUGIN_API_VERSION in
loader.py and be documented in plugins/README.md.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module-level registries. Populated as plugins call ctx.register_*.
# Kept module-level so the rest of the app (main.py, TTS selection) can
# look these up without holding a reference to the PluginManager.
_tts_providers: dict[str, Callable[..., Any]] = {}
_stt_providers: dict[str, Callable[..., Any]] = {}
_event_subscribers: dict[str, list[Callable[..., Any]]] = {}
# Chatbox sources let plugins contribute their own VRChat chatbox displays
# (now-playing screens, status overlays, etc) and also signal "busy" so the
# idle banner gets suppressed while they're active. Lower priority value runs
# first when picking what to show.
_chatbox_sources: dict[str, tuple[int, Any]] = {}
# Prompt contributors are callables that return a string (or None) appended
# to the system prompt every time it's built. Lets plugins inject dynamic
# context like current mood, weather, time-of-day flavor, etc.
_prompt_contributors: dict[str, Callable[[], Any]] = {}

# Discord-side registries. Mirror their main-session counterparts but
# scoped to the Discord bot's separate Gemini Live session, so plugins
# can extend the bot without affecting the VRChat session and vice
# versa. Plugins reach these via `ctx.discord.*`.
_discord_tool_classes: list[type] = []
_discord_prompt_contributors: dict[str, Callable[[], Any]] = {}
_discord_event_subscribers: dict[str, list[Callable[..., Any]]] = {}


class Plugin:
    """Base class for plugins. Subclass and override setup/teardown.

    Class-level fields can be overridden per plugin or filled in from the
    manifest at load time, whichever you prefer.
    """

    name: str = "unnamed"
    version: str = "0.0.0"
    api_version: int = 1
    description: str = ""
    author: str = ""

    def setup(self, ctx: "PluginContext"):
        """Called once after the plugin module loads. Register tools and
        providers, subscribe to events, etc. Avoid heavy I/O here, the
        host is still spinning up."""
        pass

    def teardown(self, ctx: "PluginContext"):
        """Called once on shutdown. Close sockets, stop threads, save
        state."""
        pass


class PluginContext:
    """Handed to each plugin so it can hook into the app without
    importing internal modules directly. One context per plugin so logs
    and config lookups are scoped to that plugin's name."""

    def __init__(self, plugin_name: str, config, plugin_dir: Path):
        self.plugin_name = plugin_name
        self.config = config
        self.plugin_dir = plugin_dir
        self.logger = logging.getLogger(f"plugin.{plugin_name}")
        # Filled in later by PluginManager.bind_app() once the rest of
        # the app is wired up. Plugins that need these can read them
        # lazily or grab them inside a 'startup' event handler.
        self._app: dict[str, Any] = {}

    def bind_app(self, **refs):
        self._app.update(refs)

    @property
    def audio(self):
        return self._app.get("audio")

    @property
    def osc(self):
        return self._app.get("osc")

    @property
    def session(self):
        return self._app.get("session")

    @property
    def tool_handler(self):
        return self._app.get("tool_handler")

    def plugin_config(self, key: str | None = None, default: Any = None):
        """Read this plugin's config from `config.yml` under
        `plugins.<name>.<key>`. Pass no key to get the whole sub-dict."""
        if self.config is None:
            return default if key is not None else {}
        if key is None:
            return self.config.get("plugins", self.plugin_name, default={}) or {}
        return self.config.get("plugins", self.plugin_name, key, default=default)

    def data_dir(self) -> Path:
        """Per-plugin data dir under data/plugins/<name>/. Created on
        demand."""
        d = Path("data") / "plugins" / self.plugin_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def discord(self) -> "DiscordPluginContext":
        """Sub-context for Discord bot integration. Use this to register
        tools, prompt contributors, and event subscribers that live on
        the Discord bot's separate Gemini Live session.

        Existing main-session hooks (`ctx.register_tool`,
        `ctx.register_prompt_contributor`, `ctx.subscribe`,
        `ctx.send_system_instruction`, `ctx.send_user_text`) are
        unchanged -- they continue to target the VRChat session only.
        Anything Discord-scoped goes through `ctx.discord.*`.

        The Discord bot does not have to be running for plugins to
        register here. If the Discord bot is disabled in config the
        registrations are still kept and simply never used. send_*
        helpers return False when the Discord session isnt up yet.
        """
        if not hasattr(self, "_discord_ctx"):
            self._discord_ctx = DiscordPluginContext(self)
        return self._discord_ctx

    # registration helpers ---------------------------------------------------

    def register_tool(self, tool_cls):
        """Register a tool class. Must extend `src.tools._base.BaseTool`.
        Equivalent to decorating the class with `@register_tool`."""
        from src.tools._base import register_tool as _reg
        _reg(tool_cls)
        # banner shows the per-plugin tool counts, no need to spam this on console
        self.logger.debug(f"registered tool {tool_cls.__name__}")

    def register_tts(self, name: str, factory: Callable[..., Any]):
        """Register a TTS provider. `factory(config)` should return an
        object with the same interface as the built in providers in
        `src/tts.py` (typically `start()`, `stop()`, plus whatever the
        Gemini Live session expects of a TTS provider)."""
        if name in _tts_providers:
            self.logger.warning(f"tts provider '{name}' already registered, overwriting")
        _tts_providers[name] = factory
        self.logger.info(f"registered tts provider '{name}'")

    def register_stt(self, name: str, factory: Callable[..., Any]):
        """Register an STT provider. The main loop uses Gemini Live for
        STT natively, so this hook is mainly for plugins that want their
        own pipeline (e.g. local Whisper inside a Discord plugin)."""
        if name in _stt_providers:
            self.logger.warning(f"stt provider '{name}' already registered, overwriting")
        _stt_providers[name] = factory
        self.logger.info(f"registered stt provider '{name}'")

    def subscribe(self, event: str, callback: Callable[..., Any]):
        """Subscribe to an app event. Callback can be sync or async.
        Built in events: 'startup', 'shutdown', 'message_in',
        'message_out'. Plugins can also emit and subscribe to their own
        custom events."""
        _event_subscribers.setdefault(event, []).append(callback)
        self.logger.debug(f"subscribed to event '{event}'")

    def register_chatbox_source(self, name: str, source: Any, priority: int = 100):
        """Register a VRChat chatbox display source.

        `source` must be an object with two methods:
          - `is_active() -> bool` -- True while this source wants screen time,
            also used by the host to mark itself as busy and suppress the idle banner
          - `render() -> str | None` -- the chatbox text to show, or None to skip

        Lower `priority` wins when multiple sources are active at once.
        Built in displays (local music, lyria) sit at priority 10/20, plugins
        default to 100 so they yield to host displays unless they ask for less.
        """
        if name in _chatbox_sources:
            self.logger.warning(f"chatbox source '{name}' already registered, overwriting")
        _chatbox_sources[name] = (priority, source)
        self.logger.info(f"registered chatbox source '{name}' (priority={priority})")

    def unregister_chatbox_source(self, name: str):
        _chatbox_sources.pop(name, None)

    def register_prompt_contributor(self, name: str, fn: Callable[[], Any]):
        """Register a function that contributes text to the system prompt.

        `fn()` is called every time the system prompt is built (session
        start, reconnect, personality switch). Should return a string to
        append, or None / empty string to skip. Exceptions are caught.

        Use this for dynamic, opt-in context that the model should know
        about - current mood, recent activity, status flags, etc. The
        contributor's text is appended after all built in appends.
        """
        if name in _prompt_contributors:
            self.logger.warning(f"prompt contributor '{name}' already registered, overwriting")
        _prompt_contributors[name] = fn
        self.logger.info(f"registered prompt contributor '{name}'")

    def unregister_prompt_contributor(self, name: str):
        _prompt_contributors.pop(name, None)

    # runtime messaging ------------------------------------------------------

    async def send_system_instruction(self, text: str):
        """Push a mid-session system instruction to the model, same path
        the WebUI uses. Wraps it as `SYSTEM INSTRUCTION: <text>`
        and sends via `send_client_content_safe`, which waits until the
        model stops speaking before injecting so it doesnt cut off a
        reply.

        Only works after the live session is up. During `setup()` the
        session is None and this returns False without doing anything.
        Returns True if the message was queued, False otherwise.
        """
        session = self._app.get("session")
        if session is None or not getattr(session, "_session", None):
            self.logger.warning("send_system_instruction called before session was ready, skipping")
            return False
        try:
            from google.genai import types
            await session.send_client_content_safe(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"SYSTEM INSTRUCTION: {text}")],
                ),
                turn_complete=True,
            )
            return True
        except Exception as e:
            self.logger.error(f"send_system_instruction failed: {e}")
            return False

    async def send_user_text(self, text: str):
        """Send a normal user-style text message into the live session,
        as if it came from the chat input. The model will respond like
        any other user turn. Returns True on success.
        """
        session = self._app.get("session")
        if session is None:
            self.logger.warning("send_user_text called before session was ready, skipping")
            return False
        if not hasattr(session, "send_text"):
            self.logger.warning("session has no send_text, cant inject user text")
            return False
        try:
            await session.send_text(text)
            return True
        except Exception as e:
            self.logger.error(f"send_user_text failed: {e}")
            return False

def get_tts_factory(name: str):
    return _tts_providers.get(name)


def get_stt_factory(name: str):
    return _stt_providers.get(name)


def list_tts_factories() -> list[str]:
    return sorted(_tts_providers.keys())


def list_stt_factories() -> list[str]:
    return sorted(_stt_providers.keys())


def iter_chatbox_sources():
    """Yield (name, source) pairs sorted by ascending priority."""
    for name, (_prio, src) in sorted(_chatbox_sources.items(), key=lambda kv: kv[1][0]):
        yield name, src


def collect_prompt_contributions() -> list[str]:
    """Call every registered prompt contributor and return their non-empty
    results in registration order. Errors get logged and skipped."""
    out: list[str] = []
    for name, fn in _prompt_contributors.items():
        try:
            text = fn()
        except Exception as e:
            logger.error(f"prompt contributor '{name}' raised: {e}")
            continue
        if text:
            out.append(str(text).strip())
    return out


def emit_event(event: str, *args, **kwargs):
    """Fire an event to all subscribers. Each handler's exception is
    caught so a single bad plugin can't break the rest of the chain.
    Async callbacks are scheduled on the running loop if there is one."""
    for cb in _event_subscribers.get(event, []):
        try:
            res = cb(*args, **kwargs)
            if asyncio.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(res)
                except RuntimeError:
                    # no running loop, run it synchronously
                    asyncio.run(res)
        except Exception as e:
            logger.error(f"event '{event}' subscriber {cb} raised: {e}")


# Discord plugin surface --------------------------------------------------

# Reference to the Discord bot's GeminiTextSession, set by the bot at
# startup so plugins can push mid-session messages into it. None when
# the bot is disabled or hasn't connected yet.
_discord_session: Any = None
# Reference to the Discord tool handler so register_tool can hot-attach
# new tools after the handler is already constructed.
_discord_tool_handler: Any = None


def _bind_discord_session(session, tool_handler):
    """Called by the Discord bot after it has constructed both objects
    so module-level helpers (and the DiscordPluginContext send_* methods)
    can find them. Plugins shouldn't call this directly."""
    global _discord_session, _discord_tool_handler
    _discord_session = session
    _discord_tool_handler = tool_handler
    # Late-binding any tools registered before the bot was up.
    if tool_handler is not None and _discord_tool_classes:
        for cls in _discord_tool_classes:
            try:
                tool_handler.register_plugin_tool(cls)
            except Exception as e:
                logger.error(f"failed to attach plugin discord tool {cls.__name__}: {e}")


def iter_discord_tool_classes() -> list[type]:
    """Return the currently registered plugin tool classes for the
    Discord bot. Used by DiscordToolHandler at startup to instantiate
    them alongside the built-in tools."""
    return list(_discord_tool_classes)


def collect_discord_prompt_contributions() -> list[str]:
    """Same as collect_prompt_contributions but for the Discord bot's
    system prompt. Errors are logged and skipped so a broken plugin
    cant kill the bot's prompt build."""
    out: list[str] = []
    for name, fn in _discord_prompt_contributors.items():
        try:
            text = fn()
        except Exception as e:
            logger.error(f"discord prompt contributor '{name}' raised: {e}")
            continue
        if text:
            out.append(str(text).strip())
    return out


def emit_discord_event(event: str, *args, **kwargs):
    """Fire a Discord-scoped event. Built-in events: 'bot_ready',
    'dm_received', 'mention_received', 'message_sent'. Plugins can
    define their own too."""
    for cb in _discord_event_subscribers.get(event, []):
        try:
            res = cb(*args, **kwargs)
            if asyncio.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(res)
                except RuntimeError:
                    asyncio.run(res)
        except Exception as e:
            logger.error(f"discord event '{event}' subscriber {cb} raised: {e}")


class DiscordPluginContext:
    """Per-plugin handle for the Discord bot's separate Gemini Live
    session. Reached via `ctx.discord` from inside `Plugin.setup()`.

    Mirrors the main-session API (`register_tool`,
    `register_prompt_contributor`, `subscribe`, `send_system_instruction`,
    `send_user_text`) but everything here is scoped to the bot. A plugin
    can register on both sides safely -- they don't share state.
    """

    def __init__(self, parent: PluginContext):
        self._parent = parent
        self.logger = logging.getLogger(f"plugin.{parent.plugin_name}.discord")

    # passthroughs to keep ergonomics nice
    @property
    def plugin_name(self) -> str:
        return self._parent.plugin_name

    @property
    def session(self):
        """The Discord bot's GeminiTextSession, or None if the bot is
        disabled or hasnt finished connecting."""
        return _discord_session

    @property
    def tool_handler(self):
        """The DiscordToolHandler instance, or None if the bot isnt up."""
        return _discord_tool_handler

    def register_tool(self, tool_cls: type):
        """Register a tool class against the Discord bot's tool handler.

        Tool class must define `declarations(self)` returning a list of
        FunctionDeclaration and `async handle(self, name, args)`
        returning a result dict or None when it doesnt own the call.
        Same shape as the built-in Discord tools in
        `discord_bot/tools/`.

        Safe to call from `setup()` even if the bot isn't running yet --
        the class is held and attached to the handler when the bot
        starts. If the bot is already running it's attached immediately.
        """
        if tool_cls in _discord_tool_classes:
            self.logger.warning(
                f"discord tool {tool_cls.__name__} already registered, skipping"
            )
            return
        _discord_tool_classes.append(tool_cls)
        if _discord_tool_handler is not None:
            try:
                _discord_tool_handler.register_plugin_tool(tool_cls)
            except Exception as e:
                self.logger.error(
                    f"failed to attach discord tool {tool_cls.__name__}: {e}"
                )
        self.logger.info(f"registered discord tool {tool_cls.__name__}")

    def register_prompt_contributor(self, name: str, fn: Callable[[], Any]):
        """Append text to the Discord bot's system prompt. `fn()` is
        called every time the bot rebuilds its prompt (session start,
        reconnect). Return a string or None. Exceptions are caught."""
        scoped = f"{self.plugin_name}.{name}"
        if scoped in _discord_prompt_contributors:
            self.logger.warning(
                f"discord prompt contributor '{scoped}' already registered, overwriting"
            )
        _discord_prompt_contributors[scoped] = fn
        self.logger.info(f"registered discord prompt contributor '{scoped}'")

    def unregister_prompt_contributor(self, name: str):
        scoped = f"{self.plugin_name}.{name}"
        _discord_prompt_contributors.pop(scoped, None)

    def subscribe(self, event: str, callback: Callable[..., Any]):
        """Subscribe to a Discord-scoped event. Built-in events:
          - 'bot_ready' -- (client) when the discord client connects
          - 'dm_received' -- (message) raw discord.Message for a DM
          - 'mention_received' -- (message) raw discord.Message
          - 'message_sent' -- (channel_id: str, text: str) bot replied
        Sync or async callbacks both fine."""
        _discord_event_subscribers.setdefault(event, []).append(callback)
        self.logger.debug(f"subscribed to discord event '{event}'")

    async def send_system_instruction(self, text: str):
        """Inject a SYSTEM INSTRUCTION style message into the Discord
        Gemini session. Same wrapping as the main-session helper:
        prefixes with 'SYSTEM INSTRUCTION: '. Returns False if the
        bot isnt connected."""
        session = _discord_session
        if session is None or not getattr(session, "_session", None):
            self.logger.debug("discord send_system_instruction: bot not connected")
            return False
        try:
            await session.inject_context(f"SYSTEM INSTRUCTION: {text}")
            return True
        except Exception as e:
            self.logger.error(f"discord send_system_instruction failed: {e}")
            return False

    async def send_user_text(self, text: str):
        """Inject a user-style text message into the Discord Gemini
        session. The bot will see this as if a user typed it but no
        actual Discord message is created. Returns False if the bot
        isnt connected."""
        session = _discord_session
        if session is None or not getattr(session, "_session", None):
            self.logger.debug("discord send_user_text: bot not connected")
            return False
        try:
            await session.inject_context(text)
            return True
        except Exception as e:
            self.logger.error(f"discord send_user_text failed: {e}")
            return False
