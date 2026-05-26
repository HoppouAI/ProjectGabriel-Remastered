"""Plugin API surface.

Plugins subclass `Plugin` and use `PluginContext` to register stuff.
The surface is intentionally small so it can be extended later without
breaking older plugins. New hooks should bump PLUGIN_API_VERSION in
loader.py and be documented in plugins/README.md.
"""
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Sensitive config paths/attrs that plugins are never allowed to read.
# Match is a case-insensitive substring check against each path element
# OR the joined "a.b.c" path. Be generous, plugins should never need
# secrets from the host config anyway, they get their own scoped subdict
# via ctx.plugin_config().
_SENSITIVE_PATTERNS = (
    "api_key",
    "apikey",
    "backup_keys",
    "token",
    "password",
    "secret",
    "cookie",
    "webhook",
    "mongodb_uri",
    "mongo_uri",
    "connection_string",
    "private_key",
    "ssh_key",
    "auth_header",
)

# Attribute names on the real Config that plugins cant touch directly,
# either because they expose secrets or because they let the plugin
# rotate keys / mutate the live key list.
_BLOCKED_CONFIG_ATTRS = frozenset({
    "api_key",
    "backup_keys",
    "rotate_key",
    "_keys",
    "_key_index",
    "_data",     # raw nested dict, contains everything including secrets
    "_raw",
    "vrchat_api_password",
})


def _path_is_sensitive(*keys) -> bool:
    """True if any element of the path, or the joined dotted form,
    matches one of the sensitive patterns."""
    if not keys:
        # A bare get() with no path returns the whole config dict, which
        # would leak every secret in one call. Always deny.
        return True
    parts = [str(k).lower() for k in keys]
    joined = ".".join(parts)
    for pat in _SENSITIVE_PATTERNS:
        if pat in joined:
            return True
    return False


class SafeConfigView:
    """A read-only proxy around the host Config that hides secrets from
    plugins. Plugins get this as `ctx.config`. Most reads pass straight
    through to the real Config, but anything matching a sensitive
    pattern raises PermissionError so a misbehaving plugin can't silently
    exfiltrate the gemini api key, vrchat password, mongo connection
    string, etc.

    Plugins should read their own settings through `ctx.plugin_config()`
    which is already scoped under `plugins.<plugin_name>` and not
    exposed to the secret denylist (so a plugin can store its own api
    keys there if it really needs to).
    """

    __slots__ = ("_real", "_plugin")

    def __init__(self, real_config: Any, plugin_name: str):
        object.__setattr__(self, "_real", real_config)
        object.__setattr__(self, "_plugin", plugin_name)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(
                f"plugin '{self._plugin}' tried to read private config attr '{name}'"
            )
        if name in _BLOCKED_CONFIG_ATTRS:
            raise PermissionError(
                f"plugin '{self._plugin}' is not allowed to read sensitive config "
                f"attribute '{name}'. Use ctx.plugin_config() for your own "
                f"settings under plugins.{self._plugin}.* in config.yml."
            )
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        raise PermissionError(
            f"plugin '{self._plugin}' tried to mutate config attr '{name}'. "
            f"Plugin config is read-only."
        )

    def get(self, *keys, default=None):
        if _path_is_sensitive(*keys):
            raise PermissionError(
                f"plugin '{self._plugin}' tried to read sensitive config path "
                f"'{'.'.join(str(k) for k in keys) or '<root>'}'. Plugins are "
                f"sandboxed away from secrets. Use ctx.plugin_config() for your "
                f"own settings."
            )
        return self._real.get(*keys, default=default)

    def __repr__(self):
        return f"<SafeConfigView for plugin={self._plugin!r}>"

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


# Periodic task registry. Each entry is a dict with keys:
#   name, interval_seconds, fn, run_immediately, plugin_name,
#   thread (set once started), stop_flag (threading.Event)
# Spawned by `start_periodic_tasks()` which loader.py calls right after
# the 'startup' event fires.
_periodic_tasks: list[dict] = []
_periodic_started: bool = False


def register_periodic_task(
    plugin_name: str,
    name: str,
    interval_seconds: float,
    fn: Callable[[], Any],
    run_immediately: bool = False,
):
    """Module-level register. Most callers should use
    `ctx.register_periodic_task()` from inside a plugin instead."""
    _periodic_tasks.append({
        "plugin_name": plugin_name,
        "name": name,
        "interval_seconds": float(interval_seconds),
        "fn": fn,
        "run_immediately": bool(run_immediately),
        "thread": None,
        "stop_flag": threading.Event(),
    })


def _run_periodic_task(task: dict, loop: Any):
    """Worker for one periodic task. Sync fns run inline, async fns get
    scheduled on the main loop via run_coroutine_threadsafe."""
    stop = task["stop_flag"]
    interval = max(0.1, task["interval_seconds"])
    fn = task["fn"]
    label = f"plugin.{task['plugin_name']}.{task['name']}"
    log = logging.getLogger(label)
    if not task["run_immediately"]:
        if stop.wait(interval):
            return
    while not stop.is_set():
        try:
            res = fn()
            if asyncio.iscoroutine(res):
                if loop is not None and loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(res, loop)
                    try:
                        # don't block the periodic schedule on a slow task,
                        # but still surface exceptions
                        fut.result(timeout=interval * 2)
                    except Exception as e:
                        log.error(f"async periodic task failed: {e}")
                else:
                    try:
                        asyncio.run(res)
                    except Exception as e:
                        log.error(f"async periodic task failed: {e}")
        except Exception as e:
            log.error(f"periodic task crashed: {e}", exc_info=True)
        if stop.wait(interval):
            return


def start_periodic_tasks(loop: Any = None):
    """Spawn a daemon thread for each registered periodic task. Idempotent,
    safe to call once at startup. The loader calls this after the
    'startup' event fires."""
    global _periodic_started
    if _periodic_started:
        return
    _periodic_started = True
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
    for task in _periodic_tasks:
        if task["thread"] is not None:
            continue
        t = threading.Thread(
            target=_run_periodic_task,
            args=(task, loop),
            daemon=True,
            name=f"plugin-periodic-{task['plugin_name']}-{task['name']}",
        )
        task["thread"] = t
        t.start()
        logger.debug(
            f"started periodic task {task['plugin_name']}.{task['name']} "
            f"every {task['interval_seconds']}s"
        )


def stop_periodic_tasks(timeout: float = 2.0):
    """Signal every periodic task to stop and wait briefly for them.
    Called from PluginManager.teardown_all()."""
    for task in _periodic_tasks:
        task["stop_flag"].set()
    deadline = time.time() + max(0.0, timeout)
    for task in _periodic_tasks:
        t = task.get("thread")
        if t is None:
            continue
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            t.join(timeout=remaining)
        except Exception:
            pass


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
        # Keep the real config private for internal plugin_config() lookups
        # (a plugin needs to read its OWN scoped subdict, even if it
        # contains things like api keys for third-party services). What
        # we expose publicly is a sandboxed SafeConfigView so plugins
        # can't read gemini's api key, vrchat creds, mongo strings, etc.
        self._real_config = config
        self.config = SafeConfigView(config, plugin_name) if config is not None else None
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
        `plugins.<name>.<key>`. Pass no key to get the whole sub-dict.

        This bypasses the SafeConfigView denylist on purpose: a plugin
        is allowed to put api keys or tokens inside its OWN scoped
        config block (e.g. plugins.voiceid.openai_api_key)."""
        if self._real_config is None:
            return default if key is not None else {}
        if key is None:
            return self._real_config.get("plugins", self.plugin_name, default={}) or {}
        return self._real_config.get("plugins", self.plugin_name, key, default=default)

    def register_periodic_task(
        self,
        name: str,
        interval_seconds: float,
        fn: Callable[[], Any],
        run_immediately: bool = False,
    ):
        """Run `fn` every `interval_seconds`. Use this instead of
        spinning up your own thread, so the host can manage lifecycle
        and shut it down cleanly on teardown.

        `fn` may be sync or async (async coroutines are scheduled on
        the main event loop). Keep the work short, don't block forever.
        Set `run_immediately=True` to also fire once at startup before
        the first interval."""
        register_periodic_task(self.plugin_name, name, interval_seconds, fn, run_immediately)

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

    async def send_realtime_text(self, text: str) -> bool:
        """Inject a short text annotation into the CURRENT incoming
        user turn without ending it. Use this for things like speaker
        labels, vision tags, sensor readings, anything you want the
        model to see as part of whatever audio is streaming right now.

        Works on both gemini 3.1 (uses send_realtime_input(text=...))
        and gemini 2.5 (uses send_client_content with
        turn_complete=False). Does NOT wait for the model to stop
        speaking, so don't spam it, the model might mix annotations
        across turns if you call it too often. Throttle in your plugin.

        Returns False if the session is not up or sending failed.
        Plugin api v3+.
        """
        session = self._app.get("session")
        if session is None or not hasattr(session, "send_inline_text"):
            self.logger.debug("send_realtime_text: session not ready")
            return False
        try:
            return await session.send_inline_text(text)
        except Exception as e:
            self.logger.error(f"send_realtime_text failed: {e}")
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
