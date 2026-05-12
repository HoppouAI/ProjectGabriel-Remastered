# Plugins

Drop-in plugin folder. Anything in here with a `plugin.yml` and an entry
module that subclasses `Plugin` will be loaded on startup.

This folder is mostly gitignored so personal plugins stay local. The
`example_hello/` folder is tracked as a reference implementation. It
ships with `enabled: false` so a fresh checkout doesn't load it. flip
that to `true` if you want to see it in action.

## Quick start

1. Make a folder: `plugins/my_thing/`
2. Add `plugins/my_thing/plugin.yml`:

   ```yaml
   name: my_thing
   version: 0.1.0
   api_version: 2
   author: HoppouAI
   description: does a cool thing
   enabled: true
   # optional: pip dependencies. The loader only warns when one is missing,
   # it never installs anything for you.
   requirements:
     - requests>=2.28
   ```

3. Add `plugins/my_thing/__init__.py`:

   ```python
   from src.plugins import Plugin, PluginContext

   class MyThing(Plugin):
       def setup(self, ctx: PluginContext):
           ctx.logger.info("hello from my_thing")

       def teardown(self, ctx: PluginContext):
           pass
   ```

Restart the app and the log will show `loaded plugin 'my_thing' v0.1.0`.

## What plugins can do

Inside `setup(ctx)` you have a `PluginContext` (`ctx`) with these methods.

### `ctx.register_tool(ToolClass)`

Adds a Gemini function-calling tool. Tool classes extend
`src.tools._base.BaseTool`, the same base class the built in tools use.
See `plugins/example_hello/__init__.py` for a working example, or any
module under `src/tools/` for more.

```python
from google.genai import types
from src.tools._base import BaseTool

class MyTool(BaseTool):
    tool_key = "my_thing"

    def declarations(self, config=None):
        return [types.FunctionDeclaration(
            name="doMyThing",
            description="Does the thing.\n**Invocation Condition:** when asked",
            parameters={"type": "OBJECT", "properties": {}},
        )]

    async def handle(self, name, args):
        if name == "doMyThing":
            return {"result": "ok"}
        return None

# in setup():
ctx.register_tool(MyTool)
```

Note: do NOT decorate plugin tool classes with `@register_tool`. That
decorator is for built in tools that auto-register at import time.
Plugins register through `ctx.register_tool` so they show up under the
plugin's owner in `config/tools.yml` and don't double register.

### `ctx.register_tts(name, factory)`

Add a TTS provider. `factory(config)` should return an object with the
same interface as the built in providers in `src/tts.py` (typically
`start()`, `stop()`, plus whatever the Gemini Live session expects).
Once registered, set `tts.external_provider: <name>` in `config.yml` to
make it the active provider.

```python
def make_my_tts(config):
    return MyTTSProvider(config)

ctx.register_tts("my_tts", make_my_tts)
```

### `ctx.register_stt(name, factory)`

Add an STT provider. The main VRChat loop uses Gemini Live for STT
natively, so this hook is mainly for plugins that want their own
pipeline (for example local Whisper inside a Discord plugin). Factory
shape matches TTS.

### `ctx.register_chatbox_source(name, source, priority=100)`

Contribute a VRChat chatbox display, like a now-playing screen or a
status banner. The host iterates registered sources in ascending
priority order and shows the first active one when no built in display
(local music, lyria) is up.

`source` must implement two methods, plus one optional method:

- `is_active() -> bool` -- True while the source wants screen time. Also
  used by the host to mark itself busy and suppress the idle banner.
- `render() -> str | None` -- the chatbox text (max 144 chars), or None
  to skip this tick.
- `on_clear()` (optional) -- fires once when this source loses the
  chatbox to a different winner OR transitions to inactive with
  nothing to take over. Useful for closing UI state. Safe to omit.

Built in displays sit at priority 10 (local music) and 20 (lyria).
Plugins default to 100 so they yield to host displays unless they ask
for less.

Lifecycle guarantees the orchestrator gives you (host API v2+):

- Same text isnt re-sent every tick. The host dedupes and only resends
  when text changes or after a force-refresh interval (about 6 sec)
  to keep the chatbox alive.
- When you go from `is_active() == True` to `False`, the host stops
  writing your text immediately and lets the idle banner take over.
  No stale text lingers.
- If your `is_active()` or `render()` raises 5 times in a row the
  host suspends your source for the rest of the run and logs a
  warning. Other sources keep working.
- A source returning `None` from `render()` while still active falls
  through to the next source instead of blanking the chatbox.

```python
class MyStatusSource:
    def is_active(self):
        return self.mgr.has_pending_alert
    def render(self):
        return f"\u26a0 {self.mgr.alert_text[:140]}"

ctx.register_chatbox_source("my_status", MyStatusSource(...), priority=80)
```

### `ctx.register_prompt_contributor(name, fn)`

Inject extra text into the system prompt every time it gets built
(session start, reconnect, personality switch). `fn()` should return a
string to append, or None / empty string to skip this build. Errors are
caught so a broken contributor cannot kill the prompt.

Use it for dynamic context the model needs in its system prompt -
current mood, current world status, time-of-day flavor, recent activity
notes, etc. The contributor's text is appended after all built in
appends.

```python
def my_status_block():
    if not weather_known():
        return None
    return f"**Weather:** It's currently {weather()} outside, factor that into your mood."

ctx.register_prompt_contributor("weather", my_status_block)
```

### `ctx.subscribe(event, callback)`

Hook into app lifecycle. Built in events:

- `startup` -- fires once after the session is up
- `shutdown` -- fires once on graceful shutdown
- `message_in` (`text, source`) -- fires on each transcribed user message
- `message_out` (`text`) -- fires on each AI reply

Callbacks can be sync or async. Exceptions in any single handler are
caught so one bad subscriber will not break the rest.

### `await ctx.send_system_instruction(text)` / `await ctx.send_user_text(text)`

Inject text into the live Gemini session mid-conversation. Same code
path the WebUI uses.

- `send_system_instruction(text)` wraps the text as
  `System instruction update - <text>` and pushes it as a user-role
  client content turn. The host waits up to 30 seconds for the model
  to stop speaking before injecting so it doesnt cut off a reply.
  Use this for runtime behavior changes ("stop using emojis", "go
  back to default voice", "the user is afk now").
- `send_user_text(text)` injects a normal user-style text message.
  The model responds like any other user turn. Use this for proxying
  external messages (relay from Discord, voice command, etc).

Both return `True` on success, `False` if the live session isn't up
yet or sending failed. Don't call these inside `setup()`, the session
is None there. Safe to call from a tool handler, an event subscriber,
or any time after `startup` fires.

```python
async def on_user_msg(text, source):
    if "shut up" in text.lower():
        await ctx.send_system_instruction("Stop talking until further notice.")

ctx.subscribe("message_in", on_user_msg)
```

### `ctx.discord` -- Discord bot integration

The Discord selfbot module runs its own Gemini Live session, separate
from the VRChat session. To extend that session use `ctx.discord.*`.
The host main session hooks (`ctx.register_tool`,
`ctx.register_prompt_contributor`, `ctx.subscribe`,
`ctx.send_system_instruction`, `ctx.send_user_text`) are unchanged
and still target VRChat only -- nothing about your existing plugins
breaks.

Plugins can register on both sides safely. They don't share state, so
a tool registered on both ends gets two separate instances.

Available calls under `ctx.discord`:

- `register_tool(tool_cls)` -- attach a tool to the Discord bot's
  tool handler. Same shape as the bot's built-in tools in
  `discord_bot/tools/`. Tool class needs `__init__(self, handler)`,
  `declarations(self) -> list[FunctionDeclaration]`, and
  `async handle(self, name, args)`.
- `register_prompt_contributor(name, fn)` -- append text to the
  Discord bot's system prompt. Called every time the bot rebuilds
  its prompt. Errors are swallowed.
- `subscribe(event, callback)` -- subscribe to Discord-scoped events:
  - `bot_ready` (`client`) -- fires when the discord client connects
  - `dm_received` (`message`) -- raw `discord.Message` for a DM
  - `mention_received` (`message`) -- raw `discord.Message` for an @
  - `message_sent` (`channel_id, text`) -- the bot replied
- `await send_system_instruction(text)` -- inject a SYSTEM
  INSTRUCTION style turn into the Discord session.
- `await send_user_text(text)` -- inject a user-style text turn.
- `session` / `tool_handler` -- properties returning the live Discord
  Gemini session and tool handler, or None if the bot is offline.

Safe to call all of these from `setup()`. If the Discord bot is
disabled in config the registrations are kept and simply never used.
`send_*` returns `False` while the bot is offline.

```python
class DiaryPlugin(Plugin):
    name = "diary"

    def setup(self, ctx):
        # Same tools work on both sides
        ctx.register_tool(DiaryTool)
        ctx.discord.register_tool(DiaryTool)

        # Same prompt context goes into both prompts
        ctx.register_prompt_contributor("diary_today", self._today_summary)
        ctx.discord.register_prompt_contributor("diary_today", self._today_summary)

        # React to incoming Discord DMs
        ctx.discord.subscribe("dm_received", self._on_dm)

    async def _on_dm(self, message):
        if "remember this" in message.content.lower():
            await self.ctx.discord.send_system_instruction(
                "User just asked you to remember the last DM verbatim."
            )
```

### `ctx.plugin_config(key=None, default=None)`

Reads this plugin's runtime settings from `config.yml`. Note: this
only holds runtime knobs (api keys, urls, thresholds, etc). Whether
the plugin loads at all is governed by `enabled:` in your
`plugin.yml`, not here.

```yaml
plugins:
  enabled: true   # global switch for the whole plugin system
  my_thing:
    api_key: "abc123"
    threshold: 0.5
```

```python
api_key = ctx.plugin_config("api_key")
threshold = ctx.plugin_config("threshold", default=0.5)
all_my_config = ctx.plugin_config()  # returns the whole sub-dict
```

### `ctx.data_dir()`

Returns a per-plugin data dir at `data/plugins/<name>/`, created on
demand. Use it for any local state the plugin wants to persist.

### `ctx.audio` / `ctx.osc` / `ctx.session` / `ctx.tool_handler`

References to the live app objects. These are `None` during `setup()`
because the rest of the app may not be wired yet. Read them lazily
(when a tool actually runs) or grab them inside a `startup` event
handler.

## Enable / disable model

There are three layers:

1. **Master switch** -- `plugins.enabled` in `config.yml`. Set to
   `false` to skip the entire plugin loader. Master kill switch.
2. **Per-plugin enable** -- `enabled:` inside the plugin's own
   `plugins/<name>/plugin.yml`. This is the "should this plugin load
   at all" flag. If `false` the plugin is never imported and none of
   its tools register.
3. **Per-tool toggles** -- `config/tools.yml` under
   `plugin_tools.<plugin>.<tool_name>`. Auto-populated on every
   startup by `src/tools_sync.py`. Set a tool to `false` and its
   `FunctionDeclaration` is filtered out of the schema sent to gemini
   on connect, so the model has no idea that tool exists. The plugin
   stays loaded and the rest of its tools keep working.

So to fully shut a plugin down you flip `enabled: false` in its
`plugin.yml`. To just hide one of its tools from the model you flip
that tool to `false` in `config/tools.yml`.

The legacy `plugins.<name>.enabled` key in `config.yml` still works as
a fallback override for upgraders, but new plugins should use the
manifest field.

## How tools.yml works

Every startup the host walks the live `@register_tool` registry plus
every plugin's registered tools and writes any newly discovered name
into `config/tools.yml`, defaulting to `true`. Existing values are
never overwritten so anything you flip off stays off across upgrades.
The schema is:

```yaml
tools:                 # built-in tools shipped with the host
  <tool_name>: bool
plugin_tools:          # tools added by modular plugins, grouped per plugin
  <plugin_name>:
    <tool_name>: bool
```

Disabled tools are filtered out of the gemini schema AND skipped at
handler instantiation time, so they cost zero memory and the model
cannot call them.

## Notes

- Plugins load BEFORE the Gemini Live session is built and BEFORE
  `tools_sync.sync_tools_yml()` runs, so any tools they register show
  up in `config/tools.yml` automatically and are present in the very
  first connect.
- Missing pip dependencies are logged as warnings, the plugin still
  tries to load. If imports fail the plugin is skipped and the host
  keeps running.
- Bump `api_version` in `plugin.yml` when targeting newer host
  capabilities. Current host API version: `1`.
