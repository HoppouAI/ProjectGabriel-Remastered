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
   api_version: 1
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

`source` must implement two methods:

- `is_active() -> bool` -- True while the source wants screen time. Also
  used by the host to mark itself busy and suppress the idle banner.
- `render() -> str | None` -- the chatbox text (max 144 chars), or None
  to skip this tick.

Built in displays sit at priority 10 (local music) and 20 (lyria).
Plugins default to 100 so they yield to host displays unless they ask
for less.

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
