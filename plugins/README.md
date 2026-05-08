# Plugins

Drop-in plugin folder. Anything in here with a `plugin.yml` and an entry
module that subclasses `Plugin` will be loaded on startup.

This folder is mostly gitignored so personal plugins stay local. The
`example_hello/` folder is tracked as a reference implementation.

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

### `ctx.subscribe(event, callback)`

Hook into app lifecycle. Built in events:

- `startup` -- fires once after the session is up
- `shutdown` -- fires once on graceful shutdown
- `message_in` (`text, source`) -- fires on each transcribed user message
- `message_out` (`text`) -- fires on each AI reply

Callbacks can be sync or async. Exceptions in any single handler are
caught so one bad subscriber will not break the rest.

### `ctx.plugin_config(key=None, default=None)`

Reads this plugin's config from `config.yml`:

```yaml
plugins:
  enabled: true   # global switch for the whole plugin system
  my_thing:
    enabled: true        # per-plugin switch (overrides plugin.yml)
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

## Notes

- Plugins load BEFORE the Gemini Live session is built, so any tools
  they register show up in the very first connect.
- Per-plugin `plugins.<name>.enabled` in `config.yml` overrides the
  manifest.
- Set `plugins.enabled: false` in `config.yml` to disable the whole
  system at once.
- Missing pip dependencies are logged as warnings, the plugin still
  tries to load. If imports fail the plugin is skipped and the host
  keeps running.
- Bump `api_version` in `plugin.yml` when targeting newer host
  capabilities. Current host API version: `1`.
