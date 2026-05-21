# ProjectGabriel -- Copilot Instructions

> **Owner:** HoppouAI  
> **Repo:** ProjectGabriel-Remaster

## Overview

ProjectGabriel is a real-time VRChat AI powered by **Gemini Live** (WebSocket audio streaming). It listens to people in VRChat, responds with voice, and controls VRChat via OSC. It includes person-following via YOLOv8 computer vision and face tracking via YOLOv8-face.

## Development Notes

- Do not make comments unless they are needed; your code should be self-explanatory. If the logic is complex, write a concise comment. Avoid obvious comments that restate what the code does.
- Don't use em dashes in the code or commits
- Config changes go in `config.yml` -- add matching properties to `Config` class
- All async code uses `asyncio` -- blocking calls wrapped with `asyncio.to_thread()`
- PyAudio requires system-level dependencies (PortAudio)
- For VRChat: user needs a virtual audio cable to route AI output to VRChat mic input
- Sensitive files (config.yml, prompt YMLs) are gitignored -- only .example files tracked
- Commit every meaningful change as a separate, logical commit -- group related changes per feature, keep commits realistic and focused

## SDK Rules

- **Always** use `google-genai` (`from google import genai`). Never use `google-generativeai`.
- Client is created per-session: `genai.Client(api_key=...)`.
- Live API: `client.aio.live.connect(model=..., config=...)`.
- Tool responses in Live API are manual -- execute function, send `FunctionResponse` back.
- Thinking: `types.ThinkingConfig(thinking_budget=..., include_thoughts=...)` in `LiveConnectConfig`.
- Context window compression: `types.ContextWindowCompressionConfig` with `SlidingWindow` mechanism.
- Session resumption: Handle persisted to `session_handle.txt`, 2-hour expiry.

## Key Patterns

### API Key Rotation
Keys are defined in `config.yml` (primary + backup list). On 429/quota errors, `Config.rotate_key()` cycles to the next key and the session reconnects automatically.

### Audio Pipeline
1. Mic -> PyAudio input -> Gemini Live (raw PCM 16kHz mono)
2. Gemini Live -> audio output -> `AudioManager.process_output_audio()` (applies boost/distortion) -> PyAudio output
3. Music/SFX -> pygame.mixer -> system audio output

### VRChat OSC
- Chatbox: `/chatbox/input [text, True, False]` -- immediate, no sound
- Typing indicator: `/chatbox/typing [bool]` -- on while model is speaking
- Voice toggle: `/input/Voice` -- press 1 then 0
- Movement: `/input/MoveForward`, `/input/LookHorizontal`, `/input/Run` (sprint)
- Grab/Drop: `/input/GrabRight` (1=grab, 0=drop)
- Use: `/input/UseRight` (momentary 1->0 click)
- Smooth look: EMA-based turning via `/input/LookHorizontal` with ramp up/down
- Crouch/Crawl: pynput keyboard simulation (C/Z keys)
- 144-char chatbox limit with automatic pagination `(1/N)` format

### Adding a New Tool
1. Create a file in `src/tools/` (or add to an existing one)
2. Create a class decorated with `@register_tool` extending `BaseTool`
3. Override `declarations(self, config=None)` returning a list of `types.FunctionDeclaration`
4. Override `handle(self, name, args)` -- return a dict result or `None` if not handled
5. Import the module in `_handler.py :: ToolHandler.__init__()` to trigger registration
6. Include `\n**Invocation Condition:**` in each tool description
7. Simple results: `{"result": "ok"}` -- complex results return relevant data
8. Access handler state via `self.handler` (audio, osc, tracker, config, etc.)

Example tool file:
```python
from google.genai import types
from src.tools._base import BaseTool, register_tool

@register_tool
class MyTool(BaseTool):
    def declarations(self, config=None):
        return [types.FunctionDeclaration(
            name="myTool",
            description="Does something.\n**Invocation Condition:** Call when ...",
            parameters={"type": "OBJECT", "properties": {
                "arg1": {"type": "STRING", "description": "..."},
            }, "required": ["arg1"]},
        )]

    async def handle(self, name, args):
        if name == "myTool":
            return {"result": "ok", "data": "..."}
        return None
```

### Plugin System
- Drop-in plugin folder at `plugins/<name>/` with a `plugin.yml` manifest and an `__init__.py`.
- Plugin classes subclass `src.plugins.Plugin` and implement `setup(ctx)` / `teardown(ctx)`.
- `PluginContext` exposes: `register_tool(cls)`, `register_tts(name, factory)`, `register_stt(name, factory)`, `register_chatbox_source(name, source, priority)`, `register_prompt_contributor(name, fn)`, `subscribe(event, cb)`, `send_system_instruction(text)` / `send_user_text(text)` (mid-session injection, waits for model to stop speaking, same path as the WebUI), `plugin_config(key)`, `data_dir()`, lazy `audio` / `osc` / `session` / `tool_handler` refs.
- `ctx.discord` sub-context (API v2+) extends the same hooks to the Discord bot's separate Gemini Live session: `ctx.discord.register_tool(cls)`, `ctx.discord.register_prompt_contributor(name, fn)`, `ctx.discord.subscribe(event, cb)`, `await ctx.discord.send_system_instruction(text)`, `await ctx.discord.send_user_text(text)`. Discord events: `bot_ready(client)`, `dm_received(message)`, `mention_received(message)`, `message_sent(channel_id, text)`. Returns `False` from send_* when bot is offline. Main and Discord registries are independent so plugins can register on both safely.
- Chatbox lifecycle is centralized in `src/gemini_live/chatbox_orchestrator.py :: ChatboxOrchestrator`. Builtins (local music, music_gen) always beat plugin sources. Lower priority value wins inside each tier. Dedupes identical text, force-refreshes after ~6s for VRChat keepalive, calls optional `source.on_clear()` when winner changes or source goes inactive, suspends a source after 5 consecutive `is_active`/`render` exceptions and logs via `record_plugin_issue`.
- Loaded BEFORE `GeminiLiveSession` is constructed so `@register_tool` fires before `ToolHandler` reads the registry.
- Built-in events: `startup`, `shutdown`, `message_in(text, source)`, `message_out(text)`. Sync or async handlers, exceptions caught per subscriber.
- Plugin TTS providers picked up via `tts.external_provider: <name>` in `config.yml` (only used when no built-in TTS is enabled).
- Per-plugin runtime config under `plugins.<name>.*` in `config.yml`. Whether a plugin LOADS is set by `enabled:` inside that plugin's own `plugins/<name>/plugin.yml`. Per-tool toggles live in `config/tools.yml` under `plugin_tools.<plugin>.<tool_name>` (auto-populated on startup by `src/tools_sync.py`). Master toggle for the whole plugin loader is `plugins.enabled` in `config.yml`.
- Per-plugin runtime data lives under `data/plugins/<name>/` (gitignored).
- Missing python deps in `plugin.yml :: requirements:` are warned, never auto-installed.
- Plugins live in a separate repo: [HoppouAI/ProjectGabriel-Plugins](https://github.com/HoppouAI/ProjectGabriel-Plugins) (public) plus a private one for non-shareable backends. Reference plugins there: `example_hello/` (sayHello tool + lifecycle subscribers), `mood/` (persistent emotion+intensity, prompt contributor), `diary/` (background sub-agent + tools). The host's `plugins/` folder is gitignored end-to-end except `plugins/README.md` (the authoring guide). Users install plugins by copying folders from the plugins repo into their `plugins/` dir.
- Plugin loader code: `src/plugins/api.py` (Plugin, PluginContext, registries), `src/plugins/loader.py` (PluginManager).

### Diary Plugin
- `plugins/diary/` -- long term first person diary for the AI, separate from the structured memory system.
- Background `DiaryScheduler` ticks every 2 hours (configurable via `plugins.diary.interval_hours`), runs after a 5 minute warmup.
- Each tick gathers the most recent N session JSON files for today from `data/conversations/` (default 5), passes them to `gemini-3.1-flash-lite-preview` along with any earlier diary entries from today.
- Sub-agent returns strict JSON `{people, mood_arc, body, highlights}`, the plugin wraps it as a `DiaryEntry` and appends to `data/plugins/diary/gabriel.diary` (custom plain text format, lenient parser).
- Multiple entries per day allowed, numbered as "part 1, part 2, ...". Tick is skipped if no new sessions appeared since the last entry.
- Tools: `readDiary(date?, limit?)`, `searchDiary(query, limit?)`, `listDiaryDates()`, `updateDiaryNow()` (force tick).
- Requires `privacy.save_conversations: true` to have anything to summarize.

### Prompt & Personality System
- **prompts.yml**: Named base prompts structured as `**Persona:** -> **Conversational Rules:** -> **General Guidelines:** -> **Guardrails:**`. Select in `config.yml` -> `gemini.prompt`.
- **appends.yml**: Auto-appends organized into 4 sections: (1) Conversational Rules & Identity, (2) Tool Invocation Conditions, (3) Guardrails, (4) Dynamic Context. Supports `{date}`, `{available_personalities}`, `{memories}` placeholders.
- **personalities.yml**: Switchable modes. Each has `name`, `description`, `enabled`, `prompt`. Model calls `list_personalities`, `switch_personality`, `get_current_personality`.
- On `switch_personality`, the prompt is injected via `send_client_content` so the model adopts it immediately.
- All prompt/personality files are gitignored. `.example` templates provided for new users.

### YOLO Person Tracking
- Model: `yolov8n.pt` auto-downloads to `models/yolov8/` on first use
- Config: `models/yolov8/config.json` for thresholds, speed, update interval
- Screen capture via `bettercam`, detection via ultralytics, movement via OSC
- Toggled via `yolo.enabled` in config

### Face Tracking
- Model: `yolov8n-face.pt` auto-downloaded from GitHub (akanametov/yolo-face)
- Two modes: Speaking (locks on closest face), Idle (random glances every 5-10s)
- Smooth EMA turning at 30 FPS, yields to player tracker and wanderer when active
- Screen capture via `mss` (avoids bettercam conflict with person tracker)
- Lazy-imported -- when `face_tracker.enabled: false`, module and deps not loaded
- Toggled via `face_tracker.enabled` in config

### Memory System
- Backend: MongoDB Atlas (primary) or SQLite (fallback)
- Types: `long_term` (permanent), `short_term` (7 days), `quick_note` (6 hours)
- Tools: `saveMemory`, `searchMemories`, `deleteMemory`, `listMemories`, `recallMemories`
- Each tool is a separate FunctionDeclaration (avoids Gemini Live 1011 errors from complex schemas)
- Recall sub-agent uses `gemini-3.1-flash-lite-preview` to summarize all memories
- Recent memories injected into system prompt via `{memories}` placeholder
- RAG providers: `gemini` (Gemini embeddings + MongoDB Atlas vector search) or `local` (LM Studio + ChromaDB)
- Local RAG: ChromaDB for vector storage, LM Studio for embeddings (EmbeddingGemma 300M), works with any backend
- ChromaDB auto-syncs existing memories on startup via background thread
- Split thresholds: `vector_min_score_gemini` (default 0.82) and `vector_min_score_local` (default 0.55)
- Legacy `vector_min_score` still works as fallback, overrides whichever provider is active

### Thinking
- Configurable via `gemini.thinking.budget` and `gemini.thinking.include_thoughts`
- Thought summaries displayed in WebUI console and VRChat chatbox ("Thinking...")
- `types.ThinkingConfig` wired in `src/gemini_live/config_builder.py :: ConfigBuilderMixin._build_config()`

### VRChat API
- REST API client in `src/vrchatapi.py` -- base URL `https://api.vrchat.cloud/api/1`
- Basic auth with cookie persistence (`data/vrchat_cookies.json`), auto-TOTP 2FA via pyotp
- Tools: `switchAvatar`, `searchAvatars` (VRCX API), `getOwnAvatar`, `getAvatarInfo`, `updateStatus`, `getCurrentStatus`, `getFriendInfo`, `searchWorlds`, `inviteSelfToInstance`
- statusDescription has a 32-char max limit (API rejects longer values)
- Friends list cached locally, refreshed periodically

### Idle Chatbox
- `src/idle_chatbox.py` -- customizable banner shown in VRChat chatbox when AI is idle
- Displays banner text, dividers, up to 3 configurable lines, active session time, and clock
- Starts when AI enters idle state, stops on thinking/speaking/reconnect
- Both idle chatbox AND idle animation are suppressed during music playback
- Config under `vrchat.idle_chatbox` (enabled, banner, divider, divider_length, lines, update_interval)

### Wanderer
- `src/wanderer.py` -- random autonomous wandering behavior via OSC movement
- When active, face tracker yields control to avoid conflicts
- Toggled via tools, supports pause/resume

### WebUI
- FastAPI control server on port 8766
- Dashboard with console log, controls, and session info
- Memory manager tab for viewing/searching/deleting memories
- Vision debug server on port 8767 (when `yolo.vision_debug: true`)
- OBS overlay system (optional, `obs.enabled: false` by default):
  - `/overlay` -- transparent text overlay (browser source)
  - `/overlay/music` -- standalone music overlay (browser source)
  - `/overlay/config` -- visual configurator with presets
  - When disabled: routes return 404, music broadcast loop doesn't start, turn_complete events not sent

### CLI
- `src/cli.py` -- colored logging formatter, startup banner, config display
- `setup_logging()` replaces `logging.basicConfig()` with compact colored output (`HH:MM:SS LEVEL name  message`)
- Log levels: green INFO, yellow WARN, red ERROR, dim gray DEBUG
- `print_startup_info(config)` shows model/voice/TTS/OSC/music + component status indicators
- Supervisor prints cyan box-drawing banner, passes through main.py output without prefix
- Uses `colorama.just_fix_windows_console()` for Windows ANSI support, `stdout.reconfigure(encoding="utf-8")` for Unicode
- Supervisor sets `PYTHONIOENCODING=utf-8` for subprocess to handle Unicode box-drawing/symbols in piped output

