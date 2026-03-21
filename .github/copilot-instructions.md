# ProjectGabriel -- Copilot Instructions

> **Owner:** HoppouAI  
> **Repo:** ProjectGabriel-Remaster

## Overview

ProjectGabriel is a real-time VRChat AI powered by **Gemini Live** (WebSocket audio streaming). It listens to people in VRChat, responds with voice, and controls VRChat via OSC. It includes person-following via YOLOv8 computer vision and face tracking via YOLOv8-face.

## Architecture

```
main.py                  -- Entry point, wires everything together
supervisor.py            -- Process supervisor, restarts main.py on crash
control_server.py        -- FastAPI WebUI server (dashboard + memory manager)
vision_server.py         -- Debug vision WebUI for YOLO detections
src/
  cli.py                 -- CLI formatting (colored logging, startup banner/info)
  config.py              -- YAML config loader + API key rotation
  gemini_live.py         -- Gemini Live session (send/receive audio, tool dispatch, transcription, thinking)
  audio.py               -- PyAudio I/O, voice boost/distortion, pygame music/SFX playback
  vrchat.py              -- VRChat OSC client (chatbox, voice, movement, grab/drop/use, smooth look)
  vrchatapi.py           -- VRChat REST API client (auth, avatar switch, friends, status, invites)
  tracker.py             -- YOLOv8 person detection + OSC movement control (bettercam)
  face_tracker.py        -- YOLOv8-face face detection + smooth face-tracking via OSC (mss)
  personalities.py       -- Personality switching system (list/switch/get via tools)
  avatars.py             -- VRCX avatar search API integration
  instance_monitor.py    -- VRChat instance join/leave monitoring via API polling
  wanderer.py            -- Random autonomous wandering behavior via OSC movement
  idle_chatbox.py        -- Idle chatbox banner display in VRChat chatbox
  tts.py                 -- Google Cloud TTS integration for voice synthesis
  myinstants.py          -- MyInstants.com sound search & download
  memory.py              -- Persistent memory system (MongoDB Atlas / SQLite)
  emotions.py            -- Avatar emotion/animation system via OSC
  tools/                 -- Gemini function tools (modular package)
    __init__.py          -- Re-exports get_tool_declarations, ToolHandler
    _base.py             -- BaseTool abstract class + @register_tool decorator
    _handler.py          -- ToolHandler dispatcher (routes calls to tools)
    soundboard.py        -- SFX playback tool
    music.py             -- Music playback tools
    voice.py             -- Voice/boost controls
    personalities.py     -- Personality switching
    movement.py          -- OSC movement tools
    tracker.py           -- Player tracker tools
    wanderer.py          -- Wanderer controls
    vrchat_api.py        -- VRChat API tools (avatar, status, friends, worlds)
    system.py            -- System info tools
    memory_tools.py      -- Memory tools (saveMemory, searchMemories, deleteMemory, listMemories, recallMemories)
    emotions_tools.py    -- Emotion/animation tools
    discord.py           -- Discord messaging tools for main session
discord_bot/             -- Discord selfbot module (separate Gemini Live session)
  __init__.py            -- Package init
  bot.py                 -- Discord client, message handling, admin commands
  config.py              -- Bot-specific YAML config loader
  config.yml             -- Bot config (gitignored, see .example)
  config.yml.example     -- Template bot config
  gemini_session.py      -- Gemini Live AUDIO session (transcription captured for text)
  conversation_store.py  -- Per-channel JSON conversation persistence
  prompts/               -- Discord-specific prompt files (separate from main AI)
    prompts.yml          -- Named system prompts (gitignored, see .example)
    prompts.yml.example  -- Template prompts
    appends.yml          -- Auto-appended instructions (gitignored, see .example)
    appends.yml.example  -- Template appends
    personalities.yml    -- Switchable personalities (gitignored, see .example)
    personalities.yml.example -- Template personalities
  tools/                 -- Bot-specific tool modules
    __init__.py          -- Exports DiscordToolHandler
    handler.py           -- Tool dispatcher for bot session
    memory.py            -- Memory tools (shared with main system, discord_ prefix)
    relay.py             -- Relay messages to main VRChat session
    discord_actions.py   -- Send messages, reactions, set status
    personalities.py     -- Personality switching tools
    system.py            -- System info tools
  data/                  -- Bot runtime data (gitignored)
    conversations/       -- Per-channel conversation JSON logs
    session_handle.txt   -- Gemini session handle for resumption
config/
  voices.yml             -- Voice configuration (gitignored, see .example)
  voices.yml.example     -- Template voice config
  prompts/
    prompts.yml          -- Named system prompts (gitignored, see .example)
    appends.yml          -- Auto-appended context (gitignored, see .example)
    personalities.yml    -- Switchable personality modes (gitignored, see .example)
    *.yml.example        -- Template files for new users
config.yml               -- Main config (gitignored, see config.yml.example)
config.yml.example       -- Template config with placeholder values
webui/                   -- Dashboard + memory manager HTML/JS/CSS
models/yolov8/           -- YOLOv8n model (auto-downloaded) + config.json
sfx/music/               -- Local music files for playback
data/conversations/      -- Auto-saved conversation transcripts (JSON)
```

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

### Thinking
- Configurable via `gemini.thinking.budget` and `gemini.thinking.include_thoughts`
- Thought summaries displayed in WebUI console and VRChat chatbox ("Thinking...")
- `types.ThinkingConfig` wired in `gemini_live.py :: _build_config()`

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

## Development Notes

- Keep comments minimal to save context window
- Don't use em dashes in the code or commits
- Config changes go in `config.yml` -- add matching properties to `Config` class
- All async code uses `asyncio` -- blocking calls wrapped with `asyncio.to_thread()`
- PyAudio requires system-level dependencies (PortAudio)
- For VRChat: user needs a virtual audio cable to route AI output to VRChat mic input
- Sensitive files (config.yml, prompt YMLs) are gitignored -- only .example files tracked
- Commit every meaningful change as a separate, logical commit -- group related changes per feature, keep commits realistic and focused
