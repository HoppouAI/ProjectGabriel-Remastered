# ProjectGabriel — Copilot Instructions

> **Owner:** HoppouAI  
> **Repo:** ProjectGabriel-Remaster

## Overview

ProjectGabriel is a real-time VRChat AI powered by **Gemini Live** (WebSocket audio streaming). It listens to people in VRChat, responds with voice, and controls VRChat via OSC. It includes person-following via YOLOv8 computer vision.

## Architecture

```
main.py                  — Entry point, wires everything together
src/
  config.py              — YAML config loader + API key rotation
  gemini_live.py         — Gemini Live session (send/receive audio, tool dispatch, transcription)
  audio.py               — PyAudio I/O, voice boost/distortion, pygame music/SFX playback
  vrchat.py              — VRChat OSC client (chatbox with pagination, voice toggle, movement)
  tools.py               — Function declarations for Gemini + ToolHandler dispatcher
  tracker.py             — YOLOv8 person detection + OSC movement control
  personalities.py       — Personality switching system (list/switch/get via tools)
  myinstants.py          — MyInstants.com sound search & download
config/
  prompts/
    prompts.yml          — Named system prompts (normal, normal_alt, russian_roommate, etc.)
    appends.yml          — Auto-appended context (date, tool reminders, personality list, etc.)
    personalities.yml    — Switchable personality modes (chill, scammer, anime_girl, etc.)
config.yml               — Main config (API keys, audio, OSC, YOLO, VAD settings)
models/yolov8/           — YOLOv8n model (auto-downloaded) + config.json
sfx/music/               — Local music files for playback
```

## SDK Rules

- **Always** use `google-genai` (`from google import genai`). Never use `google-generativeai`.
- Client is created per-session: `genai.Client(api_key=...)`.
- Live API: `client.aio.live.connect(model=..., config=...)`.
- Tool responses in Live API are manual — execute function, send `FunctionResponse` back.

## Key Patterns

### API Key Rotation
Keys are defined in `config.yml` (primary + backup list). On 429/quota errors, `Config.rotate_key()` cycles to the next key and the session reconnects automatically.

### Audio Pipeline
1. Mic → PyAudio input → Gemini Live (raw PCM 16kHz mono)
2. Gemini Live → audio output → `AudioManager.process_output_audio()` (applies boost/distortion) → PyAudio output
3. Music/SFX → pygame.mixer → system audio output

### VRChat OSC
- Chatbox: `/chatbox/input [text, True, False]` — immediate, no sound
- Typing indicator: `/chatbox/typing [bool]` — on while model is speaking
- Voice toggle: `/input/Voice` — press 1 then 0
- Movement: `/input/MoveForward`, `/input/LookHorizontal`
- 144-char chatbox limit with automatic pagination `(1/N)` format

### Adding a New Tool
1. Add `FunctionDeclaration` to the list in `src/tools.py :: get_tool_declarations()`
2. Add handler case in `ToolHandler._dispatch()`
3. Simple results: `{"result": "ok"}` — complex results return relevant data

### Prompt & Personality System
- **prompts.yml**: Named base prompts. Each entry has `name`, `description`, `prompt`. Select in `config.yml` → `gemini.prompt`.
- **appends.yml**: List of auto-appends. Each has `name`, `enabled`, `content`. Supports `{date}` and `{available_personalities}` placeholders.
- **personalities.yml**: Switchable modes. Each has `name`, `description`, `enabled`, `prompt`. Model can call `list_personalities`, `switch_personality`, `get_current_personality`.
- On `switch_personality`, the prompt is injected via `send_client_content` so the model adopts it immediately.

### YOLO Tracking
- Model: `yolov8n.pt` auto-downloads to `models/yolov8/` on first use
- Config: `models/yolov8/config.json` for thresholds, speed, update interval
- Screen capture via `mss`, detection via ultralytics, movement via OSC

## Development Notes

- Keep comments minimal to save context window
- Don't use em dashes in the code.
- Config changes go in `config.yml` — add matching properties to `Config` class
- All async code uses `asyncio` — blocking calls wrapped with `asyncio.to_thread()`
- PyAudio requires system-level dependencies (PortAudio)
- For VRChat: user needs a virtual audio cable to route AI output to VRChat mic input
