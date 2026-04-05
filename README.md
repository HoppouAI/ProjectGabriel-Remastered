# Project Gabriel - Remaster

The 2026 remaster of Project Gabriel by [Hoppou.AI](https://hoppou.ai/). Gabriel is our VRChat AI, the Indian guy in the blue polo shirt. Same concept as the original but way more features, cleaner code, and a lot more stable. He walks around worlds, talks to people, remembers who they are, and has his own personality system.

![Gabriel Picture](https://github.com/HoppouAI/ProjectGabriel-Framework/blob/main/Other%20Stuff/Gabriel_Picture.png?raw=true)

---

## Summary

Python-based system for running a live AI in VRChat. Handles real-time audio streaming through Gemini Live, VRChat OSC integration (movement, chatbox, voice), a REST API client for VRChat, memory, vision, and a Discord bot running its own separate Gemini Live session. Everything runs through a supervisor that auto-restarts on crashes, with a web dashboard for monitoring.

- **Main Entry Point:** `supervisor.py`
- **Key Features:** Gemini Live audio streaming, YOLOv8 person tracking, YOLOv8-face face tracking, OSC control, Discord bot, WebUI dashboard, persistent memory, personality switching, multiple TTS providers

---

## What's New in the Remaster

The original was getting messy and hard to maintain. This version is a full rewrite with a cleaner architecture. Compared to the original:

- Gemini Live native audio (real-time bidirectional streaming)
- YOLOv8 person tracking and YOLOv8-face face tracking (two separate models)
- Discord selfbot with its own Gemini Live session
- FastAPI WebUI dashboard at port 8766 (console output, controls, memory manager)
- Persistent memory system backed by MongoDB Atlas or SQLite
- Switchable personalities (at runtime via tools)
- VRChat REST API client (avatar switching, friend info, world search, status updates)
- Multiple TTS providers (Gemini native, Qwen3 server, Hoppou AI cloud, Google Cloud Chirp 3 HD)
- API key rotation for handling quota limits automatically
- Autonomous wandering behavior
- Emotion and animation system via OSC
- Idle chatbox with configurable banner display
- Session resumption (2 hour session handle persistence)
- Proper context window compression for unlimited session length

---

## Prerequisites

Before setting up, you need the following:

1. **Virtual Audio Cables** - Two separate virtual audio lines to route audio to and from VRChat.
   - [VB-Audio Cable](https://vb-audio.com/Cable/) (Standard)
   - [VB-Audio Hi-Fi Cable](https://vb-audio.com/Cable/#DownloadASIOBridge) (Secondary)
2. **Gemini API Key** - Get one from [Google AI Studio](https://aistudio.google.com/apikey).
3. **Python 3.11 or 3.12** - The project requires one of these versions. Personally I use 3.12.11 and it works fine. 3.13+ is not supported.

Optional:
- MongoDB Atlas connection string (for cloud memory storage, falls back to SQLite if not set)
- Google Cloud credentials (for Chirp 3 HD TTS)
- VRChat account credentials (for REST API features like avatar switching)

---

## Installation

### Easy (Recommended)

Just run `setup.bat` in the project root. It will:

- Download UV (the package manager) into a local `.uv` folder
- Create a Python 3.12 virtual environment
- Install all dependencies
- Detect if you have an NVIDIA GPU and ask if you want CUDA PyTorch
- Copy all the example config files for you

Once it finishes you just need to fill in your API key in `config.yml`.

### Manual Setup

We recommend using **uv** for this.

**Install uv:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal, then run these in the project folder:

```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

**Standard pip (if you prefer):**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**GPU support (NVIDIA):**

If you have an NVIDIA GPU, replace the default torch install with the CUDA version for better vision performance:

```bash
# Using uv
uv pip uninstall torch torchvision torchaudio
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

# Using pip
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## Configuration

### 1. Main Config

Copy the example config and fill in your values:

```bash
copy config.yml.example config.yml
```

Open `config.yml` and at minimum set your Gemini API key:

```yaml
gemini:
  api_key: "YOUR_GEMINI_API_KEY_HERE"
```

The config file has comments explaining every option. Most defaults are fine to leave as-is.

### 2. Prompts and Personality

Copy the example prompt files in `config/prompts/`:

```bash
copy config\prompts\prompts.yml.example config\prompts\prompts.yml
copy config\prompts\appends.yml.example config\prompts\appends.yml
copy config\prompts\personalities.yml.example config\prompts\personalities.yml
```

Edit `prompts.yml` to define the AI's base persona, `appends.yml` for any extra context appended every session, and `personalities.yml` for switchable personality modes the AI can activate at runtime.

### 3. Voices

```bash
copy config\voices.yml.example config\voices.yml
```

Edit `voices.yml` to configure the voice effect chain (boost, distortion, etc.).

### 4. Performance Tuning

If you are on a lower-end machine or don't have a GPU, disable the YOLO trackers in `config.yml`:

```yaml
yolo:
  enabled: false

face_tracker:
  enabled: false
```

---

## Audio Routing

For the AI to speak in VRChat, you need to route audio correctly. You must run the app first (`python supervisor.py`) so it shows up in the Windows Volume Mixer.

### Windows Volume Mixer

| Application | Output | Input |
| :--- | :--- | :--- |
| Python | CABLE Input (VB-Audio Virtual Cable) | Hi-Fi Cable Output (VB-Audio Hi-Fi) |
| VRChat | Hi-Fi Cable Input (VB-Audio Hi-Fi) | Default / Microphone |

### VRChat In-Game Settings

Go to Settings -> Audio -> Microphone:

1. **Microphone Device:** `CABLE Output` (VB-Audio Virtual Cable)
2. **Noise Suppression:** OFF
3. **Activation Threshold:** 0%
4. **Volume:** Mute Music/SFX, keep Voices at 100%

---

## Usage

Start the app by running the supervisor:

```bash
python supervisor.py
```

The supervisor manages the main process and will automatically restart it if it crashes. To stop everything press `CTRL+C`.

The WebUI dashboard is available at `http://localhost:8766` once running. It shows the console output, lets you manage memories, and has some basic controls.

---

## Discord Bot

> **Disclaimer:** The Discord bot module uses a selfbot (a user account token, not a bot token). Self-botting is against Discord's Terms of Service and your account could be banned. Use this at your own risk. We are not responsible for any action taken against your account.

The Discord selfbot is a separate module in `discord_bot/`. It runs its own Gemini Live session and can send and receive messages in Discord channels.

To configure it:
```bash
copy discord_bot\config.yml.example discord_bot\config.yml
```

Fill in the bot token and other settings, then it will start automatically with the main app if enabled in `config.yml`.

---

## Project Structure

```
main.py              -- Core application logic
supervisor.py        -- Process supervisor (auto-restart on crash)
control_server.py    -- FastAPI WebUI (dashboard + memory manager)
src/
  gemini_live.py     -- Gemini Live session (audio streaming, tool dispatch)
  audio.py           -- Audio I/O, effects, music/SFX playback
  vrchat.py          -- VRChat OSC client
  vrchatapi.py       -- VRChat REST API client
  tracker.py         -- YOLOv8 person tracking
  face_tracker.py    -- YOLOv8-face face tracking
  memory.py          -- Persistent memory (MongoDB / SQLite)
  personalities.py   -- Personality switching
  tools/             -- Gemini function tool modules
discord_bot/         -- Discord selfbot (separate Gemini Live session)
config/
  voices.yml         -- Voice configuration
  prompts/           -- System prompts, appends, personalities (YAML)
webui/               -- Dashboard HTML/JS/CSS
```

---

## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for details.

Additional terms under AGPL Section 7 apply to the Gabriel AI persona. See [NOTICE.md](NOTICE.md).
