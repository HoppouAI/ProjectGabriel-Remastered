<p align="center">
  <picture>
    <img alt="Project Gabriel" src="https://hoppou.ai/images/projects/ProjectCardHoppouAI-GabrielRemaster.webp" width="600">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/HoppouAI/ProjectGabriel-Remastered/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/HoppouAI/ProjectGabriel-Remastered?style=flat-square&color=6366f1"></a>
  <a href="https://github.com/HoppouAI/ProjectGabriel-Remastered/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-6366f1?style=flat-square"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11_|_3.12-6366f1?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://discord.gg/ZNWTYTk4Vq"><img alt="Discord" src="https://img.shields.io/badge/discord-join-6366f1?style=flat-square&logo=discord&logoColor=white"></a>
</p>

# Project Gabriel

A real-time AI companion built for VRChat. Gabriel walks around, talks to people, remembers who they are, and has his own personality. He listens through Gemini Live's native audio streaming, sees through computer vision, and controls his avatar through OSC. Built by [Hoppou AI](https://hoppou.ai).

**VRChat is completely optional.** Gabriel runs just fine without it: you can talk to him through Discord, the WebUI, or any other interface. He was designed around VRChat and that's where he shines brightest, but nothing locks you into it. The Discord bot, social server, and local backend all work standalone.

Most of the well known VRChat AI companions out there are either closed source, locked behind a paywall, or both. Gabriel breaks that pattern. He is free, open source, and designed so anyone with a Gemini API key and a Windows machine can have their own AI companion without paying a cent or signing up for someone else's service. That's the whole point.

---

## Features

<table>
  <tr>
    <td>🎙️</td>
    <td><strong>Gemini Live audio</strong><br>Real-time voice conversations powered by Gemini. Gabriel talks naturally with one of eight built-in voices, no robotic TTS middleware needed.</td>
    <td>👁️</td>
    <td><strong>Computer vision</strong><br>Sees the game world through screen capture. Finds people in the room, and can follow them around.</td>
  </tr>
  <tr>
    <td>🎮</td>
    <td><strong>Full OSC control</strong><br>Walks, turns, jumps, crouches, grabs objects, and types into the chatbox. Everything you can do with a keyboard and mouse, Gabriel does through VRChat's OSC interface.</td>
    <td>🧠</td>
    <td><strong>Persistent memory</strong><br>Remembers people, places, and conversations across sessions. Stores long term facts, short term context, and quick notes with smart semantic search.</td>
  </tr>
  <tr>
    <td>🎭</td>
    <td><strong>Switchable personalities</strong><br>Define different personas for Gabriel and let him switch between them on the fly. He can change his whole vibe mid-conversation based on context.</td>
    <td>🧭</td>
    <td><strong>Spatial navigation</strong><br>Maps out VRChat worlds and finds his way around. Saves waypoints, explores on his own, and remembers the layout of every world he visits.</td>
  </tr>
  <tr>
    <td>🌐</td>
    <td><strong>WebUI dashboard</strong><br>A clean browser dashboard with live console output, memory management, vision controls, mapping view, waypoints, and OBS overlay support.</td>
    <td>🔌</td>
    <td><strong>Plugin system</strong><br>Extend Gabriel with drop-in plugins. Add new tools, custom voices, chatbox widgets, or hook into events. Official Plugins: https://github.com/HoppouAI/ProjectGabriel-Plugins</td>
  </tr>
  <tr>
    <td>💬</td>
    <td><strong>Discord bot</strong><br>Gabriel hangs out in your Discord server too, with his own Gemini session. Reads messages, replies naturally, and keeps track of each channel separately.</td>
    <td>🔗</td>
    <td><strong>VRChat API</strong><br>Search and switch avatars, look up friends, find worlds, update your status, send friend requests, and invite people to your instance.</td>
  </tr>
  <tr>
    <td>🔑</td>
    <td><strong>API key rotation</strong><br>Add backup Gemini API keys and Gabriel cycles through them automatically when he hits a rate limit. No downtime, no manual intervention.</td>
    <td>🏠</td>
    <td><strong>Local backend</strong><br>Optionally run everything on your own machine with LM Studio, Moonshine speech recognition, and any TTS provider. Nothing leaves your computer.</td>
  </tr>
</table>

---

## Quick Start

1. Download the **latest release** from the [Releases page](https://github.com/HoppouAI/ProjectGabriel-Remastered/releases) and extract it.
2. Open the folder and run `setup.bat`. It downloads Python 3.12, installs everything, and asks about GPU support.
3. When setup finishes, run `run.bat` to start Gabriel.

If Windows is hiding file extensions, the files will just say `setup` and `run` without the `.bat`. Same thing.

The WebUI is at **http://localhost:8766** once running.

---

## Prerequisites

- **Two virtual audio cables** for routing audio through VRChat. [VB-Audio Cable](https://vb-audio.com/Cable/) (standard) plus [VB-Audio Hi-Fi Cable](https://vb-audio.com/Cable/#DownloadASIOBridge) (secondary).
- **A Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey). Free tier works.
- **Python 3.11 or 3.12.** The setup script auto-downloads 3.12 if needed. 3.13+ is not supported.
- **VRChat** on the same machine, with OSC enabled in the action menu.

---

## Audio Routing

After starting Gabriel, set these in the Windows Volume Mixer (right-click the speaker icon):

| Application | Output | Input |
|:---|:---|:---|
| Python | CABLE Input (VB-Audio Virtual Cable) | Hi-Fi Cable Output (VB-Audio Hi-Fi) |
| VRChat | Hi-Fi Cable Input (VB-Audio Hi-Fi) | Default / Microphone |

Then in VRChat Settings &rarr; Audio &rarr; Microphone:

- **Microphone Device:** `CABLE Output` (VB-Audio Virtual Cable)
- **Noise Suppression:** OFF
- **Activation Threshold:** 0%

---

## Configuration

Everything lives in `config.yml`. Key sections:

| Section | What it does |
|:---|:---|
| `gemini` &rarr; `api_key` | Your Gemini API key (required) |
| `gemini` &rarr; `model` | Which Gemini Live model to use |
| `gemini` &rarr; `voice` | Prebuilt voice: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr |
| `gemini` &rarr; `vad` &rarr; `mode` | Voice Activity Detection: `auto` (server-side) or `silero` (local) |
| `gemini` &rarr; `thinking` | Thinking budget/config for the model's inner monologue |
| `vrchat` &rarr; `osc_ip`, `osc_send_port`, `osc_receive_port` | OSC IP and ports |
| `vrchat_api` | VRChat account credentials for avatar switching, friends, etc |
| `yolo` &rarr; `enabled` | Toggle person/face tracking |
| `memory` | Memory backend settings |
| `plugins` | Plugin loader and trust settings |

Prompt files live in `config/prompts/`. Personalities live in `config/prompts/personalities.yml`.

---

## FAQ

<details>
<summary><strong>"I can't log into VRChat" / "VRChat API calls fail with 401"</strong></summary>

Delete `data/vrchat_cookies.json` and restart Gabriel. The auth cookies can get stale or corrupted, and deleting them forces a fresh login on next attempt. If it still fails, double-check the `username` and `password` fields under the `vrchat_api` section in `config.yml`.

</details>

<details>
<summary><strong>"Is VRChat required?"</strong></summary>

Not at all. Gabriel runs just fine as a standalone voice AI. You can talk to him through Discord, the WebUI at `http://localhost:8766`, or any other interface. The Discord bot, social server, and local backend all work without VRChat even installed. He was designed with VRChat in mind and that's where all the OSC, vision, and navigation features live, but nothing forces you to use it.

</details>

<details>
<summary><strong>"Does the AI have to be named Gabriel?"</strong></summary>

No. Gabriel is just the project name and the default app name. Change it in `config.yml` under `app_name` and update your prompt files in `config/prompts/`. You can give him any name, personality, or backstory you want. Everything is configurable.

</details>

<details>
<summary><strong>"How do I change the AI's voice?"</strong></summary>

Set the `voice` field under `gemini` in `config.yml` to any of the eight built-in names: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, or Zephyr. Puck and Kore are the most popular. You can also use external TTS providers like Hoppou or Chirp 3 HD through the local backend.

</details>

<details>
<summary><strong>"Does the AI have to be male?"</strong></summary>

Not at all. Gabriel is fully customizable. Pick a female voice like Aoede or Zephyr, write a prompt that describes a female persona, and that's who your AI is. The project name is just a name. Everything from voice to personality to gender is defined by your config and prompt files.

</details>

<details>
<summary><strong>"Gemini Live disconnects with error 1007 or 1008"</strong></summary>

These are precondition failures, usually caused by sending audio while the model is mid-turn or sending text before the model is ready. Try switching `mode` under `gemini` &rarr; `vad` to `silero` since it gates audio during tool calls and model speech and is generally more stable on 3.1 models.

If you're on a 2.5 model, make sure `context_window_compression` is set to `enabled: true` under `gemini` to prevent the session from hitting the token limit.

</details>

<details>
<summary><strong>"I'm getting 429 rate limit errors"</strong></summary>

Add backup API keys under `gemini` &rarr; `backup_keys` in `config.yml`. Gabriel rotates keys automatically when he hits a quota limit. Each free API key has its own quota, so having a few on hand keeps things running.

</details>

<details>
<summary><strong>"The AI can't hear me" / "It never responds"</strong></summary>

Check your audio routing in the Windows Volume Mixer. Python's output should go to `CABLE Input`, and VRChat's microphone should be set to `CABLE Output`.

If routing looks right, try lowering `silence_duration_ms` under `gemini` &rarr; `vad`. 200ms is the default and works well for natural conversation.

</details>

<details>
<summary><strong>"The AI voice sounds robotic or choppy"</strong></summary>

Disable the voice effects chain by removing or commenting out the voice config in `config/voices.yml`. Pedalboard effects can sometimes introduce artifacts. If that fixes it, re-enable effects one at a time to find the culprit.

Also try switching to a different Gemini voice. Puck and Kore tend to be the most reliable.

</details>

<details>
<summary><strong>"Setup fails / packages won't install"</strong></summary>

Make sure you have Visual C++ build tools installed. If you get compiler errors during the `uv sync` step, download [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and make sure the "Desktop development with C++" workload is selected.

If you're on an ARM machine (Snapdragon X, etc), some packages like `pyaudio` may need special handling. Hit up the Discord for help.

</details>

<details>
<summary><strong>"How do I enable the GPU for faster vision/YOLO?"</strong></summary>

Run `setup.bat` and choose option 2 (NVIDIA GPU ONLY) or manually swap in CUDA PyTorch:

```bash
.\bin\uv.exe pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio --reinstall
```

</details>

<details>
<summary><strong>"The chatbox shows garbled text or cuts off"</strong></summary>

VRChat's chatbox has a 144 character hard limit. Gabriel auto-paginates with `(1/N)` prefixes, but if you're seeing issues, try adjusting `chatbox_page_delay` under `vrchat` in `config.yml` (default is 3 seconds).

</details>

<details>
<summary><strong>"Can I run this on Linux?"</strong></summary>

The project targets Windows first. Some things will work on Linux (Python code, Discord bot, social server) but OSC routing, audio device enumeration, pynput keyboard control, and the screen capture for YOLO are all Windows-specific at the moment. PRs welcome.

</details>

<details>
<summary><strong>"Where do I put my music/SFX files?"</strong></summary>

Drop them in `sfx/music/`. Gabriel can play them via the soundboard & play music tool(s).

</details>

<details>
<summary><strong>"How do I add a custom TTS voice?"</strong></summary>

Gabriel supports several TTS providers out of the box: Hoppou AI cloud, Google Chirp 3 HD, and TikTok TTS.

</details>

<details>
<summary><strong>"My config.yml got corrupted / things are acting weird"</strong></summary>

Don't panic. Compare your `config.yml` against `config.yml.example`. The example file always reflects the current schema. Look for missing keys, wrong indentation, or leftover values from an older version. If you recently upgraded, new fields may have been added that you're missing.

</details>

---

## Plugins

Gabriel has a drop-in plugin system. Create a folder under `plugins/<name>/` with a `plugin.yml` manifest and an `__init__.py` that subclasses `Plugin`. Plugins can:

- Register Gemini function-calling tools
- Register custom TTS providers, and custom STT / ASR providers for the local backend
- Grab vision frames the AI sees on demand (`ctx.capture_vision_frame()`)
- Write to the VRChat chatbox
- Inject text into the system prompt
- Subscribe to lifecycle events

Official plugins live at **[HoppouAI/ProjectGabriel-Plugins](https://github.com/HoppouAI/ProjectGabriel-Plugins)**. The author guide is in [plugins/README.md](plugins/README.md).

### Installing plugins

Run **`plugins.bat`** for an interactive TUI that pulls the latest plugin
list from the official repo, shows what's already installed, and one
keypress installs a plugin folder plus runs `bin\uv.exe pip install` for
any dependencies it needs. Press `L` inside the TUI to install from a
local folder or `G` to point it at a fork instead.

---

## Project Structure

```
main.py              Entry point
supervisor.py        Auto-restart on crash
configurator.py      Interactive setup wizard
plugin_installer.py  Interactive plugin installer (TUI)
src/
  audio.py           Audio I/O, effects, music/SFX
  vrchat.py          VRChat OSC client (movement, chatbox, voice)
  vrchatapi.py       VRChat REST API client (avatars, friends, worlds)
  tracker.py         YOLOv8 person tracking
  face_tracker.py    YOLOv8 face tracking
  gemini_live/       Gemini Live session management
  memory/            Persistent memory (MongoDB / SQLite + vector search)
  tools/             Gemini function-calling tools
  plugins/           Plugin loader and API
discord_bot/         Discord bot (separate Gemini Live session)
social_server/       Social messaging API server (Node.js)
webui/               Dashboard HTML/JS/CSS
config/              Prompt files, voices, tool toggles
```

---

## License

Licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

Additional terms under AGPL Section 7 apply to the Gabriel AI persona. See [NOTICE.md](NOTICE.md).

---

<p align="center">
  <a href="https://discord.gg/ZNWTYTk4Vq"><img alt="Discord" src="https://img.shields.io/badge/discord-join_for_support-6366f1?style=for-the-badge&logo=discord&logoColor=white"></a>
  &nbsp;
  <a href="https://github.com/HoppouAI/ProjectGabriel-Plugins"><img alt="Plugins" src="https://img.shields.io/badge/plugins-repo-6366f1?style=for-the-badge&logo=github&logoColor=white"></a>
</p>
