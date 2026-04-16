# Gabriel Voice Control - Vencord Plugin

A [Vencord](https://vencord.dev) UserPlugin that allows ProjectGabriel AI to control Discord voice channels and calls via a local WebSocket API.

## How It Works

```
AI (ProjectGabriel) --WebSocket--> Vencord Plugin (native.ts) --IPC--> Discord (renderer)
                                                                         |
Discord Events <-- IPC <-- Vencord Plugin <-- WebSocket broadcast <-- AI receives events
```

The plugin runs a local WebSocket server inside Discord (Electron main process). The AI connects as a client and sends commands like `join_voice`, `call_user`, `leave_voice`, etc. The plugin executes them using Discord's internal APIs and sends back results.

## Prerequisites

You need Vencord built from source. If you haven't done this yet, follow the [official guide](https://docs.vencord.dev/installing/):

1. Install [git](https://git-scm.com/downloads), [Node.js](https://nodejs.org/en/download/), and [pnpm](https://pnpm.io/installation)
2. Clone Vencord:
   ```bash
   git clone https://github.com/Vendicated/Vencord
   cd Vencord
   ```
3. Install dependencies:
   ```bash
   pnpm install --frozen-lockfile
   ```

## Installation

### 1. Copy the plugin

Create the `src/userplugins` folder in your Vencord repo if it doesn't exist, then copy this plugin folder into it:

```
Vencord/src/userplugins/gabrielVoiceControl/
    index.ts    (renderer - plugin entry, Discord API calls)
    native.ts   (main process - WebSocket server)
    types.ts    (shared TypeScript types)
```

> **Warning:** Do not leave empty folders or empty files inside `userplugins` -- this causes a `TypeError: Cannot read properties of undefined (reading 'localeCompare')` error.

### 2. Build & inject Vencord

```bash
cd Vencord
pnpm build
pnpm inject
```

The injector will open and let you patch your Discord install. After patching, restart Discord.

> Whenever you make changes to the plugin, rebuild with `pnpm build` and restart Discord. You can also use `pnpm build --watch` during development.

### 3. Enable the plugin

Open Discord > Settings > Vencord > Plugins > search **GabrielVoiceControl** > Enable it.

Configure the WebSocket port in plugin settings (default: `9473`).

> **Note:** Do NOT use port 6463-6472 as Discord's RPC server uses those.

### 4. AI side (ProjectGabriel)

The voice control tools are automatically available in both the main VRChat AI session and the Discord bot session. The AI connects to `ws://127.0.0.1:9473` on demand when a voice command is invoked.

**Requires `websockets` Python package** (already in requirements.txt).

## Commands (WebSocket API)

Send JSON over WebSocket:

| Command | Fields | Description |
|---------|--------|-------------|
| `join_voice` | `channel_id` | Join a voice channel |
| `leave_voice` | - | Leave current voice channel |
| `call_user` | `channel_id` | Start a DM/group call |
| `answer_call` | `channel_id` | Answer an incoming call |
| `hang_up` | - | Hang up current call |
| `get_voice_state` | - | Get current voice status |
| `set_mute` | `mute` (bool) | Toggle self mute |
| `set_deaf` | `deaf` (bool) | Toggle self deaf |

### Example

```json
{"op": "join_voice", "channel_id": "123456789", "nonce": "abc-123"}
```

Response:
```json
{"op": "command_result", "nonce": "abc-123", "success": true, "data": {"channel_id": "123456789", "guild_id": "987654321"}}
```

## Events (pushed to AI clients)

| Event | Data | Description |
|-------|------|-------------|
| `voice_state_update` | `user_id`, `channel_id`, `guild_id` | Someone joined/left/moved |
| `call_incoming` | `channel_id`, `ringing` | Incoming call |

## Security

- WebSocket binds to `127.0.0.1` only (localhost)
- No authentication (local-only by design)
- All commands execute in the context of the logged-in Discord user

## AI Tools

### Discord Bot Session
- `joinVoiceChannel(channel_id)` - Join a voice channel
- `callUser(channel_id?)` - Start a DM call (defaults to current channel)
- `leaveVoiceChannel()` - Leave voice / hang up
- `getVoiceState()` - Check current voice status

### Main VRChat Session
- `discord_joinVoice(channel_id)` - Join a Discord voice channel
- `discord_callUser(channel_id)` - Start a Discord call
- `discord_leaveVoice()` - Leave Discord voice
- `discord_getVoiceState()` - Get Discord voice status
