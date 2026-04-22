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
| `join_voice` | `channel_id` | Join a voice channel (server or DM) |
| `leave_voice` | - | Leave current voice channel |
| `call_user` | `channel_id` | Join a DM/group DM channel and ring recipients |
| `call_user_by_id` | `user_id` | Create or open DM, join voice, and ring a specific user |
| `answer_call` | `channel_id` | Accept an incoming call |
| `hang_up` | - | Disconnect from voice and stop ringing |
| `get_voice_state` | - | Get current voice state (channel, users, mute/deaf) |
| `set_mute` | `mute` (bool) | Toggle self mute |
| `set_deaf` | `deaf` (bool) | Toggle self deafen |
| `find_user` | `query` | Search for a user by username or display name |
| `ping` | - | Health check |

All commands accept an optional `nonce` string for response matching.

### Example

```json
{"op": "join_voice", "channel_id": "123456789", "nonce": "abc-123"}
```

Response:
```json
{"op": "command_result", "nonce": "abc-123", "success": true, "data": {"channel_id": "123456789", "guild_id": "987654321"}}
```

`find_user` response:
```json
{"op": "command_result", "success": true, "data": {"users": [{"id": "...", "username": "...", "display_name": "...", "dm_channel_id": "...", "is_friend": true, "mutual_guild_ids": ["..."]}], "count": 1}}
```

## Events (pushed to AI clients)

| Event | Data | Description |
|-------|------|-------------|
| `voice_state_update` | `user_id`, `channel_id`, `guild_id` | User joined, left, or moved voice channel |
| `call_incoming` | `channel_id`, `ringing` | Incoming call received |

## Security

- WebSocket binds to `127.0.0.1` only (localhost)
- No authentication (local-only by design)
- All commands execute in the context of the logged-in Discord user

## AI Tools

These are the Gemini function-calling tools exposed to the AI sessions. They map to the WebSocket commands above internally.

### Discord Bot Session

| Tool | Parameters | Description |
|------|-----------|-------------|
| `joinVoiceChannel` | `channel_id` | Join a Discord voice channel by ID |
| `callUser` | `channel_id?`, `user_id?`, `name?` | Call someone. Resolves by channel ID, user ID, or name search (auto-creates DM if needed) |
| `leaveVoiceChannel` | - | Leave voice or hang up a call |
| `findUser` | `query` | Search for a Discord user by username or display name |
| `getVoiceState` | - | Get current voice state (channel, who's in it, mute/deaf status) |

### Main VRChat Session

| Tool | Parameters | Description |
|------|-----------|-------------|
| `sendDiscordMessage` | `username`, `message` | Send a DM to a Discord user via the selfbot |
| `relayToDiscord` | `content` | Relay a message to the Discord bot session (AI-to-AI coordination) |
| `getDiscordStatus` | - | Check if the Discord bot is connected and get its status |
| `discord_hangUp` | - | Hang up / leave the current Discord voice call |
