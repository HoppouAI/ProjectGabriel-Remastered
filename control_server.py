"""Gabriel Control Panel WebUI - manage the Gemini Live session."""
import asyncio
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

SESSION_HANDLE_FILE = Path("session_handle.txt")

shared_state = {
    "session": None,
    "usage_metadata": None,
    "is_connected": False,
    "mic_muted": False,
    "last_activity": None,
    "personality_mgr": None,
    "audio_mgr": None,
    "memory_mgr": None,
    "get_emotion_fn": None,
}

# Console log buffer (last 100 entries)
console_logs: deque = deque(maxlen=100)

websocket_clients: list[WebSocket] = []

app = FastAPI(title="Gabriel Control Panel")


def add_console_log(log_type: str, content: str, extra: dict = None):
    """Add a log entry to the console buffer and schedule broadcast to clients."""
    entry = {
        "type": log_type,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "extra": extra or {}
    }
    console_logs.append(entry)
    # Schedule broadcast in background (non-blocking)
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcast_log(entry))
    except Exception:
        pass
    return entry


def get_session_handle_info() -> dict:
    """Get current session handle info."""
    if not SESSION_HANDLE_FILE.exists():
        return {"exists": False, "handle": None, "saved_at": None, "age_minutes": None}
    try:
        data = json.loads(SESSION_HANDLE_FILE.read_text(encoding="utf-8"))
        saved_at = datetime.fromisoformat(data["created"])
        age = (datetime.now() - saved_at).total_seconds() / 60
        return {
            "exists": True,
            "handle": data["handle"][:20] + "..." if data.get("handle") else None,
            "saved_at": data["created"],
            "age_minutes": round(age, 1)
        }
    except Exception:
        return {"exists": True, "handle": None, "saved_at": None, "age_minutes": None, "error": "Parse error"}


async def broadcast_state():
    """Send current state to all connected WebSocket clients."""
    state = get_full_state()
    disconnected = []
    for ws in websocket_clients:
        try:
            await ws.send_json({"type": "state", "data": state})
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        websocket_clients.remove(ws)


async def broadcast_log(entry: dict):
    """Broadcast a new log entry to all WebSocket clients."""
    disconnected = []
    for ws in websocket_clients:
        try:
            await ws.send_json({"type": "log", "data": entry})
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        websocket_clients.remove(ws)


def get_full_state():
    """Get full application state."""
    session = shared_state.get("session")
    usage = None
    mic_muted = shared_state.get("mic_muted", False)
    is_connected = shared_state.get("is_connected", False)

    if session:
        is_connected = session._session is not None
        mic_muted = getattr(session, "_mic_muted", False)
        if hasattr(session, "_usage_metadata"):
            usage = session._usage_metadata

    # Get personalities
    personalities = []
    current_personality = None
    personality_mgr = shared_state.get("personality_mgr")
    if personality_mgr:
        try:
            personalities_data = personality_mgr.list_personalities()
            personalities = personalities_data.get("personalities", [])
            current_data = personality_mgr.get_current()
            current_personality = current_data.get("id")
        except Exception:
            pass

    return {
        "is_connected": is_connected,
        "mic_muted": mic_muted,
        "usage_metadata": usage,
        "last_activity": shared_state.get("last_activity"),
        "session_handle": get_session_handle_info(),
        "personalities": personalities,
        "current_personality": current_personality,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gabriel Control</title>
    <style>
        :root {
            --bg-dark: #0d0d12;
            --bg-card: rgba(18, 18, 25, 0.95);
            --bg-card-hover: rgba(25, 25, 35, 0.98);
            --bg-input: rgba(30, 30, 42, 0.8);
            --border: rgba(80, 80, 120, 0.25);
            --border-hover: rgba(124, 92, 255, 0.4);
            --text: #f0f0f8;
            --text-secondary: #9090a8;
            --text-muted: #606078;
            --accent: #7c5cff;
            --accent-glow: rgba(124, 92, 255, 0.35);
            --accent-secondary: #5c8cff;
            --success: #4ade80;
            --danger: #f87171;
            --warning: #fbbf24;
            --music-accent: #ff6b9d;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at 20% 0%, rgba(124, 92, 255, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(92, 140, 255, 0.04) 0%, transparent 40%);
        }
        
        /* Header */
        .header {
            background: linear-gradient(180deg, rgba(124, 92, 255, 0.08) 0%, transparent 100%);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .logo {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-secondary) 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            box-shadow: 0 4px 16px var(--accent-glow);
        }
        .header-title h1 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text);
        }
        .header-title .subtitle {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        .status-badges {
            display: flex;
            gap: 0.75rem;
        }
        .badge {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.75rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            font-size: 0.75rem;
        }
        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
        }
        .badge-dot.connected { background: var(--success); box-shadow: 0 0 8px var(--success); }
        .badge-dot.disconnected { background: var(--danger); }
        .badge-dot.muted { background: var(--warning); }
        
        /* Main Layout */
        .main-container {
            display: grid;
            grid-template-columns: 300px 1fr 320px;
            gap: 1.5rem;
            padding: 1.5rem 2rem;
            max-width: 1600px;
            margin: 0 auto;
            min-height: calc(100vh - 80px);
        }
        
        /* Columns */
        .column {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        /* Cards */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.25rem;
            transition: all 0.2s ease;
        }
        .card:hover {
            border-color: var(--border-hover);
        }
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .card-header .icon {
            font-size: 1.1rem;
        }
        
        /* Buttons */
        .btn {
            padding: 0.65rem 1rem;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-secondary) 100%);
            color: white;
            box-shadow: 0 2px 10px var(--accent-glow);
        }
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px var(--accent-glow);
        }
        .btn-secondary {
            background: var(--bg-input);
            color: var(--text);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover {
            border-color: var(--accent);
            background: rgba(124, 92, 255, 0.1);
        }
        .btn-danger {
            background: rgba(248, 113, 113, 0.15);
            color: var(--danger);
            border: 1px solid rgba(248, 113, 113, 0.3);
        }
        .btn-danger:hover { background: rgba(248, 113, 113, 0.25); }
        .btn-warning {
            background: rgba(251, 191, 36, 0.15);
            color: var(--warning);
            border: 1px solid rgba(251, 191, 36, 0.3);
        }
        .btn-warning:hover { background: rgba(251, 191, 36, 0.25); }
        .btn-success {
            background: rgba(74, 222, 128, 0.15);
            color: var(--success);
            border: 1px solid rgba(74, 222, 128, 0.3);
        }
        .btn-success:hover { background: rgba(74, 222, 128, 0.25); }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        .btn-group {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .btn-group .btn { flex: 1; min-width: 80px; }
        
        /* Inputs */
        .input-group {
            display: flex;
            gap: 0.5rem;
        }
        .text-input {
            flex: 1;
            padding: 0.65rem 0.875rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 0.8rem;
            transition: all 0.15s ease;
        }
        .text-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .text-input::placeholder { color: var(--text-muted); }
        
        /* Session Info */
        .session-info {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
        }
        .session-info code {
            background: var(--bg-input);
            padding: 0.1rem 0.35rem;
            border-radius: 4px;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.7rem;
        }
        
        /* Now Playing */
        .now-playing {
            background: linear-gradient(135deg, rgba(255, 107, 157, 0.1) 0%, rgba(124, 92, 255, 0.1) 100%);
            border-color: rgba(255, 107, 157, 0.3);
        }
        .now-playing-content {
            padding: 0.25rem 0;
        }
        .song-title {
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .progress-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .progress-bar {
            flex: 1;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--music-accent), var(--accent));
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        .time-info {
            font-size: 0.65rem;
            color: var(--text-muted);
            min-width: 35px;
        }
        .music-controls {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.75rem;
        }
        .music-btn {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--bg-input);
            border: 1px solid var(--border);
            color: var(--text);
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
        }
        .music-btn:hover {
            border-color: var(--music-accent);
            background: rgba(255, 107, 157, 0.15);
        }
        .music-btn.play-btn {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--music-accent), var(--accent));
            border: none;
            font-size: 1rem;
        }
        .music-btn.play-btn:hover {
            transform: scale(1.05);
        }
        .volume-control {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-left: auto;
        }
        .volume-label {
            font-size: 0.8rem;
        }
        .volume-slider {
            width: 70px;
            -webkit-appearance: none;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            outline: none;
        }
        .volume-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }
        
        /* Music List */
        .music-library-card {
            display: flex;
            flex-direction: column;
            max-height: 280px;
        }
        .music-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        .music-header-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .upload-btn {
            padding: 0.3rem 0.6rem;
            font-size: 0.7rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .upload-btn:hover {
            border-color: var(--accent);
            color: var(--text);
        }
        .music-list {
            flex: 1;
            overflow-y: auto;
            min-height: 0;
        }
        .music-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.6rem;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.15s ease;
        }
        .music-item:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        .music-item.playing {
            background: rgba(255, 107, 157, 0.15);
        }
        .music-icon {
            width: 36px;
            height: 36px;
            background: var(--bg-input);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
        }
        .music-info {
            flex: 1;
            min-width: 0;
        }
        .music-name {
            font-size: 0.8rem;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .music-duration {
            font-size: 0.7rem;
            color: var(--text-muted);
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }
        .stat-item {
            background: var(--bg-input);
            border-radius: 10px;
            padding: 0.875rem;
            text-align: center;
        }
        .stat-value {
            font-size: 1.25rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-label {
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.2rem;
        }
        
        /* Memory List */
        .memory-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .memory-item {
            padding: 0.6rem;
            border-radius: 6px;
            margin-bottom: 0.4rem;
            background: var(--bg-input);
            font-size: 0.75rem;
        }
        .memory-item .key {
            color: var(--accent);
            font-weight: 500;
        }
        .memory-item .value {
            color: var(--text-secondary);
            margin-top: 0.2rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .memory-empty {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            padding: 1.5rem;
        }
        
        /* Console */
        .console-card {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-height: 400px;
        }
        .console-card .card-header {
            justify-content: space-between;
        }
        .console {
            flex: 1;
            min-height: 200px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 0.75rem;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.75rem;
            line-height: 1.5;
        }
        .console-entry {
            padding: 0.3rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .console-entry:last-child {
            border-bottom: none;
        }
        .console-entry.transcription { color: #8be9fd; }
        .console-entry.response { color: #50fa7b; }
        .console-entry.tool_call { color: #ffb86c; }
        .console-entry.tool_response { color: #bd93f9; }
        .console-entry.error { color: #ff5555; }
        .console-entry.info { color: var(--text-secondary); }
        .console-detail {
            margin-top: 0.25rem;
            padding-left: 1.5rem;
            color: var(--text-muted);
            font-size: 0.7rem;
        }
        .text-input-card {
            flex-shrink: 0;
        }
        .btn-icon {
            background: transparent;
            border: none;
            cursor: pointer;
            font-size: 0.9rem;
            padding: 0.25rem;
            opacity: 0.6;
            transition: opacity 0.15s ease;
        }
        .btn-icon:hover {
            opacity: 1;
        }
        .column-left {
            min-width: 250px;
        }
        .column-center {
            display: flex;
            flex-direction: column;
        }
        .column-right {
            min-width: 280px;
        }
        
        /* Toast */
        .toast-container {
            position: fixed;
            bottom: 1.5rem;
            right: 1.5rem;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .toast {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.875rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            animation: slideIn 0.3s ease;
            backdrop-filter: blur(10px);
            font-size: 0.8rem;
        }
        .toast.success { border-left: 3px solid var(--success); }
        .toast.error { border-left: 3px solid var(--danger); }
        .toast.info { border-left: 3px solid var(--accent); }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .main-container {
                grid-template-columns: 1fr 1fr;
            }
            .column:nth-child(3) {
                grid-column: span 2;
            }
        }
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                gap: 1rem;
                padding: 1rem;
            }
            .main-container {
                grid-template-columns: 1fr;
                padding: 1rem;
            }
            .column:nth-child(3) {
                grid-column: span 1;
            }
            .stats-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <div class="logo">🎭</div>
            <div class="header-title">
                <h1>Gabriel Control</h1>
                <div class="subtitle">VRChat AI Session Manager</div>
            </div>
        </div>
        <div class="status-badges">
            <div class="badge">
                <div class="badge-dot" id="connectionDot"></div>
                <span id="connectionStatus">Disconnected</span>
            </div>
            <div class="badge">
                <div class="badge-dot" id="micDot"></div>
                <span id="micStatus">Mic Active</span>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <!-- Left Column: Music -->
        <div class="column column-left">
            <div class="card now-playing">
                <div class="card-header"><span class="icon">🎵</span> Now Playing</div>
                <div class="now-playing-content">
                    <div class="song-title" id="songTitle">No music playing</div>
                    <div class="progress-row">
                        <span class="time-info" id="currentTime">0:00</span>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                        </div>
                        <span class="time-info" id="totalTime">0:00</span>
                    </div>
                    <div class="music-controls">
                        <button class="music-btn play-btn" id="playBtn" onclick="togglePlay()">▶️</button>
                        <button class="music-btn" onclick="stopMusic()">⏹️</button>
                        <div class="volume-control">
                            <span class="volume-label">🔊</span>
                            <input type="range" class="volume-slider" id="volumeSlider" min="0" max="100" value="70" onchange="setVolume(this.value)">
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card music-library-card">
                <div class="music-header-row">
                    <div class="music-header-title"><span class="icon">📁</span> Music Library</div>
                    <button class="upload-btn" onclick="openMusicFolder()">📂 Open Folder</button>
                </div>
                <div class="music-list" id="musicList">
                    <div class="memory-empty">Loading music...</div>
                </div>
            </div>
        </div>
        
        <!-- Center Column: Console -->
        <div class="column column-center">
            <div class="card console-card">
                <div class="card-header">
                    <span class="icon">📟</span> Console
                    <button class="btn-icon" onclick="clearConsole()" title="Clear">🗑️</button>
                </div>
                <div class="console" id="console">
                    <div class="console-entry info">Waiting for connection...</div>
                </div>
            </div>
            <div class="card text-input-card">
                <div class="input-group">
                    <input type="text" class="text-input" id="textInput" placeholder="Type message to model..." onkeypress="if(event.key==='Enter')sendText()">
                    <button class="btn btn-primary" onclick="sendText()">📤 Send</button>
                </div>
            </div>
        </div>
        
        <!-- Right Column: Controls & Stats -->
        <div class="column column-right">
            <div class="card">
                <div class="card-header"><span class="icon">🎤</span> Session Control</div>
                <div class="btn-group">
                    <button class="btn btn-secondary" id="reconnectBtn" onclick="reconnect()">🔄 Reconnect</button>
                    <button class="btn btn-danger" id="clearSessionBtn" onclick="clearSession()">🗑️ Clear</button>
                </div>
                <div class="btn-group" style="margin-top: 0.5rem;">
                    <button class="btn btn-warning" id="muteBtn" onclick="toggleMute()">🔇 Mute</button>
                </div>
                <div class="session-info" id="sessionInfo">Loading session info...</div>
            </div>
            
            <div class="card">
                <div class="card-header"><span class="icon">🎭</span> Personality</div>
                <select class="text-input" id="personalitySelect" onchange="switchPersonality()" style="width: 100%;">
                    <option value="">Loading...</option>
                </select>
            </div>
            
            <div class="card">
                <div class="card-header"><span class="icon">📊</span> Usage Stats</div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="promptTokens">-</div>
                        <div class="stat-label">Prompt</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="responseTokens">-</div>
                        <div class="stat-label">Response</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="totalTokens">-</div>
                        <div class="stat-label">Total</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="toolCalls">-</div>
                        <div class="stat-label">Tools</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header"><span class="icon">🧠</span> Recent Memories</div>
                <div class="memory-list" id="memoryList">
                    <div class="memory-empty">No memories yet</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="toast-container" id="toastContainer"></div>
    
    <script>
        let ws = null;
        let isMuted = false;
        let isPlaying = false;
        let musicFiles = [];
        let currentTrackIndex = -1;
        
        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
            toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
            document.getElementById('toastContainer').appendChild(toast);
            setTimeout(() => toast.remove(), 4000);
        }
        
        let lastStreamingEntry = null;
        
        function addConsoleEntry(type, content, extra = {}) {
            const consoleEl = document.getElementById('console');
            
            // For streaming responses, append to last entry
            if (extra.streaming && lastStreamingEntry && lastStreamingEntry.dataset.type === type) {
                const textNode = lastStreamingEntry.childNodes[0];
                if (textNode) {
                    textNode.textContent += content;
                }
                consoleEl.scrollTop = consoleEl.scrollHeight;
                return;
            }
            
            const entry = document.createElement('div');
            entry.className = `console-entry ${type}`;
            entry.dataset.type = type;
            
            let prefix = '';
            if (type === 'transcription') prefix = '🎤 ';
            else if (type === 'response') prefix = '🤖 ';
            else if (type === 'tool_call') prefix = '🔧 ';
            else if (type === 'tool_response') prefix = '📥 ';
            else if (type === 'error') prefix = '❌ ';
            else if (type === 'info') prefix = 'ℹ️ ';
            
            entry.textContent = prefix + content;
            if (extra.detail) {
                const detail = document.createElement('div');
                detail.className = 'console-detail';
                detail.textContent = extra.detail;
                entry.appendChild(detail);
            }
            
            consoleEl.appendChild(entry);
            consoleEl.scrollTop = consoleEl.scrollHeight;
            
            // Track for streaming
            if (extra.streaming) {
                lastStreamingEntry = entry;
            } else {
                lastStreamingEntry = null;
            }
            
            // Limit entries
            while (consoleEl.children.length > 200) {
                consoleEl.removeChild(consoleEl.firstChild);
            }
        }
        
        function clearConsole() {
            const consoleEl = document.getElementById('console');
            consoleEl.innerHTML = '<div class="console-entry info">Console cleared</div>';
            lastStreamingEntry = null;
        }
        
        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            
            ws.onopen = () => {
                addConsoleEntry('info', 'WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'state') {
                    updateUI(msg.data);
                } else if (msg.type === 'toast') {
                    showToast(msg.message, msg.level);
                } else if (msg.type === 'log') {
                    addConsoleEntry(msg.data.type, msg.data.content, msg.data.extra || {});
                }
            };
            
            ws.onclose = () => {
                addConsoleEntry('error', 'WebSocket disconnected, reconnecting...');
                setTimeout(connectWS, 2000);
            };
        }
        
        function updateUI(state) {
            // Connection status
            const connDot = document.getElementById('connectionDot');
            const connStatus = document.getElementById('connectionStatus');
            if (state.is_connected) {
                connDot.className = 'badge-dot connected';
                connStatus.textContent = 'Connected';
            } else {
                connDot.className = 'badge-dot disconnected';
                connStatus.textContent = 'Disconnected';
            }
            
            // Mic status
            const micDot = document.getElementById('micDot');
            const micStatus = document.getElementById('micStatus');
            isMuted = state.mic_muted;
            if (isMuted) {
                micDot.className = 'badge-dot muted';
                micStatus.textContent = 'Muted';
                document.getElementById('muteBtn').innerHTML = '🔊 Unmute';
            } else {
                micDot.className = 'badge-dot connected';
                micStatus.textContent = 'Mic Active';
                document.getElementById('muteBtn').innerHTML = '🔇 Mute';
            }
            
            // Session info
            const sessionInfo = document.getElementById('sessionInfo');
            if (state.session_handle && state.session_handle.exists) {
                sessionInfo.innerHTML = `Handle: <code>${state.session_handle.handle || '...'}</code><br>Age: ${state.session_handle.age_minutes || '?'}m`;
            } else {
                sessionInfo.innerHTML = 'No active session';
            }
            
            // Usage stats
            if (state.usage_metadata) {
                document.getElementById('promptTokens').textContent = formatNumber(state.usage_metadata.prompt_tokens);
                document.getElementById('responseTokens').textContent = formatNumber(state.usage_metadata.response_tokens);
                document.getElementById('totalTokens').textContent = formatNumber(state.usage_metadata.total_tokens);
                document.getElementById('toolCalls').textContent = state.usage_metadata.tool_calls || '-';
            }
            
            // Personalities
            if (state.personalities && state.personalities.length > 0) {
                const select = document.getElementById('personalitySelect');
                const currentVal = select.value;
                select.innerHTML = state.personalities.map(p => 
                    `<option value="${p.id}" ${p.id === state.current_personality ? 'selected' : ''}>${p.name}</option>`
                ).join('');
            }
            
            // Music progress
            if (state.music_progress) {
                updateMusicProgress(state.music_progress);
            }
            
            // Memories
            if (state.recent_memories) {
                updateMemories(state.recent_memories);
            }
        }
        
        function formatNumber(n) {
            if (!n && n !== 0) return '-';
            if (n >= 1000000) return (n/1000000).toFixed(1) + 'M';
            if (n >= 1000) return (n/1000).toFixed(1) + 'K';
            return n.toString();
        }
        
        function formatTime(seconds) {
            if (!seconds || seconds < 0) return '0:00';
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${m}:${s.toString().padStart(2, '0')}`;
        }
        
        function updateMusicProgress(data) {
            if (data.is_playing) {
                isPlaying = true;
                document.getElementById('songTitle').textContent = data.song_name || 'Unknown';
                document.getElementById('currentTime').textContent = formatTime(data.position);
                document.getElementById('totalTime').textContent = formatTime(data.duration);
                const progress = data.duration > 0 ? (data.position / data.duration) * 100 : 0;
                document.getElementById('progressFill').style.width = progress + '%';
                document.getElementById('playBtn').innerHTML = '⏸️';
            } else {
                isPlaying = false;
                document.getElementById('playBtn').innerHTML = '▶️';
                if (!data.song_name) {
                    document.getElementById('songTitle').textContent = 'No music playing';
                    document.getElementById('progressFill').style.width = '0%';
                }
            }
        }
        
        function updateMemories(memories) {
            const list = document.getElementById('memoryList');
            if (!memories || memories.length === 0) {
                list.innerHTML = '<div class="memory-empty">No memories yet</div>';
                return;
            }
            list.innerHTML = memories.slice(0, 10).map(m => `
                <div class="memory-item">
                    <div class="key">${m.key}</div>
                    <div class="value">${m.value}</div>
                </div>
            `).join('');
        }
        
        async function loadMusicList() {
            try {
                const resp = await fetch('/api/music-list');
                const data = await resp.json();
                musicFiles = data.files || [];
                const list = document.getElementById('musicList');
                if (musicFiles.length === 0) {
                    list.innerHTML = '<div class="memory-empty">No music files found</div>';
                    return;
                }
                list.innerHTML = musicFiles.map((f, i) => `
                    <div class="music-item" onclick="playTrack(${i})">
                        <div class="music-icon">🎵</div>
                        <div class="music-info">
                            <div class="music-name">${f.name}</div>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('musicList').innerHTML = '<div class="memory-empty">Failed to load music</div>';
            }
        }
        
        async function apiCall(endpoint, method = 'POST', body = null) {
            try {
                const options = { method };
                if (body) {
                    options.headers = { 'Content-Type': 'application/json' };
                    options.body = JSON.stringify(body);
                }
                const response = await fetch(endpoint, options);
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Request failed');
                return data;
            } catch (err) {
                showToast(err.message, 'error');
                throw err;
            }
        }
        
        async function reconnect() {
            document.getElementById('reconnectBtn').disabled = true;
            try {
                await apiCall('/api/reconnect');
                showToast('Reconnecting...', 'info');
            } finally {
                setTimeout(() => document.getElementById('reconnectBtn').disabled = false, 2000);
            }
        }
        
        async function clearSession() {
            if (!confirm('Clear session and start fresh?')) return;
            document.getElementById('clearSessionBtn').disabled = true;
            try {
                await apiCall('/api/clear-session');
                showToast('Session cleared', 'success');
            } finally {
                setTimeout(() => document.getElementById('clearSessionBtn').disabled = false, 2000);
            }
        }
        
        async function toggleMute() {
            try { await apiCall('/api/toggle-mute'); } catch (err) {}
        }
        
        async function sendText() {
            const input = document.getElementById('textInput');
            const text = input.value.trim();
            if (!text) return;
            try {
                addConsoleEntry('info', `Sending: ${text}`);
                await apiCall('/api/send-text', 'POST', { text });
                input.value = '';
            } catch (err) {}
        }
        
        async function switchPersonality() {
            const select = document.getElementById('personalitySelect');
            const personality = select.value;
            if (!personality) return;
            try {
                await apiCall('/api/switch-personality', 'POST', { personality });
                showToast('Personality switched', 'success');
            } catch (err) {}
        }
        
        async function playTrack(index) {
            if (index < 0 || index >= musicFiles.length) return;
            currentTrackIndex = index;
            try {
                await apiCall('/api/play-music', 'POST', { filename: musicFiles[index].name });
            } catch (err) {}
        }
        
        async function togglePlay() {
            try {
                if (isPlaying) {
                    await apiCall('/api/pause-music');
                } else {
                    await apiCall('/api/resume-music');
                }
            } catch (err) {}
        }
        
        async function stopMusic() {
            try { await apiCall('/api/stop-music'); } catch (err) {}
        }
        
        async function setVolume(val) {
            try { await apiCall('/api/set-volume', 'POST', { volume: val / 100 }); } catch (err) {}
        }
        
        async function openMusicFolder() {
            try {
                await apiCall('/api/open-music-folder');
                showToast('Music folder opened', 'info');
            } catch (err) {}
        }
        
        // Initialize
        fetch('/api/state').then(r => r.json()).then(updateUI).catch(() => {});
        loadMusicList();
        connectWS();
        setInterval(() => fetch('/api/state').then(r => r.json()).then(updateUI).catch(() => {}), 5000);
    </script>
</body>
</html>"""


class TextInput(BaseModel):
    text: str


class MusicInput(BaseModel):
    filename: str


class VolumeInput(BaseModel):
    volume: float


class PersonalityInput(BaseModel):
    personality: str


class EmotionInput(BaseModel):
    emotion: str


@app.get("/api/state")
async def get_state():
    session = shared_state.get("session")
    usage = None
    mic_muted = shared_state.get("mic_muted", False)
    is_connected = shared_state.get("is_connected", False)
    
    if session:
        is_connected = session._session is not None
        mic_muted = getattr(session, "_mic_muted", False)
        if hasattr(session, "_usage_metadata"):
            usage = session._usage_metadata
    
    # Get personalities
    personalities = []
    current_personality = None
    personality_mgr = shared_state.get("personality_mgr")
    if personality_mgr:
        try:
            personalities_data = personality_mgr.list_personalities()
            personalities = personalities_data.get("personalities", [])
            current_data = personality_mgr.get_current()
            current_personality = current_data.get("id")
        except Exception:
            pass
    
    # Get music progress
    music_progress = None
    audio_mgr = shared_state.get("audio_mgr")
    if audio_mgr and hasattr(audio_mgr, "get_music_progress"):
        try:
            raw_progress = audio_mgr.get_music_progress()
            if raw_progress:
                music_progress = {
                    "is_playing": True,
                    "song_name": raw_progress.get("name", "Unknown"),
                    "artist": "",  # Not tracked currently
                    "position": raw_progress.get("position", 0),
                    "duration": raw_progress.get("duration", 0),
                }
            else:
                music_progress = {"is_playing": False}
        except Exception:
            music_progress = {"is_playing": False}
    
    # Get recent memories
    recent_memories = []
    memory_mgr = shared_state.get("memory_mgr")
    if memory_mgr and hasattr(memory_mgr, "list_memories"):
        try:
            result = memory_mgr.list_memories(limit=10)
            recent_memories = result.get("memories", [])
        except Exception:
            pass
    
    return {
        "is_connected": is_connected,
        "mic_muted": mic_muted,
        "usage_metadata": usage,
        "last_activity": shared_state.get("last_activity"),
        "session_handle": get_session_handle_info(),
        "personalities": personalities,
        "current_personality": current_personality,
        "music_progress": music_progress,
        "recent_memories": recent_memories,
    }


@app.post("/api/reconnect")
async def reconnect():
    session = shared_state.get("session")
    if session and hasattr(session, "request_reconnect"):
        session.request_reconnect()
        return {"message": "Reconnect requested"}
    return {"message": "No active session to reconnect"}


@app.post("/api/clear-session")
async def clear_session():
    # Delete session handle file
    if SESSION_HANDLE_FILE.exists():
        SESSION_HANDLE_FILE.unlink()
    
    # Request reconnect
    session = shared_state.get("session")
    if session and hasattr(session, "request_reconnect"):
        if hasattr(session, "_session_handle"):
            session._session_handle = None
        session.request_reconnect()
        return {"message": "Session cleared and reconnect requested"}
    return {"message": "Session handle cleared"}


@app.post("/api/toggle-mute")
async def toggle_mute():
    session = shared_state.get("session")
    if session and hasattr(session, "_mic_muted"):
        session._mic_muted = not session._mic_muted
        shared_state["mic_muted"] = session._mic_muted
    else:
        shared_state["mic_muted"] = not shared_state.get("mic_muted", False)
    await broadcast_state()
    return {"muted": shared_state["mic_muted"]}


@app.post("/api/send-text")
async def send_text(data: TextInput):
    session = shared_state.get("session")
    if not session:
        raise HTTPException(status_code=400, detail="No active session")
    
    if hasattr(session, "send_text"):
        await session.send_text(data.text)
        return {"message": "Text sent"}
    else:
        raise HTTPException(status_code=400, detail="Session does not support text input")


@app.get("/api/music-list")
async def get_music_list():
    """Get list of available music files."""
    music_dir = Path("sfx/music")
    if not music_dir.exists():
        return {"files": []}
    
    files = []
    for f in music_dir.iterdir():
        if f.suffix.lower() in (".mp3", ".wav", ".ogg", ".flac"):
            files.append({
                "name": f.name,
                "path": str(f),
            })
    return {"files": sorted(files, key=lambda x: x["name"])}


@app.post("/api/play-music")
async def play_music(data: MusicInput):
    """Play a music file."""
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    
    try:
        result = audio_mgr.play_music(data.filename)
        await broadcast_state()
        return {"success": result, "filename": data.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pause-music")
async def pause_music():
    """Pause currently playing music."""
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    
    try:
        result = audio_mgr.pause_music()
        await broadcast_state()
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/resume-music")
async def resume_music():
    """Resume paused music."""
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    
    try:
        result = audio_mgr.resume_music()
        await broadcast_state()
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/stop-music")
async def stop_music():
    """Stop music playback."""
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    
    try:
        audio_mgr.stop_music()
        await broadcast_state()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-volume")
async def set_volume(data: VolumeInput):
    """Set music volume (0.0 to 1.0)."""
    audio_mgr = shared_state.get("audio_mgr")
    if not audio_mgr:
        raise HTTPException(status_code=400, detail="Audio manager not available")
    
    try:
        # Convert 0-1 to 0-100 (AudioManager expects 0-300, we use 0-100 range)
        volume_int = int(data.volume * 100)
        result = audio_mgr.set_music_volume(volume_int)
        return {"success": result, "volume": data.volume}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/open-music-folder")
async def open_music_folder():
    """Open the music folder in the system file explorer."""
    import subprocess
    import os
    
    music_dir = Path("sfx/music").resolve()
    music_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if os.name == 'nt':  # Windows
            os.startfile(str(music_dir))
        else:  # macOS/Linux
            subprocess.run(['xdg-open' if os.name != 'darwin' else 'open', str(music_dir)])
        return {"success": True, "path": str(music_dir)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/switch-personality")
async def switch_personality(data: PersonalityInput):
    """Switch to a different personality."""
    personality_mgr = shared_state.get("personality_mgr")
    session = shared_state.get("session")
    
    if not personality_mgr:
        raise HTTPException(status_code=400, detail="Personality manager not available")
    
    try:
        result = personality_mgr.switch(data.personality)
        if result.get("status") == "switched" and session:
            # Inject personality prompt into session
            if hasattr(session, "send_text"):
                prompt = result.get("prompt", "")
                if prompt:
                    await session.send_text(f"[SYSTEM: Your personality has changed. {prompt}]")
        await broadcast_state()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/trigger-emotion")
async def trigger_emotion(data: EmotionInput):
    """Trigger an avatar emotion/animation."""
    get_emotion_fn = shared_state.get("get_emotion_fn")
    emotion_mgr = None
    if get_emotion_fn:
        emotion_mgr = get_emotion_fn()
    
    if not emotion_mgr:
        raise HTTPException(status_code=400, detail="Emotion manager not available")
    
    try:
        result = emotion_mgr.play_emotion(data.emotion)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.append(websocket)
    
    # Send initial state
    state = {
        "is_connected": shared_state.get("is_connected", False),
        "mic_muted": shared_state.get("mic_muted", False),
        "usage_metadata": shared_state.get("usage_metadata"),
        "session_handle": get_session_handle_info(),
    }
    await websocket.send_json({"type": "state", "data": state})
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)


def run_control_server(host: str = "0.0.0.0", port: int = 8766):
    """Run the control panel server."""
    print("\n🎭 Gabriel Control Panel")
    print("=" * 40)
    print(f"Open http://localhost:{port} in your browser")
    print("=" * 40 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_control_server()
