"""Gabriel Control Panel WebUI - manage the Gemini Live session."""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

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
}

websocket_clients: list[WebSocket] = []

app = FastAPI(title="Gabriel Control Panel")


def get_session_handle_info() -> dict:
    """Get current session handle info."""
    if not SESSION_HANDLE_FILE.exists():
        return {"exists": False, "handle": None, "saved_at": None, "age_minutes": None}
    try:
        data = json.loads(SESSION_HANDLE_FILE.read_text(encoding="utf-8"))
        saved_at = datetime.fromisoformat(data["saved_at"])
        age = (datetime.now() - saved_at).total_seconds() / 60
        return {
            "exists": True,
            "handle": data["handle"][:20] + "..." if data.get("handle") else None,
            "saved_at": data["saved_at"],
            "age_minutes": round(age, 1)
        }
    except Exception:
        return {"exists": True, "handle": None, "saved_at": None, "age_minutes": None, "error": "Parse error"}


async def broadcast_state():
    """Send current state to all connected WebSocket clients."""
    state = {
        "is_connected": shared_state.get("is_connected", False),
        "mic_muted": shared_state.get("mic_muted", False),
        "usage_metadata": shared_state.get("usage_metadata"),
        "last_activity": shared_state.get("last_activity"),
        "session_handle": get_session_handle_info(),
    }
    disconnected = []
    for ws in websocket_clients:
        try:
            await ws.send_json({"type": "state", "data": state})
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        websocket_clients.remove(ws)


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
            --bg-dark: #0a0a0f;
            --bg-card: rgba(20, 20, 30, 0.8);
            --bg-card-hover: rgba(30, 30, 45, 0.9);
            --border: rgba(100, 100, 150, 0.2);
            --text: #e8e8f0;
            --text-secondary: #8888a0;
            --accent: #7c5cff;
            --accent-glow: rgba(124, 92, 255, 0.3);
            --accent-secondary: #5c8cff;
            --success: #4ade80;
            --danger: #f87171;
            --warning: #fbbf24;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at 50% 0%, rgba(124, 92, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 100% 100%, rgba(92, 140, 255, 0.05) 0%, transparent 40%);
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }
        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        .logo {
            width: 64px;
            height: 64px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-secondary) 100%);
            border-radius: 16px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px var(--accent-glow);
        }
        h1 {
            font-size: 1.75rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--text) 0%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
        }
        .status-dot.connected { background: var(--success); box-shadow: 0 0 8px var(--success); }
        .status-dot.disconnected { background: var(--danger); }
        .status-dot.muted { background: var(--warning); }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            transition: all 0.2s ease;
        }
        .card:hover {
            background: var(--bg-card-hover);
            border-color: rgba(124, 92, 255, 0.3);
        }
        .card-title {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .card-title span { font-size: 1.25rem; }
        .btn-group {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .btn {
            flex: 1;
            min-width: 120px;
            padding: 0.875rem 1.25rem;
            border: none;
            border-radius: 10px;
            font-weight: 500;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-secondary) 100%);
            color: white;
            box-shadow: 0 4px 15px var(--accent-glow);
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px var(--accent-glow);
        }
        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent);
        }
        .btn-danger {
            background: rgba(248, 113, 113, 0.1);
            color: var(--danger);
            border: 1px solid rgba(248, 113, 113, 0.3);
        }
        .btn-danger:hover {
            background: rgba(248, 113, 113, 0.2);
        }
        .btn-warning {
            background: rgba(251, 191, 36, 0.1);
            color: var(--warning);
            border: 1px solid rgba(251, 191, 36, 0.3);
        }
        .btn-warning:hover {
            background: rgba(251, 191, 36, 0.2);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        .text-input-container {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .text-input {
            flex: 1;
            min-width: 200px;
            padding: 0.875rem 1rem;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text);
            font-size: 0.875rem;
            transition: all 0.2s ease;
        }
        .text-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .text-input::placeholder { color: var(--text-secondary); }
        .usage-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
        }
        .usage-item {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .usage-value {
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .usage-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        .session-info {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }
        .session-info code {
            background: rgba(255, 255, 255, 0.05);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
        }
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
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            animation: slideIn 0.3s ease;
            backdrop-filter: blur(10px);
        }
        .toast.success { border-left: 3px solid var(--success); }
        .toast.error { border-left: 3px solid var(--danger); }
        .toast.info { border-left: 3px solid var(--accent); }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @media (max-width: 600px) {
            .container { padding: 1rem; }
            .logo { width: 56px; height: 56px; font-size: 1.75rem; }
            h1 { font-size: 1.5rem; }
            .card { padding: 1.25rem; }
            .btn { min-width: 100px; padding: 0.75rem 1rem; }
            .btn-group { gap: 0.5rem; }
            .usage-grid { grid-template-columns: repeat(2, 1fr); }
            .text-input-container { flex-direction: column; }
            .text-input { min-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">🎭</div>
            <h1>Gabriel Control Panel</h1>
            <div class="subtitle">Manage your VRChat AI session</div>
        </header>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot" id="connectionDot"></div>
                <span id="connectionStatus">Disconnected</span>
            </div>
            <div class="status-item">
                <div class="status-dot" id="micDot"></div>
                <span id="micStatus">Mic Active</span>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title"><span>🎤</span> Session Control</div>
            <div class="btn-group">
                <button class="btn btn-secondary" id="reconnectBtn" onclick="reconnect()">
                    🔄 Reconnect
                </button>
                <button class="btn btn-danger" id="clearSessionBtn" onclick="clearSession()">
                    🗑️ Clear & New
                </button>
                <button class="btn btn-warning" id="muteBtn" onclick="toggleMute()">
                    🔇 Mute Mic
                </button>
            </div>
            <div class="session-info" id="sessionInfo">
                Loading session info...
            </div>
        </div>
        
        <div class="card">
            <div class="card-title"><span>💬</span> Send Text to Model</div>
            <div class="text-input-container">
                <input type="text" class="text-input" id="textInput" placeholder="Type a message to inject into the conversation..." onkeypress="if(event.key==='Enter')sendText()">
                <button class="btn btn-primary" onclick="sendText()">
                    📤 Send
                </button>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title"><span>📊</span> Usage Metadata</div>
            <div class="usage-grid" id="usageGrid">
                <div class="usage-item">
                    <div class="usage-value" id="promptTokens">-</div>
                    <div class="usage-label">Prompt Tokens</div>
                </div>
                <div class="usage-item">
                    <div class="usage-value" id="responseTokens">-</div>
                    <div class="usage-label">Response Tokens</div>
                </div>
                <div class="usage-item">
                    <div class="usage-value" id="totalTokens">-</div>
                    <div class="usage-label">Total Tokens</div>
                </div>
                <div class="usage-item">
                    <div class="usage-value" id="toolCalls">-</div>
                    <div class="usage-label">Tool Calls</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="toast-container" id="toastContainer"></div>
    
    <script>
        let ws = null;
        let isMuted = false;
        
        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
            toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
            document.getElementById('toastContainer').appendChild(toast);
            setTimeout(() => toast.remove(), 4000);
        }
        
        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'state') {
                    updateUI(msg.data);
                } else if (msg.type === 'toast') {
                    showToast(msg.message, msg.level);
                }
            };
            
            ws.onclose = () => {
                setTimeout(connectWS, 2000);
            };
        }
        
        function updateUI(state) {
            const connDot = document.getElementById('connectionDot');
            const connStatus = document.getElementById('connectionStatus');
            const micDot = document.getElementById('micDot');
            const micStatus = document.getElementById('micStatus');
            
            if (state.is_connected) {
                connDot.className = 'status-dot connected';
                connStatus.textContent = 'Connected';
            } else {
                connDot.className = 'status-dot disconnected';
                connStatus.textContent = 'Disconnected';
            }
            
            isMuted = state.mic_muted;
            if (isMuted) {
                micDot.className = 'status-dot muted';
                micStatus.textContent = 'Mic Muted';
                document.getElementById('muteBtn').innerHTML = '🔊 Unmute';
            } else {
                micDot.className = 'status-dot connected';
                micStatus.textContent = 'Mic Active';
                document.getElementById('muteBtn').innerHTML = '🔇 Mute Mic';
            }
            
            // Session info
            const sessionInfo = document.getElementById('sessionInfo');
            if (state.session_handle && state.session_handle.exists) {
                sessionInfo.innerHTML = `
                    Session handle: <code>${state.session_handle.handle || 'Unknown'}</code><br>
                    Age: ${state.session_handle.age_minutes || '?'} minutes
                `;
            } else {
                sessionInfo.innerHTML = 'No active session handle (will create new on connect)';
            }
            
            // Usage metadata
            if (state.usage_metadata) {
                document.getElementById('promptTokens').textContent = state.usage_metadata.prompt_tokens || '-';
                document.getElementById('responseTokens').textContent = state.usage_metadata.response_tokens || '-';
                document.getElementById('totalTokens').textContent = state.usage_metadata.total_tokens || '-';
                document.getElementById('toolCalls').textContent = state.usage_metadata.tool_calls || '-';
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
            try {
                document.getElementById('reconnectBtn').disabled = true;
                await apiCall('/api/reconnect');
                showToast('Reconnecting...', 'info');
            } finally {
                setTimeout(() => document.getElementById('reconnectBtn').disabled = false, 2000);
            }
        }
        
        async function clearSession() {
            if (!confirm('Clear session handle and start fresh? This will disconnect and create a new session.')) return;
            try {
                document.getElementById('clearSessionBtn').disabled = true;
                await apiCall('/api/clear-session');
                showToast('Session cleared, reconnecting...', 'success');
            } finally {
                setTimeout(() => document.getElementById('clearSessionBtn').disabled = false, 2000);
            }
        }
        
        async function toggleMute() {
            try {
                await apiCall('/api/toggle-mute');
            } catch (err) {}
        }
        
        async function sendText() {
            const input = document.getElementById('textInput');
            const text = input.value.trim();
            if (!text) return;
            
            try {
                await apiCall('/api/send-text', 'POST', { text });
                showToast('Text sent to model', 'success');
                input.value = '';
            } catch (err) {}
        }
        
        // Fetch initial state
        fetch('/api/state').then(r => r.json()).then(updateUI).catch(() => {});
        
        // Connect WebSocket
        connectWS();
        
        // Poll state every 5 seconds as backup
        setInterval(() => {
            fetch('/api/state').then(r => r.json()).then(updateUI).catch(() => {});
        }, 5000);
    </script>
</body>
</html>"""


class TextInput(BaseModel):
    text: str


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
    
    return {
        "is_connected": is_connected,
        "mic_muted": mic_muted,
        "usage_metadata": usage,
        "last_activity": shared_state.get("last_activity"),
        "session_handle": get_session_handle_info(),
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
