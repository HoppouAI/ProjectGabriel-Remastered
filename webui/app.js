/* Gabriel Control Panel - Client Logic */

// --- State ---
let ws = null;
let isMuted = false;
let isPlaying = false;
let musicFiles = [];
let allUploadFiles = [];
let lastStreamingEntry = null;

// --- Tabs ---

function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${name}"]`).classList.add('active');
    document.querySelectorAll('main.container').forEach(m => m.classList.add('hidden'));
    document.getElementById('tab-' + name).classList.remove('hidden');
}

// --- Toast ---

function showToast(message, type) {
    type = type || 'info';
    const c = document.getElementById('toastContainer');
    const t = document.createElement('div');
    t.className = 'toast ' + type;
    const icons = { success: '\u2705', error: '\u274c', info: '\u2139\ufe0f' };
    t.innerHTML = '<span>' + (icons[type] || icons.info) + '</span><span>' + escapeHtml(message) + '</span>';
    c.appendChild(t);
    setTimeout(function () { t.remove(); }, 4000);
}

// --- Utility ---

function escapeHtml(text) {
    var d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

function escapeJs(text) {
    return text.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

function formatNumber(n) {
    if (n === null || n === undefined) return '-';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toString();
}

function formatTime(seconds) {
    if (!seconds || seconds < 0) return '0:00';
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
}

// --- API Helpers ---

async function apiCall(endpoint, method, body) {
    method = method || 'POST';
    try {
        var opts = { method: method };
        if (body) {
            opts.headers = { 'Content-Type': 'application/json' };
            opts.body = JSON.stringify(body);
        }
        var resp = await fetch(endpoint, opts);
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Request failed');
        return data;
    } catch (err) {
        showToast(err.message, 'error');
        throw err;
    }
}

// --- Session Control ---

async function apiReconnect() {
    var btn = document.getElementById('reconnectBtn');
    btn.disabled = true;
    try {
        await apiCall('/api/reconnect');
        showToast('Reconnecting...', 'info');
    } finally {
        setTimeout(function () { btn.disabled = false; }, 2000);
    }
}

async function apiClearSession() {
    if (!confirm('Clear session and start fresh?')) return;
    var btn = document.getElementById('clearSessionBtn');
    btn.disabled = true;
    try {
        await apiCall('/api/clear-session');
        showToast('Session cleared', 'success');
    } finally {
        setTimeout(function () { btn.disabled = false; }, 2000);
    }
}

async function apiToggleMute() {
    try { await apiCall('/api/toggle-mute'); } catch (e) { /* handled */ }
}

async function apiSendText() {
    var input = document.getElementById('textInput');
    var text = input.value.trim();
    if (!text) return;
    try {
        addConsoleEntry('info', 'Sending: ' + text);
        await apiCall('/api/send-text', 'POST', { text: text });
        input.value = '';
    } catch (e) { /* handled */ }
}

async function apiSendSystemInstruction() {
    var input = document.getElementById('sysInstructionInput');
    var text = input.value.trim();
    if (!text) return;
    try {
        addConsoleEntry('info', 'System instruction: ' + text);
        await apiCall('/api/send-system-instruction', 'POST', { text: text });
        input.value = '';
    } catch (e) { /* handled */ }
}

async function apiSendTextConsole() {
    var input = document.getElementById('consoleTextInput');
    var text = input.value.trim();
    if (!text) return;
    try {
        addConsoleEntry('info', 'Sending: ' + text);
        await apiCall('/api/send-text', 'POST', { text: text });
        input.value = '';
    } catch (e) { /* handled */ }
}

async function apiSwitchPersonality() {
    var sel = document.getElementById('personalitySelect');
    var val = sel.value;
    if (!val) return;
    try {
        await apiCall('/api/switch-personality', 'POST', { personality: val });
        showToast('Personality switched', 'success');
    } catch (e) { /* handled */ }
}

// --- Music Playback ---

async function apiTogglePlay() {
    try {
        if (isPlaying) {
            await apiCall('/api/pause-music');
        } else {
            await apiCall('/api/resume-music');
        }
    } catch (e) { /* handled */ }
}

async function apiStopMusic() {
    try { await apiCall('/api/stop-music'); } catch (e) { /* handled */ }
}

async function apiSetVolume(val) {
    try { await apiCall('/api/set-volume', 'POST', { volume: val / 100 }); } catch (e) { /* handled */ }
}

async function apiPlayTrack(filename) {
    try {
        await apiCall('/api/play-music', 'POST', { filename: filename });
    } catch (e) { /* handled */ }
}

async function apiOpenMusicFolder() {
    try {
        await apiCall('/api/open-music-folder');
        showToast('Music folder opened', 'info');
    } catch (e) { /* handled */ }
}

// --- Music Library (Playback List) ---

async function loadPlaybackMusicList() {
    try {
        var resp = await fetch('/api/music-list');
        var data = await resp.json();
        musicFiles = data.files || [];
    } catch (e) {
        musicFiles = [];
    }
}

// --- Music File Management ---

async function loadMusicFiles() {
    try {
        var resp = await fetch('/api/music-files');
        allUploadFiles = await resp.json();
        renderMusicFiles();
    } catch (e) {
        document.getElementById('fileList').innerHTML = '<div class="empty-state">Failed to load files</div>';
    }
}

async function loadMusicFolders() {
    try {
        var resp = await fetch('/api/music-folders');
        var folders = await resp.json();
        var sel = document.getElementById('folderSelect');
        sel.innerHTML = folders.map(function (f) {
            return '<option value="' + escapeHtml(f) + '">' + (f || '(root)') + '</option>';
        }).join('');
    } catch (e) { /* ignore */ }
}

function renderMusicFiles() {
    var filterVal = document.getElementById('filterInput').value.toLowerCase();
    var filtered = filterVal
        ? allUploadFiles.filter(function (f) { return f.name.toLowerCase().indexOf(filterVal) !== -1; })
        : allUploadFiles;

    document.getElementById('fileCount').textContent = filtered.length + ' file' + (filtered.length !== 1 ? 's' : '');

    if (!filtered.length) {
        document.getElementById('fileList').innerHTML = '<div class="empty-state">No music files found</div>';
        return;
    }

    var grouped = {};
    filtered.forEach(function (file) {
        var folder = file.folder || '(root)';
        if (!grouped[folder]) grouped[folder] = [];
        grouped[folder].push(file);
    });

    var html = '';
    var folders = Object.keys(grouped).sort();

    folders.forEach(function (folder) {
        if (folders.length > 1 || folder !== '(root)') {
            html += '<div class="file-folder-header">\ud83d\udcc1 ' + escapeHtml(folder) + '</div>';
        }
        grouped[folder].forEach(function (file) {
            html += '<div class="file-item">' +
                '<div class="file-icon">\ud83c\udfb5</div>' +
                '<div class="file-info">' +
                    '<div class="file-name">' + escapeHtml(file.display_name) + '</div>' +
                    '<div class="file-meta">' + file.size_mb + ' MB \u2022 ' + new Date(file.modified).toLocaleString() + '</div>' +
                '</div>' +
                '<div class="file-actions">' +
                    '<button class="btn-small" onclick="apiPlayTrack(\'' + escapeJs(file.name) + '\')" title="play">\u25b6</button>' +
                    '<button class="btn-small" onclick="deleteMusicFile(\'' + escapeJs(file.name) + '\')" title="delete" style="color:var(--danger)">\ud83d\uddd1</button>' +
                '</div>' +
            '</div>';
        });
    });

    document.getElementById('fileList').innerHTML = html;
}

async function deleteMusicFile(path) {
    if (!confirm('Delete "' + path + '"?')) return;
    try {
        var resp = await fetch('/api/music-files/' + encodeURIComponent(path), { method: 'DELETE' });
        if (resp.ok) {
            showToast('Deleted: ' + path, 'success');
            loadMusicFiles();
            loadPlaybackMusicList();
        } else {
            var data = await resp.json();
            showToast(data.detail || 'Delete failed', 'error');
        }
    } catch (e) {
        showToast('Delete failed', 'error');
    }
}

// --- Upload ---

function setupUpload() {
    var dropZone = document.getElementById('dropZone');
    var fileInput = document.getElementById('fileInput');

    dropZone.addEventListener('click', function () { fileInput.click(); });

    dropZone.addEventListener('dragover', function (e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', function () {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', function (e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleUploadFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', function () {
        handleUploadFiles(fileInput.files);
    });

    document.getElementById('filterInput').addEventListener('input', renderMusicFiles);
}

async function handleUploadFiles(files) {
    if (!files.length) return;

    var progressBar = document.getElementById('uploadProgress');
    var progressFill = document.getElementById('uploadProgressFill');
    progressBar.classList.add('active');

    var uploaded = 0;
    var folder = document.getElementById('newFolder').value.trim() || document.getElementById('folderSelect').value;

    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        var formData = new FormData();
        formData.append('file', file);
        formData.append('folder', folder);

        try {
            var resp = await fetch('/api/music-upload', { method: 'POST', body: formData });
            var data = await resp.json();

            if (resp.ok) {
                var msg = 'Uploaded: ' + file.name;
                if (data.extracted_count) {
                    msg = 'Extracted ' + data.extracted_count + ' files from ' + file.name;
                }
                showToast(msg, 'success');
                if (data.errors && data.errors.length) {
                    data.errors.forEach(function (err) { showToast(err, 'error'); });
                }
            } else {
                showToast(data.detail || 'Failed: ' + file.name, 'error');
            }
        } catch (e) {
            showToast('Error uploading ' + file.name, 'error');
        }

        uploaded++;
        progressFill.style.width = ((uploaded / files.length) * 100) + '%';
    }

    setTimeout(function () {
        progressBar.classList.remove('active');
        progressFill.style.width = '0%';
    }, 500);

    loadMusicFiles();
    loadMusicFolders();
    loadPlaybackMusicList();
    document.getElementById('fileInput').value = '';
    document.getElementById('newFolder').value = '';
}

// --- Console ---

function addConsoleEntry(type, content, extra) {
    extra = extra || {};
    var consoleEl = document.getElementById('console');

    if (extra.streaming && lastStreamingEntry && lastStreamingEntry.dataset.type === type) {
        var textNode = lastStreamingEntry.childNodes[0];
        if (textNode) textNode.textContent += content;
        consoleEl.scrollTop = consoleEl.scrollHeight;
        return;
    }

    var entry = document.createElement('div');
    entry.className = 'console-entry ' + type;
    entry.dataset.type = type;

    var prefixes = {
        transcription: '\ud83c\udf99 ',
        response: '\ud83e\udd16 ',
        tool_call: '\ud83d\udd27 ',
        tool_response: '\ud83d\udce5 ',
        error: '\u274c ',
        info: '\u2139\ufe0f '
    };
    entry.textContent = (prefixes[type] || '') + content;

    if (extra.detail) {
        var det = document.createElement('div');
        det.className = 'console-detail';
        det.textContent = extra.detail;
        entry.appendChild(det);
    }

    consoleEl.appendChild(entry);
    consoleEl.scrollTop = consoleEl.scrollHeight;

    if (extra.streaming) {
        lastStreamingEntry = entry;
    } else {
        lastStreamingEntry = null;
    }

    while (consoleEl.children.length > 300) {
        consoleEl.removeChild(consoleEl.firstChild);
    }
}

function clearConsole() {
    var consoleEl = document.getElementById('console');
    consoleEl.innerHTML = '<div class="console-entry info">Console cleared</div>';
    lastStreamingEntry = null;
}

// --- WebSocket ---

function connectWS() {
    var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(protocol + '//' + location.host + '/ws');

    ws.onopen = function () {
        addConsoleEntry('info', 'WebSocket connected');
    };

    ws.onmessage = function (event) {
        var msg = JSON.parse(event.data);
        if (msg.type === 'state') {
            updateUI(msg.data);
        } else if (msg.type === 'toast') {
            showToast(msg.message, msg.level);
        } else if (msg.type === 'log') {
            addConsoleEntry(msg.data.type, msg.data.content, msg.data.extra || {});
        }
    };

    ws.onclose = function () {
        addConsoleEntry('error', 'WebSocket disconnected, reconnecting...');
        setTimeout(connectWS, 2500);
    };
}

// --- UI Update ---

function updateUI(state) {
    // Connection
    var connDot = document.getElementById('connectionDot');
    var connLabel = document.getElementById('connectionLabel');
    if (state.is_connected) {
        connDot.className = 'badge-dot connected';
        connLabel.textContent = 'Connected';
    } else {
        connDot.className = 'badge-dot disconnected';
        connLabel.textContent = 'Disconnected';
    }

    // Mic
    var micDot = document.getElementById('micDot');
    var micLabel = document.getElementById('micLabel');
    isMuted = state.mic_muted;
    if (isMuted) {
        micDot.className = 'badge-dot muted';
        micLabel.textContent = 'Muted';
        document.getElementById('muteBtn').textContent = 'Unmute Mic';
    } else {
        micDot.className = 'badge-dot active';
        micLabel.textContent = 'Mic Active';
        document.getElementById('muteBtn').textContent = 'Mute Mic';
    }

    // Session info
    var meta = document.getElementById('sessionMeta');
    if (state.session_handle && state.session_handle.exists) {
        meta.innerHTML = 'handle: <code>' + (state.session_handle.handle || '...') + '</code><br>age: ' + (state.session_handle.age_minutes || '?') + 'm';
    } else {
        meta.textContent = 'No active session';
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
        var sel = document.getElementById('personalitySelect');
        sel.innerHTML = state.personalities.map(function (p) {
            return '<option value="' + escapeHtml(p.id) + '"' +
                (p.id === state.current_personality ? ' selected' : '') +
                '>' + escapeHtml(p.name) + '</option>';
        }).join('');
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

function updateMusicProgress(data) {
    if (data.is_playing) {
        isPlaying = true;
        document.getElementById('songTitle').textContent = data.song_name || 'Unknown';
        document.getElementById('currentTime').textContent = formatTime(data.position);
        document.getElementById('totalTime').textContent = formatTime(data.duration);
        var pct = data.duration > 0 ? (data.position / data.duration) * 100 : 0;
        document.getElementById('progressFill').style.width = pct + '%';
        document.getElementById('playBtn').innerHTML = '\u23f8\ufe0f';
    } else {
        isPlaying = false;
        document.getElementById('playBtn').innerHTML = '\u25b6\ufe0f';
        if (!data.song_name) {
            document.getElementById('songTitle').textContent = 'No music playing';
            document.getElementById('progressFill').style.width = '0%';
        }
    }
}

function updateMemories(memories) {
    var list = document.getElementById('memoryList');
    if (!memories || memories.length === 0) {
        list.innerHTML = '<div class="empty-state">No memories yet</div>';
        return;
    }
    list.innerHTML = memories.slice(0, 10).map(function (m) {
        return '<div class="memory-item">' +
            '<div class="mem-key">' + escapeHtml(m.key) + '</div>' +
            '<div class="mem-val">' + escapeHtml(m.value) + '</div>' +
        '</div>';
    }).join('');
}

// --- 7-Zip Status ---

function check7Zip() {
    fetch('/api/sevenzip-status')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            var el = document.getElementById('sevenzipStatus');
            if (data.available) {
                el.textContent = '7-Zip available';
                el.style.color = 'var(--success)';
            } else {
                el.textContent = '7-Zip not found (archives disabled)';
                el.style.color = 'var(--warning)';
            }
        })
        .catch(function () {});
}

// --- Init ---

function init() {
    setupUpload();
    check7Zip();
    connectWS();
    loadMusicFiles();
    loadMusicFolders();
    loadPlaybackMusicList();

    // Initial state fetch
    fetch('/api/state').then(function (r) { return r.json(); }).then(updateUI).catch(function () {});

    // Periodic state poll (backup for WebSocket)
    setInterval(function () {
        fetch('/api/state').then(function (r) { return r.json(); }).then(updateUI).catch(function () {});
    }, 5000);

    // Periodic music file refresh
    setInterval(function () {
        loadMusicFiles();
    }, 15000);
}

init();
