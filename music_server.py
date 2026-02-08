"""FastAPI Music Upload Server with WebUI for ProjectGabriel."""
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

MUSIC_DIR = Path("sfx/music")
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
ARCHIVE_EXTENSIONS = {".zip", ".7z", ".rar"}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

app = FastAPI(title="Gabriel Music Manager", description="Upload and manage music files for ProjectGabriel")

MUSIC_DIR.mkdir(parents=True, exist_ok=True)


def find_7zip():
    """Find 7-Zip executable."""
    candidates = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "7z.exe",
        "7z",
    ]
    for path in candidates:
        if shutil.which(path) or (Path(path).exists() and Path(path).is_file()):
            return path
    return None


SEVENZIP_PATH = find_7zip()


def get_file_info(filepath: Path, base_dir: Path = MUSIC_DIR) -> dict:
    stat = filepath.stat()
    rel_path = filepath.relative_to(base_dir)
    return {
        "name": str(rel_path),
        "display_name": filepath.name,
        "folder": str(rel_path.parent) if rel_path.parent != Path(".") else "",
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def get_all_music_files(base_dir: Path = MUSIC_DIR) -> list[dict]:
    """Recursively get all music files including subdirectories."""
    files = []
    for filepath in base_dir.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(get_file_info(filepath, base_dir))
    return files


def get_folder_structure(base_dir: Path = MUSIC_DIR) -> list[str]:
    """Get all subdirectories."""
    folders = [""]  # root folder
    for path in base_dir.rglob("*"):
        if path.is_dir():
            rel = str(path.relative_to(base_dir))
            if rel not in folders:
                folders.append(rel)
    return sorted(folders)


def extract_archive(archive_path: Path, extract_to: Path) -> tuple[int, list[str]]:
    """Extract archive using 7-Zip and return (count, errors)."""
    if not SEVENZIP_PATH:
        return 0, ["7-Zip not found. Install it from https://www.7-zip.org/"]
    
    errors = []
    extracted_count = 0
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            result = subprocess.run(
                [SEVENZIP_PATH, "x", str(archive_path), f"-o{tmp_path}", "-y"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                errors.append(f"Extraction failed: {result.stderr or 'Unknown error'}")
                return 0, errors
            
            for filepath in tmp_path.rglob("*"):
                if filepath.is_file() and filepath.suffix.lower() in ALLOWED_EXTENSIONS:
                    rel_path = filepath.relative_to(tmp_path)
                    dest = extract_to / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    
                    counter = 1
                    while dest.exists():
                        stem = rel_path.stem
                        dest = extract_to / rel_path.parent / f"{stem}_{counter}{rel_path.suffix}"
                        counter += 1
                    
                    shutil.copy2(filepath, dest)
                    extracted_count += 1
    except subprocess.TimeoutExpired:
        errors.append("Extraction timed out (max 5 minutes)")
    except Exception as e:
        errors.append(f"Extraction error: {str(e)}")
    
    return extracted_count, errors


@app.get("/", response_class=HTMLResponse)
async def index():
    sevenzip_status = "✅ 7-Zip found" if SEVENZIP_PATH else "⚠️ 7-Zip not installed (ZIP uploads disabled)"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gabriel Music</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-hover: #79b8ff;
            --success: #3fb950;
            --danger: #f85149;
            --warning: #d29922;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        .logo {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--accent), #a371f7);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }}
        h1 {{ font-size: 1.75rem; font-weight: 600; }}
        .subtitle {{ color: var(--text-secondary); font-size: 0.875rem; }}
        .status-badge {{
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            background: var(--bg-tertiary);
            border-radius: 4px;
            margin-left: auto;
        }}
        .upload-zone {{
            background: var(--bg-secondary);
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.2s ease;
            cursor: pointer;
            margin-bottom: 1rem;
        }}
        .upload-zone:hover, .upload-zone.dragover {{
            border-color: var(--accent);
            background: rgba(88, 166, 255, 0.05);
        }}
        .upload-zone.dragover {{ transform: scale(1.01); }}
        .upload-icon {{ font-size: 3rem; margin-bottom: 1rem; }}
        .upload-text {{ color: var(--text-secondary); margin-bottom: 0.5rem; }}
        .upload-hint {{ color: var(--text-secondary); font-size: 0.75rem; }}
        input[type="file"] {{ display: none; }}
        .folder-select {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }}
        .folder-select label {{ color: var(--text-secondary); font-size: 0.875rem; }}
        .folder-select select {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
        }}
        .folder-select input {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            flex: 1;
            min-width: 150px;
        }}
        .btn {{
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            padding: 0.625rem 1.25rem;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .btn:hover {{ background: var(--accent-hover); }}
        .btn-danger {{ background: var(--danger); }}
        .btn-danger:hover {{ background: #ff6b6b; }}
        .btn-sm {{ padding: 0.375rem 0.75rem; font-size: 0.875rem; }}
        .btn-secondary {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-primary);
        }}
        .btn-secondary:hover {{ background: var(--bg-tertiary); border-color: var(--accent); }}
        .section-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}
        .section-title {{ font-size: 1.25rem; font-weight: 600; }}
        .file-count {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            padding: 0.25rem 0.625rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        .filter-input {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.375rem 0.75rem;
            border-radius: 6px;
            font-size: 0.875rem;
            margin-left: auto;
        }}
        .music-list {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            max-height: 600px;
            overflow-y: auto;
        }}
        .folder-header {{
            background: var(--bg-tertiary);
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
        }}
        .music-item {{
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            transition: background 0.15s ease;
        }}
        .music-item:last-child {{ border-bottom: none; }}
        .music-item:hover {{ background: var(--bg-tertiary); }}
        .music-icon {{
            width: 36px;
            height: 36px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.75rem;
            font-size: 1rem;
        }}
        .music-info {{ flex: 1; min-width: 0; }}
        .music-name {{
            font-weight: 500;
            font-size: 0.9rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .music-path {{
            color: var(--accent);
            font-size: 0.7rem;
            opacity: 0.7;
        }}
        .music-meta {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            margin-top: 0.125rem;
        }}
        .music-actions {{ display: flex; gap: 0.5rem; }}
        .empty-state {{
            padding: 3rem;
            text-align: center;
            color: var(--text-secondary);
        }}
        .toast-container {{
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            max-width: 400px;
        }}
        .toast {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            animation: slideIn 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .toast.success {{ border-left: 3px solid var(--success); }}
        .toast.error {{ border-left: 3px solid var(--danger); }}
        .toast.info {{ border-left: 3px solid var(--accent); }}
        @keyframes slideIn {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        .progress-bar {{
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 1rem;
            display: none;
        }}
        .progress-bar.active {{ display: block; }}
        .progress-fill {{
            height: 100%;
            background: var(--accent);
            width: 0%;
            transition: width 0.3s ease;
        }}
        
        /* Mobile responsive styles */
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            header {{
                flex-wrap: wrap;
                gap: 0.75rem;
            }}
            .logo {{
                width: 40px;
                height: 40px;
                font-size: 1.25rem;
            }}
            h1 {{ font-size: 1.25rem; }}
            .subtitle {{ font-size: 0.75rem; }}
            .status-badge {{
                order: 3;
                width: 100%;
                text-align: center;
                margin-left: 0;
                margin-top: 0.5rem;
            }}
            .upload-zone {{
                padding: 2rem 1rem;
            }}
            .upload-icon {{ font-size: 2.5rem; }}
            .upload-text {{ font-size: 0.9rem; }}
            .upload-hint {{ font-size: 0.7rem; line-height: 1.4; }}
            .folder-select {{
                flex-direction: column;
                align-items: stretch;
                gap: 0.75rem;
            }}
            .folder-select label {{
                display: none;
            }}
            .folder-select select,
            .folder-select input {{
                width: 100%;
                min-width: auto;
                padding: 0.75rem 1rem;
            }}
            .section-header {{
                flex-direction: column;
                align-items: stretch;
                gap: 0.75rem;
            }}
            .section-title {{
                font-size: 1.1rem;
            }}
            .filter-input {{
                margin-left: 0;
                width: 100%;
                padding: 0.5rem 0.75rem;
            }}
            .file-count {{
                position: absolute;
                right: 1rem;
                margin-top: -1.5rem;
            }}
            .section-header {{
                position: relative;
            }}
            .music-list {{
                max-height: none;
                border-radius: 8px;
            }}
            .music-item {{
                padding: 0.75rem;
                gap: 0.5rem;
            }}
            .music-icon {{
                width: 32px;
                height: 32px;
                font-size: 0.9rem;
                margin-right: 0.5rem;
                flex-shrink: 0;
            }}
            .music-name {{
                font-size: 0.85rem;
            }}
            .music-meta {{
                font-size: 0.7rem;
            }}
            .music-actions {{
                flex-shrink: 0;
            }}
            .btn-sm {{
                padding: 0.5rem 0.75rem;
            }}
            .toast-container {{
                bottom: 1rem;
                right: 1rem;
                left: 1rem;
                max-width: none;
            }}
            .toast {{
                padding: 0.75rem 1rem;
                font-size: 0.85rem;
            }}
            .empty-state {{
                padding: 2rem 1rem;
                font-size: 0.9rem;
            }}
            .folder-header {{
                padding: 0.5rem 0.75rem;
                font-size: 0.8rem;
            }}
        }}
        
        @media (max-width: 480px) {{
            body {{ padding: 0.75rem; }}
            header {{
                gap: 0.5rem;
            }}
            .logo {{
                width: 36px;
                height: 36px;
                font-size: 1rem;
            }}
            h1 {{ font-size: 1.1rem; }}
            .upload-zone {{
                padding: 1.5rem 1rem;
                border-radius: 8px;
            }}
            .upload-icon {{ font-size: 2rem; margin-bottom: 0.5rem; }}
            .upload-text {{ font-size: 0.85rem; }}
            .music-item {{
                padding: 0.625rem 0.75rem;
            }}
            .music-icon {{
                width: 28px;
                height: 28px;
                font-size: 0.8rem;
            }}
            .music-name {{
                font-size: 0.8rem;
            }}
            .music-meta {{
                font-size: 0.65rem;
            }}
        }}
        
        /* Touch-friendly improvements */
        @media (hover: none) and (pointer: coarse) {{
            .upload-zone:hover {{
                border-color: var(--border);
                background: var(--bg-secondary);
            }}
            .music-item:hover {{
                background: transparent;
            }}
            .music-item:active {{
                background: var(--bg-tertiary);
            }}
            .btn {{
                min-height: 44px;
            }}
            .btn-sm {{
                min-height: 40px;
                padding: 0.5rem 1rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">🎵</div>
            <div>
                <h1>Gabriel Music Manager</h1>
                <div class="subtitle">Upload and manage music files for your VRChat AI</div>
            </div>
            <span class="status-badge">{sevenzip_status}</span>
        </header>
        
        <div class="upload-zone" id="dropZone">
            <div class="upload-icon">📁</div>
            <div class="upload-text">Drag & drop music files here or click to browse</div>
            <div class="upload-hint">Supports MP3, WAV, OGG, FLAC, M4A, AAC, WMA + ZIP/7Z/RAR archives (max 2GB)</div>
            <input type="file" id="fileInput" multiple accept=".mp3,.wav,.ogg,.flac,.m4a,.aac,.wma,.zip,.7z,.rar">
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
        
        <div class="folder-select">
            <label>Upload to folder:</label>
            <select id="folderSelect"></select>
            <span>or create new:</span>
            <input type="text" id="newFolder" placeholder="folder/subfolder">
        </div>
        
        <div class="section-header">
            <span class="section-title">Music Library</span>
            <span class="file-count" id="fileCount">0 files</span>
            <input type="text" class="filter-input" id="filterInput" placeholder="🔍 Filter files...">
            <button class="btn btn-sm btn-secondary" onclick="loadFiles()">🔄 Refresh</button>
        </div>
        
        <div class="music-list" id="musicList">
            <div class="empty-state">Loading...</div>
        </div>
    </div>
    
    <div class="toast-container" id="toastContainer"></div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const musicList = document.getElementById('musicList');
        const fileCount = document.getElementById('fileCount');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const toastContainer = document.getElementById('toastContainer');
        const folderSelect = document.getElementById('folderSelect');
        const newFolder = document.getElementById('newFolder');
        const filterInput = document.getElementById('filterInput');
        
        let allFiles = [];
        
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {{
            e.preventDefault();
            dropZone.classList.add('dragover');
        }});
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {{
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        }});
        fileInput.addEventListener('change', () => handleFiles(fileInput.files));
        filterInput.addEventListener('input', renderFiles);
        
        function showToast(message, type = 'success') {{
            const toast = document.createElement('div');
            toast.className = `toast ${{type}}`;
            const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
            toast.innerHTML = `<span>${{icon}}</span><span>${{message}}</span>`;
            toastContainer.appendChild(toast);
            setTimeout(() => toast.remove(), 5000);
        }}
        
        async function handleFiles(files) {{
            if (!files.length) return;
            progressBar.classList.add('active');
            let uploaded = 0;
            
            const folder = newFolder.value.trim() || folderSelect.value;
            
            for (const file of files) {{
                const formData = new FormData();
                formData.append('file', file);
                formData.append('folder', folder);
                
                try {{
                    const response = await fetch('/upload', {{ method: 'POST', body: formData }});
                    const data = await response.json();
                    
                    if (response.ok) {{
                        let msg = `Uploaded: ${{file.name}}`;
                        if (data.extracted_count) {{
                            msg = `Extracted ${{data.extracted_count}} files from ${{file.name}}`;
                        }}
                        showToast(msg, 'success');
                        if (data.errors && data.errors.length) {{
                            data.errors.forEach(err => showToast(err, 'error'));
                        }}
                    }} else {{
                        showToast(data.detail || `Failed: ${{file.name}}`, 'error');
                    }}
                }} catch (err) {{
                    showToast(`Error uploading ${{file.name}}`, 'error');
                }}
                
                uploaded++;
                progressFill.style.width = `${{(uploaded / files.length) * 100}}%`;
            }}
            
            setTimeout(() => {{
                progressBar.classList.remove('active');
                progressFill.style.width = '0%';
            }}, 500);
            
            loadFiles();
            loadFolders();
            fileInput.value = '';
            newFolder.value = '';
        }}
        
        async function deleteFile(path) {{
            if (!confirm(`Delete "${{path}}"?`)) return;
            
            try {{
                const response = await fetch(`/files/${{encodeURIComponent(path)}}`, {{ method: 'DELETE' }});
                if (response.ok) {{
                    showToast(`Deleted: ${{path}}`, 'success');
                    loadFiles();
                }} else {{
                    const data = await response.json();
                    showToast(data.detail || 'Delete failed', 'error');
                }}
            }} catch (err) {{
                showToast('Delete failed', 'error');
            }}
        }}
        
        function renderFiles() {{
            const filter = filterInput.value.toLowerCase();
            const filtered = filter ? allFiles.filter(f => f.name.toLowerCase().includes(filter)) : allFiles;
            
            fileCount.textContent = `${{filtered.length}} file${{filtered.length !== 1 ? 's' : ''}}`;
            
            if (!filtered.length) {{
                musicList.innerHTML = '<div class="empty-state">No music files found</div>';
                return;
            }}
            
            const grouped = {{}};
            filtered.forEach(file => {{
                const folder = file.folder || '(root)';
                if (!grouped[folder]) grouped[folder] = [];
                grouped[folder].push(file);
            }});
            
            let html = '';
            const folders = Object.keys(grouped).sort();
            
            for (const folder of folders) {{
                if (folders.length > 1 || folder !== '(root)') {{
                    html += `<div class="folder-header">📁 ${{escapeHtml(folder)}}</div>`;
                }}
                grouped[folder].forEach(file => {{
                    html += `
                        <div class="music-item">
                            <div class="music-icon">🎵</div>
                            <div class="music-info">
                                <div class="music-name">${{escapeHtml(file.display_name)}}</div>
                                <div class="music-meta">${{file.size_mb}} MB • ${{new Date(file.modified).toLocaleString()}}</div>
                            </div>
                            <div class="music-actions">
                                <button class="btn btn-sm btn-danger" onclick="deleteFile('${{escapeJs(file.name)}}')">🗑️</button>
                            </div>
                        </div>
                    `;
                }});
            }}
            
            musicList.innerHTML = html;
        }}
        
        async function loadFiles() {{
            try {{
                const response = await fetch('/files');
                allFiles = await response.json();
                renderFiles();
            }} catch (err) {{
                musicList.innerHTML = '<div class="empty-state">Failed to load files</div>';
            }}
        }}
        
        async function loadFolders() {{
            try {{
                const response = await fetch('/folders');
                const folders = await response.json();
                folderSelect.innerHTML = folders.map(f => 
                    `<option value="${{escapeHtml(f)}}">${{f || '(root)'}}</option>`
                ).join('');
            }} catch (err) {{}}
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        function escapeJs(text) {{
            return text.replace(/\\\\/g, '\\\\\\\\').replace(/'/g, "\\\\'");
        }}
        
        loadFiles();
        loadFolders();
        setInterval(loadFiles, 10000);
    </script>
</body>
</html>"""


@app.get("/files")
async def list_files():
    return sorted(get_all_music_files(), key=lambda x: x["name"].lower())


@app.get("/folders")
async def list_folders():
    return get_folder_structure()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), folder: str = ""):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = Path(file.filename).suffix.lower()
    is_archive = ext in ARCHIVE_EXTENSIONS
    
    if not is_archive and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS | ARCHIVE_EXTENSIONS)}"
        )
    
    if is_archive and not SEVENZIP_PATH:
        raise HTTPException(
            status_code=400,
            detail="7-Zip not installed. Cannot extract archives."
        )
    
    folder = folder.strip().strip("/\\")
    folder = "".join(c for c in folder if c.isalnum() or c in "._- /\\").strip()
    target_dir = MUSIC_DIR / folder if folder else MUSIC_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ").strip()
    if not safe_name:
        safe_name = f"music_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    
    dest = target_dir / safe_name
    counter = 1
    while dest.exists():
        stem = Path(safe_name).stem
        dest = target_dir / f"{stem}_{counter}{ext}"
        counter += 1
    
    try:
        size = 0
        with open(dest, "wb") as f:
            while chunk := await file.read(65536):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    f.close()
                    dest.unlink()
                    raise HTTPException(status_code=400, detail="File too large (max 2GB)")
                f.write(chunk)
        
        if is_archive:
            extracted_count, errors = extract_archive(dest, target_dir)
            dest.unlink()
            return {
                "message": "Archive extracted",
                "extracted_count": extracted_count,
                "errors": errors
            }
        
        return {"message": "Upload successful", "filename": dest.name, "size": size}
    except HTTPException:
        raise
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{filepath:path}")
async def delete_file(filepath: str):
    full_path = MUSIC_DIR / filepath
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    if full_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")
    try:
        full_path.resolve().relative_to(MUSIC_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    try:
        full_path.unlink()
        return {"message": "File deleted", "filename": filepath}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    print("\n🎵 Gabriel Music Manager")
    print("=" * 40)
    print("Open http://localhost:8765 in your browser")
    print("=" * 40 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")


if __name__ == "__main__":
    main()
