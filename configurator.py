"""
ProjectGabriel - Onboarding Configurator
Launches a web-based setup wizard that writes config.yml.
Run after setup.bat installs dependencies.
"""

import http.server
import json
import os
import sys
import threading
import urllib.request
import webbrowser
from collections.abc import MutableMapping
from pathlib import Path

import yaml
from ruamel.yaml import YAML

PORT = 8769
CONFIG_EXAMPLE = Path("config.yml.example")
CONFIG_OUTPUT = Path("config.yml")
PROMPTS_DIR = Path("config/prompts")
PROMPTS_OUTPUT = PROMPTS_DIR / "prompts.yml"
APPENDS_EXAMPLE = PROMPTS_DIR / "appends.yml.example"
APPENDS_OUTPUT = PROMPTS_DIR / "appends.yml"
ONBOARDING_HTML = Path("onboarding/index.html")


def get_audio_devices():
    """Try to list audio devices via PyAudio."""
    devices = []
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": info.get("name", f"Device {i}"),
                "max_input": info.get("maxInputChannels", 0),
                "max_output": info.get("maxOutputChannels", 0),
            })
        p.terminate()
    except Exception:
        pass
    return devices


def load_defaults():
    """Load config.yml.example as the default config."""
    if CONFIG_EXAMPLE.exists():
        with open(CONFIG_EXAMPLE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_existing_config():
    """Load existing config.yml values (plain dict, no comments)."""
    if CONFIG_OUTPUT.exists():
        with open(CONFIG_OUTPUT, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_update(base, updates):
    """Recursively update base with values from updates, preserving structure."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, MutableMapping):
            _deep_update(base[key], value)
        else:
            base[key] = value


def save_config(user_values):
    """Save config preserving comments from the example template."""
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True

    # Always start from the example template (has all comments)
    with open(CONFIG_EXAMPLE, "r", encoding="utf-8") as f:
        data = yaml_rt.load(f)

    # If existing config.yml has values, merge them first (preserves settings the UI doesn't show)
    if CONFIG_OUTPUT.exists():
        existing = load_existing_config()
        if existing:
            _deep_update(data, existing)

    # Apply user's new values on top
    _deep_update(data, user_values)

    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        yaml_rt.dump(data, f)


def save_prompts(prompt_data):
    """Add or update a prompt entry in prompts.yml, preserving existing entries."""
    if not prompt_data:
        return
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True

    prompt_name = prompt_data["name"]
    entry = {
        "name": prompt_data["charName"],
        "description": prompt_data["desc"],
        "prompt": prompt_data["prompt"],
    }

    if PROMPTS_OUTPUT.exists():
        with open(PROMPTS_OUTPUT, "r", encoding="utf-8") as f:
            data = yaml_rt.load(f)
        if data is None:
            data = {}
        # Check if file only has the unchanged "default" from the example
        is_stock_default = (
            prompt_name == "default"
            or (list(data.keys()) == ["default"] and data["default"].get("name") == "Your Character Name")
        )
        if is_stock_default:
            # Overwrite the default entry
            data["default"] = entry
        else:
            # Add or update the named entry
            data[prompt_name] = entry
    elif Path("config/prompts/prompts.yml.example").exists():
        # Start from the example template for comments
        with open("config/prompts/prompts.yml.example", "r", encoding="utf-8") as f:
            data = yaml_rt.load(f)
        if data is None:
            data = {}
        if prompt_name == "default":
            data["default"] = entry
        else:
            data[prompt_name] = entry
    else:
        data = {prompt_name: entry}

    with open(PROMPTS_OUTPUT, "w", encoding="utf-8") as f:
        yaml_rt.dump(data, f)


def save_appends(appends_data):
    """Copy appends.yml.example to appends.yml, replacing appearance/details."""
    if not appends_data:
        return
    if not APPENDS_EXAMPLE.exists():
        return
    content = APPENDS_EXAMPLE.read_text(encoding="utf-8")
    if appends_data.get("appearance"):
        content = content.replace(
            "Describe your avatar's appearance here.",
            appends_data["appearance"],
        )
    if appends_data.get("details"):
        content = content.replace(
            "Add personal details, friends, backstory here.",
            appends_data["details"],
        )
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(APPENDS_OUTPUT, "w", encoding="utf-8") as f:
        f.write(content)


class ConfigHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            if ONBOARDING_HTML.exists():
                html = ONBOARDING_HTML.read_text(encoding="utf-8")
                self._html_response(html)
            else:
                self._json_response({"error": "onboarding/index.html not found"}, 404)

        elif self.path == "/api/defaults":
            defaults = load_defaults()
            self._json_response(defaults)

        elif self.path == "/api/audio-devices":
            devices = get_audio_devices()
            self._json_response(devices)

        elif self.path == "/api/check-config":
            self._json_response({"exists": CONFIG_OUTPUT.exists()})

        elif self.path == "/api/check-prompts":
            self._json_response({"exists": PROMPTS_OUTPUT.exists()})

        elif self.path == "/api/load-config":
            existing = load_existing_config()
            if existing:
                self._json_response({"exists": True, "config": existing})
            else:
                self._json_response({"exists": False})

        elif self.path == "/api/shutdown":
            self._json_response({"success": True})
            threading.Thread(target=lambda: (server.shutdown()), daemon=True).start()

        else:
            # Serve static files from onboarding/
            MIME_TYPES = {".css": "text/css", ".js": "application/javascript", ".html": "text/html"}
            safe_path = Path("onboarding") / Path(self.path.lstrip("/"))
            if safe_path.exists() and safe_path.is_file() and Path("onboarding").resolve() in safe_path.resolve().parents:
                ext = safe_path.suffix.lower()
                content_type = MIME_TYPES.get(ext, "application/octet-stream")
                body = safe_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                config_values = data.get("config", data)
                prompt_data = data.get("prompt")
                appends_data = data.get("appends")

                if prompt_data:
                    # Point config to the saved prompt key name
                    config_values.setdefault("gemini", {})["prompt"] = prompt_data["name"]

                save_config(config_values)

                saved_files = [str(CONFIG_OUTPUT.resolve())]
                if prompt_data:
                    save_prompts(prompt_data)
                    saved_files.append(str(PROMPTS_OUTPUT.resolve()))
                if appends_data:
                    save_appends(appends_data)
                    saved_files.append(str(APPENDS_OUTPUT.resolve()))

                self._json_response({
                    "success": True,
                    "config_path": str(CONFIG_OUTPUT.resolve()),
                    "saved_files": saved_files,
                })
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif self.path == "/api/generate-prompt":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                api_key = data["api_key"]
                system_prompt = data["system"]
                user_msg = data["user"]

                payload = json.dumps({
                    "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "generationConfig": {"temperature": 1.0, "maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "low"}},
                }).encode()

                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode())

                text = result["candidates"][0]["content"]["parts"][0]["text"]
                self._json_response({"text": text})
            except urllib.error.HTTPError as e:
                err_body = e.read().decode() if e.readable() else str(e)
                self._json_response({"error": f"API error {e.code}: {err_body}"}, 500)
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif self.path == "/api/shutdown":
            self._json_response({"success": True})
            threading.Thread(target=lambda: (server.shutdown()), daemon=True).start()

        else:
            self.send_error(404)


def main():
    global server

    if not CONFIG_EXAMPLE.exists():
        print("Error: config.yml.example not found. Run from project root.")
        sys.exit(1)

    if not ONBOARDING_HTML.exists():
        print("Error: onboarding/index.html not found.")
        sys.exit(1)

    server = http.server.HTTPServer(("127.0.0.1", PORT), ConfigHandler)
    url = f"http://127.0.0.1:{PORT}"

    print()
    print("  ====================================================")
    print("       Project Gabriel - Configuration Wizard")
    print("  ====================================================")
    print()
    print(f"  Opening configurator at {url}")
    print("  Press Ctrl+C to cancel.")
    print()

    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    server.server_close()

    if CONFIG_OUTPUT.exists():
        print()
        print("  ====================================================")
        print("       Configuration saved successfully!")
        print("  ====================================================")
        print()
        print("  You can close this window now.")
        print()
        print("  To start Gabriel:    run.bat")
        print("  To edit config:      configurator.bat")
        print()
    else:
        print()
        print("  No config saved. Run configurator.bat again or edit config.yml manually.")
        print()


if __name__ == "__main__":
    main()
