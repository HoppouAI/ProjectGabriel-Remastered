import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationStore:
    """Persists per-channel conversation history to JSON files.

    Each channel gets a separate JSON file. Messages are appended
    and periodically saved. On startup, recent history is loaded
    to provide context to the Gemini session.
    """

    def __init__(self, save_dir="discord_bot/data/conversations"):
        self._dir = Path(save_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conversations = {}  # channel_id -> list of entries

    def _file_for(self, channel_id):
        return self._dir / f"{channel_id}.json"

    def load(self, channel_id, limit=50):
        """Load recent conversation history for a channel."""
        channel_id = str(channel_id)
        path = self._file_for(channel_id)
        if not path.exists():
            self._conversations[channel_id] = []
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data.get("messages", [])[-limit:]
            self._conversations[channel_id] = entries
            return entries
        except Exception as e:
            logger.error(f"Failed to load conversation {channel_id}: {e}")
            self._conversations[channel_id] = []
            return []

    def add_message(self, channel_id, role, content, username=None, attachments=None):
        """Add a message to the conversation history."""
        channel_id = str(channel_id)
        if channel_id not in self._conversations:
            self.load(channel_id)

        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if username:
            entry["username"] = username
        if attachments:
            entry["attachments"] = attachments

        with self._lock:
            self._conversations[channel_id].append(entry)

        self._save_async(channel_id)

    def add_tool_call(self, channel_id, name, args):
        channel_id = str(channel_id)
        if channel_id not in self._conversations:
            self._conversations[channel_id] = []
        with self._lock:
            self._conversations[channel_id].append({
                "role": "tool_call",
                "name": name,
                "arguments": args,
                "timestamp": datetime.now().isoformat(),
            })
        self._save_async(channel_id)

    def get_context(self, channel_id, count=15):
        """Get recent messages formatted as context for Gemini."""
        channel_id = str(channel_id)
        if channel_id not in self._conversations:
            self.load(channel_id)

        entries = self._conversations.get(channel_id, [])[-count:]
        parts = []
        for entry in entries:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            username = entry.get("username", "")
            if role == "user":
                prefix = f"{username}: " if username else ""
                parts.append(f"{prefix}{content}")
            elif role == "assistant":
                parts.append(f"You: {content}")
            elif role == "tool_call":
                parts.append(f"[Tool: {entry.get('name', '?')}]")
        return "\n".join(parts)

    def _save_async(self, channel_id):
        with self._lock:
            entries = list(self._conversations.get(channel_id, []))
        threading.Thread(
            target=self._write_file,
            args=(channel_id, entries),
            daemon=True,
        ).start()

    def _write_file(self, channel_id, entries):
        try:
            path = self._file_for(channel_id)
            data = {
                "channel_id": channel_id,
                "last_updated": datetime.now().isoformat(),
                "messages": entries,
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save conversation {channel_id}: {e}")
