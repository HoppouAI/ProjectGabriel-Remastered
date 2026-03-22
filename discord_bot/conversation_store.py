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

    def get_turns(self, channel_id, count=15, channel_info=""):
        """Get recent messages as structured turns for incremental content updates.

        Returns a list of dicts with 'role' ('user'/'model') and 'text'.
        Consecutive same-role messages are merged.
        """
        channel_id = str(channel_id)
        if channel_id not in self._conversations:
            self.load(channel_id)

        entries = self._conversations.get(channel_id, [])[-count:]
        if not entries:
            return []

        turns = []
        for entry in entries:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            username = entry.get("username", "")

            if role == "user":
                prefix = f"{username}: " if username else ""
                text = f"{prefix}{content}"
                gemini_role = "user"
            elif role == "assistant":
                text = content
                gemini_role = "model"
            elif role == "tool_call":
                text = f"[Tool call: {entry.get('name', '?')}]"
                gemini_role = "model"
            else:
                continue

            # Merge consecutive same-role turns
            if turns and turns[-1]["role"] == gemini_role:
                turns[-1]["text"] += f"\n{text}"
            else:
                turns.append({"role": gemini_role, "text": text})

        # Prepend channel info to first user turn if present
        if channel_info and turns:
            for turn in turns:
                if turn["role"] == "user":
                    turn["text"] = f"[CHANNEL: {channel_info}]\n{turn['text']}"
                    break

        return turns

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
