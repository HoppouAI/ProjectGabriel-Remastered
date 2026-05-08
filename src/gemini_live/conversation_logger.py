"""Conversation transcript logger for Gemini Live sessions.

Disabled by default for privacy. Flip on via privacy.save_conversations
in config.yml to write per session JSON transcripts to data/conversations/.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CONVERSATION_DIR = Path("data/conversations")


class ConversationLogger:
    """Logs conversation history to JSON files per session. Thread-safe, non-blocking.
    Disabled by default for privacy, flip on via privacy.save_conversations in config.yml.
    When disabled all methods are no-ops and nothing is written to disk."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._entries = []
        self._system_instruction = None
        self._session_start = datetime.now()
        self._lock = threading.Lock()
        self._pending_user_idx = None
        self._file_path = None
        if self.enabled:
            CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)
            self._file_path = CONVERSATION_DIR / f"{self._session_start.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    def set_system_instruction(self, text: str):
        if not self.enabled:
            return
        self._system_instruction = text
        self._save_async()

    def stream_user_message(self, text: str):
        """Update user message in-place or create new entry. Called on each transcription event.
        Only updates in-memory entries - disk writes happen at turn boundaries."""
        if not self.enabled:
            return
        text = text.strip()
        if not text:
            return
        with self._lock:
            if self._pending_user_idx is not None and self._pending_user_idx < len(self._entries):
                self._entries[self._pending_user_idx]["content"] = text
            else:
                self._pending_user_idx = len(self._entries)
                self._entries.append({
                    "role": "user",
                    "content": text,
                    "timestamp": datetime.now().isoformat(),
                })

    def finalize_user_message(self):
        """Reset pending index so next stream call creates a new entry."""
        if not self.enabled:
            return
        self._pending_user_idx = None

    def add_user_message(self, text: str):
        if not self.enabled:
            return
        text = text.strip()
        if not text:
            return
        self.finalize_user_message()
        with self._lock:
            self._entries.append({
                "role": "user",
                "content": text,
                "timestamp": datetime.now().isoformat(),
            })
        self._save_async()

    def add_assistant_message(self, text: str):
        if not self.enabled:
            return
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._entries.append({
                "role": "assistant",
                "content": text,
                "timestamp": datetime.now().isoformat(),
            })
        self._save_async()

    def add_tool_call(self, name: str, args: dict):
        if not self.enabled:
            return
        with self._lock:
            self._entries.append({
                "role": "tool_call",
                "name": name,
                "arguments": args,
                "timestamp": datetime.now().isoformat(),
            })
        self._save_async()

    def add_tool_response(self, name: str, response: dict):
        if not self.enabled:
            return
        with self._lock:
            self._entries.append({
                "role": "tool_response",
                "name": name,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            })
        self._save_async()

    def _save_async(self):
        if not self.enabled or self._file_path is None:
            return
        with self._lock:
            data = {
                "session_start": self._session_start.isoformat(),
                "system_instruction": self._system_instruction,
                "messages": list(self._entries),
            }
        threading.Thread(target=self._write_file, args=(data,), daemon=True).start()

    def get_recent_entries(self, count=10):
        """Return last N user/assistant entries for context replay."""
        with self._lock:
            relevant = [e for e in self._entries if e["role"] in ("user", "assistant")]
            return relevant[-count:]

    def _write_file(self, data):
        try:
            self._file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save conversation log: {e}")
