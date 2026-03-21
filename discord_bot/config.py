import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BOT_DIR = Path(__file__).parent


class BotConfig:
    def __init__(self, path=None):
        if path is None:
            path = BOT_DIR / "config.yml"
        with open(path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}
        self._keys = [self._data["gemini"]["api_key"]]
        backup = self._data["gemini"].get("backup_keys") or []
        if backup:
            self._keys.extend(backup)
        self._key_index = 0

    def get(self, *keys, default=None):
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    @property
    def discord_token(self):
        return self._data.get("discord_token", "")

    @property
    def authorized_users(self):
        raw = self._data.get("authorized_users", [])
        return [str(uid) for uid in raw]

    @property
    def api_key(self):
        return self._keys[self._key_index]

    def rotate_key(self):
        old_idx = self._key_index
        self._key_index = (self._key_index + 1) % len(self._keys)
        if self._key_index != old_idx:
            logger.info(f"Rotated to API key index {self._key_index}")
        return self.api_key

    @property
    def model(self):
        return self.get("gemini", "model", default="gemini-2.5-flash-native-audio-preview-09-2025")

    @property
    def voice(self):
        return self.get("gemini", "voice", default="Puck")

    @property
    def system_prompt(self):
        return self.get("gemini", "system_prompt", default="You are a friendly AI chatting on Discord.")

    @property
    def temperature(self):
        return self.get("gemini", "temperature")

    @property
    def top_p(self):
        return self.get("gemini", "top_p")

    @property
    def top_k(self):
        return self.get("gemini", "top_k")

    @property
    def max_output_tokens(self):
        return self.get("gemini", "max_output_tokens")

    @property
    def thinking_budget(self):
        return self.get("gemini", "thinking", "budget")

    @property
    def thinking_include_thoughts(self):
        return self.get("gemini", "thinking", "include_thoughts", default=False)

    @property
    def compression_enabled(self):
        return self.get("gemini", "context_window_compression", "enabled", default=True)

    @property
    def compression_trigger_tokens(self):
        return self.get("gemini", "context_window_compression", "trigger_tokens")

    @property
    def compression_target_tokens(self):
        return self.get("gemini", "context_window_compression", "target_tokens")

    @property
    def auto_respond_dms(self):
        return self.get("behavior", "auto_respond_dms", default=True)

    @property
    def auto_respond_mentions(self):
        return self.get("behavior", "auto_respond_mentions", default=True)

    @property
    def auto_respond_channels(self):
        raw = self.get("behavior", "auto_respond_channels", default=[])
        return [str(c) for c in raw] if raw else []

    @property
    def typing_delay_ms(self):
        return self.get("behavior", "typing_delay_ms", default=1500)

    @property
    def max_message_length(self):
        return self.get("behavior", "max_message_length", default=2000)

    @property
    def response_cooldown(self):
        return self.get("behavior", "response_cooldown", default=2.0)

    @property
    def context_message_count(self):
        return self.get("behavior", "context_message_count", default=15)

    @property
    def memory_enabled(self):
        return self.get("memory", "enabled", default=True)

    @property
    def memory_key_prefix(self):
        return self.get("memory", "key_prefix", default="discord_")

    @property
    def relay_enabled(self):
        return self.get("relay", "enabled", default=True)

    @property
    def conversations_enabled(self):
        return self.get("conversations", "enabled", default=True)

    @property
    def conversations_dir(self):
        return self.get("conversations", "save_dir", default="discord_bot/data/conversations")

    @property
    def log_level(self):
        return self.get("log_level", default="INFO")
