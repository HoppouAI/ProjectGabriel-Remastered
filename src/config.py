import yaml
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("config/prompts")


class Config:
    def __init__(self, path="config.yml"):
        with open(path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        self._keys = [self._data["gemini"]["api_key"]]
        backup = self._data["gemini"].get("backup_keys") or []
        if backup:
            self._keys.extend(backup)
        self._key_index = 0
        self._prompts = self._load_prompts()
        self._appends = self._load_appends()
        self._voices = self._load_voices()

    def _load_prompts(self) -> dict:
        prompts_file = PROMPTS_DIR / "prompts.yml"
        if prompts_file.exists():
            with open(prompts_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_appends(self) -> list:
        appends_file = PROMPTS_DIR / "appends.yml"
        if appends_file.exists():
            with open(appends_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or []
        return []

    def _load_voices(self) -> dict:
        voices_file = Path("config/voices.yml")
        if voices_file.exists():
            with open(voices_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("voices", {})
        return {}

    def get_voice(self, voice_name: str) -> dict | None:
        return self._voices.get(voice_name)

    def list_voices(self) -> dict:
        return self._voices

    def get(self, *keys, default=None):
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    @property
    def api_key(self):
        return self._keys[self._key_index]

    def rotate_key(self):
        old_idx = self._key_index
        self._key_index = (self._key_index + 1) % len(self._keys)
        if self._key_index == old_idx:
            logger.warning("No backup keys available, reusing same key")
        else:
            logger.info(f"Rotated to API key index {self._key_index}")
        return self.api_key

    @property
    def model(self):
        return self.get("gemini", "model", default="gemini-2.5-flash-native-audio-preview-12-2025")

    @property
    def system_instruction(self):
        return self.build_system_instruction()

    def build_system_instruction(self, personality_mgr=None):
        prompt_name = self.get("gemini", "prompt", default="normal")
        raw = self._prompts.get(prompt_name, "")
        if isinstance(raw, dict):
            base = raw.get("prompt", "")
        else:
            base = str(raw) if raw else ""
        if not base:
            logger.warning(f"Prompt '{prompt_name}' not found in prompts.yml, using empty")

        parts = [base.strip()]
        personalities_text = ""
        if personality_mgr:
            personalities_text = personality_mgr.get_available_text()
        
        # Get memory content for prompt
        memories_text = ""
        if self.memory_enabled:
            try:
                from src.memory import get_memory_content_for_prompt
                memories_text = get_memory_content_for_prompt(self.prompt_memory_count)
            except Exception as e:
                logger.warning(f"Failed to get memories for prompt: {e}")
        
        for append in self._appends:
            if not append.get("enabled", True):
                continue
            content = append.get("content", "")
            content = content.replace("{date}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            content = content.replace("{available_personalities}", personalities_text)
            content = content.replace("{memories}", memories_text)
            parts.append(content.strip())

        return "\n\n".join(parts)

    @property
    def voice(self):
        return self.get("gemini", "voice", default="Kore")

    @property
    def vad_disabled(self):
        return self.get("gemini", "vad", "disabled", default=False)

    @property
    def vad_start_sensitivity(self):
        return self.get("gemini", "vad", "start_of_speech_sensitivity", default="START_SENSITIVITY_HIGH")

    @property
    def vad_end_sensitivity(self):
        return self.get("gemini", "vad", "end_of_speech_sensitivity", default="END_SENSITIVITY_HIGH")

    @property
    def vad_prefix_padding_ms(self):
        return self.get("gemini", "vad", "prefix_padding_ms", default=20)

    @property
    def vad_silence_duration_ms(self):
        return self.get("gemini", "vad", "silence_duration_ms", default=500)

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
    def enable_affective_dialog(self):
        return self.get("gemini", "enable_affective_dialog")

    @property
    def proactivity(self):
        return self.get("gemini", "proactivity")

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
    def language(self):
        return self.get("gemini", "language")

    @property
    def input_device(self):
        return self.get("audio", "input_device")

    @property
    def output_device(self):
        return self.get("audio", "output_device")

    @property
    def send_sample_rate(self):
        return self.get("audio", "send_sample_rate", default=16000)

    @property
    def receive_sample_rate(self):
        return self.get("audio", "receive_sample_rate", default=24000)

    @property
    def chunk_size(self):
        return self.get("audio", "chunk_size", default=1024)

    @property
    def osc_ip(self):
        return self.get("vrchat", "osc_ip", default="127.0.0.1")

    @property
    def osc_port(self):
        return self.get("vrchat", "osc_send_port", default=9000)

    @property
    def osc_receive_port(self):
        return self.get("vrchat", "osc_receive_port", default=9001)

    @property
    def chatbox_page_delay(self):
        return self.get("vrchat", "chatbox_page_delay", default=3.0)

    @property
    def music_dir(self):
        return self.get("music", "music_dir", default="sfx/music")

    @property
    def tracker_enabled(self):
        return self.get("yolo", "enabled", default=True)

    @property
    def face_tracker_enabled(self):
        return self.get("face_tracker", "enabled", default=False)

    @property
    def wanderer_enabled(self):
        return self.get("wanderer", "enabled", default=False)

    @property
    def vision_debug(self):
        return self.get("yolo", "vision_debug", default=False)

    @property
    def vision_debug_port(self):
        return self.get("yolo", "vision_debug_port", default=8767)

    @property
    def yolo_model_dir(self):
        return self.get("yolo", "model_dir", default="models/yolov8")

    @property
    def yolo_model_name(self):
        return self.get("yolo", "model_name", default="yolov8n.pt")

    @property
    def vision_enabled(self):
        return self.get("vision", "enabled", default=False)

    @property
    def vision_monitor(self):
        return self.get("vision", "monitor", default=1)

    @property
    def vision_interval(self):
        return self.get("vision", "interval", default=1.0)

    @property
    def vision_max_size(self):
        return self.get("vision", "max_size", default=1024)

    @property
    def vision_quality(self):
        return self.get("vision", "quality", default=80)

    @property
    def memory_enabled(self):
        return self.get("memory", "enabled", default=True)

    @property
    def prompt_memory_count(self):
        return self.get("memory", "prompt_memory_count", default=10)

    @property
    def tts_provider(self):
        return self.get("tts", "provider", default="gemini")

    @property
    def tts_qwen3_enabled(self):
        return self.tts_provider == "qwen3"

    @property
    def tts_hoppou_enabled(self):
        return self.tts_provider == "hoppou"

    @property
    def tts_chirp3_hd_enabled(self):
        return self.tts_provider == "chirp3_hd"

    @property
    def vrchat_api_username(self):
        return self.get("vrchat_api", "username", default="")

    @property
    def vrchat_api_password(self):
        return self.get("vrchat_api", "password", default="")

    @property
    def vrchat_api_allow_bio_edit(self):
        return self.get("vrchat_api", "allow_bio_edit", default=False)

    @property
    def tts_switchable_providers(self):
        return self.get("tts", "switchable_providers", default=["gemini"])

    @property
    def emotion_enabled(self):
        return self.get("emotions", "enabled", default=True)

    @property
    def emotion_config(self):
        return self.get("emotions", default={}) or {}
