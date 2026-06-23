import yaml
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("config/prompts")


# Hardcoded internal-command rule appended to every assembled system
# prompt. Lives in code (not appends.yml) so users cant accidentally
# delete it and break mid-session steering from the WebUI or plugins.
_SYSTEM_INSTRUCTION_RULE = (
    "INTERNAL COMMANDS: Any user message that begins with \"SYSTEM INSTRUCTION:\" "
    "(or with \"Personality update\") is an internal control signal, not something "
    "a real person said. Silently obey what comes after the colon. Never read it "
    "out loud, never repeat it, never quote it, never paraphrase it, never say the "
    "words \"system instruction\". Do not acknowledge it (no \"okay\", \"got it\", "
    "\"understood\", \"switching\"). Just do the thing. If your reply would mention "
    "the instruction text in any form, you have failed."
)


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
        self._tools_cfg = self._load_tools_cfg()

    def _load_tools_cfg(self) -> dict:
        # Tool enable map. Reads config/tools.yml (auto generated on
        # startup by src.tools_sync). Falls back to the .example for
        # first-run before sync has had a chance to create the real file.
        # The schema is: {tools: {name: bool}, plugin_tools: {plugin: {name: bool}}}.
        # Plugin enable itself lives in plugins/<name>/plugin.yml, NOT here.
        path = Path("config/tools.yml")
        if not path.exists():
            path = Path("config/tools.yml.example")
        if not path.exists():
            return {"tools": {}, "plugin_tools": {}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(f"failed to load {path}: {exc}")
            return {"tools": {}, "plugin_tools": {}}
        return {
            "tools": data.get("tools") or {},
            "plugin_tools": data.get("plugin_tools") or {},
        }

    def reload_tools_cfg(self):
        """Reread tools.yml. Call after src.tools_sync.sync_tools_yml() so
        any newly written entries are visible to is_tool_enabled."""
        self._tools_cfg = self._load_tools_cfg()

    def is_tool_enabled(self, name: str) -> bool:
        # Check the built-in section first, then walk every plugin's
        # sub block. Anything not listed defaults to enabled, otherwise
        # an upgrade that adds a new tool would silently disable it.
        tools_map = self._tools_cfg.get("tools", {}) or {}
        if name in tools_map:
            return bool(tools_map[name])
        for _pname, sub in (self._tools_cfg.get("plugin_tools", {}) or {}).items():
            if isinstance(sub, dict) and name in sub:
                return bool(sub[name])
        return True

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
    def app_name(self):
        return self.get("app_name", default="Gabriel")

    @property
    def backend(self):
        # which brain runs the show. gemini_live = cloud websocket (default,
        # the og setup). local = LM Studio + parakeet.cpp STT + external TTS,
        # offline aside from the small flash-lite sub-agents we keep around for
        # summaries.
        val = (self.get("backend", default="gemini_live") or "gemini_live").lower()
        if val not in ("gemini_live", "local"):
            logger.warning(f"unknown backend '{val}', defaulting to gemini_live")
            return "gemini_live"
        return val

    # local backend (LM Studio + parakeet.cpp) settings
    @property
    def local_llm_base_url(self):
        return self.get("local", "llm", "base_url", default="http://localhost:1234/v1").rstrip("/")

    @property
    def local_llm_model(self):
        # whatever model identifier LM Studio shows. usually the file/repo name.
        return self.get("local", "llm", "model", default="local-model")

    @property
    def local_llm_api_key(self):
        # LM Studio ignores this but the openai-compat path requires a value
        return self.get("local", "llm", "api_key", default="lm-studio")

    @property
    def local_llm_temperature(self):
        return self.get("local", "llm", "temperature", default=0.8)

    @property
    def local_llm_top_p(self):
        return self.get("local", "llm", "top_p", default=0.95)

    @property
    def local_llm_max_tokens(self):
        return self.get("local", "llm", "max_tokens", default=1024)

    @property
    def local_llm_history_messages(self):
        # how many prior user/assistant turns to keep in the rolling context.
        return self.get("local", "llm", "history_messages", default=30)

    @property
    def local_llm_request_timeout(self):
        return self.get("local", "llm", "request_timeout", default=120)

    @property
    def local_llm_max_tool_iterations(self):
        # safety net so a misbehaving model can't loop tool calls forever.
        return self.get("local", "llm", "max_tool_iterations", default=6)

    @property
    def local_vision_enabled(self):
        # send a screen capture with every user turn. requires a multimodal
        # model loaded in LM Studio (eg qwen2.5-vl).
        return bool(self.get("local", "vision", "enabled", default=False))

    @property
    def local_vision_max_size(self):
        return self.get("local", "vision", "max_size", default=768)

    @property
    def local_vision_quality(self):
        return self.get("local", "vision", "quality", default=70)

    @property
    def local_stt_model(self):
        # parakeet.cpp model name (downloaded as GGUF from HuggingFace) or a
        # path to a local .gguf. default is the streaming multilingual nemotron.
        # other options: realtime_eou_120m-v1, tdt-0.6b-v3, ctc-0.6b, rnnt-0.6b.
        return self.get("local", "stt", "model", default="nemotron-3.5-asr-streaming-0.6b")

    @property
    def local_stt_quant(self):
        # GGUF quantization to fetch. q8_0 is near lossless; q4_k is smallest.
        # options: f16, q8_0, q6_k, q5_k, q4_k.
        return self.get("local", "stt", "quant", default="q8_0")

    @property
    def local_stt_compute(self):
        # which prebuilt parakeet runtime to use: auto (vulkan when a driver is
        # present, else cpu), vulkan, or cpu.
        return self.get("local", "stt", "compute", default="auto")

    @property
    def local_stt_mode(self):
        # auto picks streaming for streaming/eou models and offline (Silero VAD
        # segmented) for the rest. force with 'streaming' or 'offline'.
        return self.get("local", "stt", "mode", default="auto")

    @property
    def local_stt_decoder(self):
        # offline decoder head for tdt/ctc/rnnt models: default, ctc, or tdt.
        # ignored by streaming models.
        return self.get("local", "stt", "decoder", default="default")

    @property
    def local_stt_language(self):
        # locale prompt for multilingual (nemotron) models, eg en, de, auto.
        # 'auto' lets the model detect. ignored by single-language models.
        return self.get("local", "stt", "language", default="auto")

    @property
    def local_stt_min_speech_ms(self):
        # ignore blips shorter than this, stops random keyboard clicks from
        # triggering a full LLM turn.
        return self.get("local", "stt", "min_speech_ms", default=400)

    @property
    def local_stt_max_utterance_ms(self):
        # hard cap so a noisy room doesn't accumulate forever.
        return self.get("local", "stt", "max_utterance_ms", default=30000)

    @property
    def local_stt_pre_roll_ms(self):
        # keep a small ring of audio from before VAD triggered so we don't
        # clip the first phoneme.
        return self.get("local", "stt", "pre_roll_ms", default=300)

    @property
    def local_stt_external_provider(self):
        # name of a plugin-registered STT/ASR provider (ctx.register_stt).
        # when set, the local backend uses it instead of parakeet. empty
        # / unset means the built in Silero VAD + parakeet pipeline.
        return self.get("local", "stt", "external_provider", default=None)

    @property
    def conversation_logging_enabled(self):
        # off by default for privacy. flip on in config.yml under
        # privacy.save_conversations: true to write JSON transcripts
        # of every Gemini Live session to data/conversations/.
        return bool(self.get("privacy", "save_conversations", default=False))

    @property
    def log_level(self):
        # logging.<LEVEL> string from config. accepts debug/info/warn/error.
        # set to "DEBUG" if you want every chatty third party log back.
        return str(self.get("logging", "level", default="INFO") or "INFO")

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

        # Plugins can inject extra context via
        # ctx.register_prompt_contributor(). Failures are swallowed inside
        # collect_prompt_contributions so a broken plugin can't kill the prompt.
        try:
            from src.plugins import collect_prompt_contributions
            for extra in collect_prompt_contributions():
                if extra:
                    parts.append(extra)
        except Exception as e:
            logger.warning(f"plugin prompt contributors failed: {e}")

        # Always-on internal command handling rule. Baked into the code
        # so users cant accidentally delete it from appends.yml and
        # break mid-session steering. Goes last so the rule lands at
        # the bottom of the assembled prompt where the model weighs it
        # most heavily.
        parts.append(_SYSTEM_INSTRUCTION_RULE)

        return "\n\n".join(parts)

    @property
    def voice(self):
        return self.get("gemini", "voice", default="Kore")

    @property
    def vad_mode(self):
        """VAD mode: 'auto' (Gemini server-side) or 'silero' (local Silero VAD model).
        Also supports legacy 'disabled: true' which maps to 'silero'."""
        mode = self.get("gemini", "vad", "mode")
        if mode:
            return mode.lower()
        # Legacy compat: disabled=true maps to silero mode
        if self.get("gemini", "vad", "disabled", default=False):
            return "silero"
        return "auto"

    @property
    def vad_disabled(self):
        """True when using client-side VAD (Silero). Used internally."""
        return self.vad_mode == "silero"

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
    def vad_silero_threshold(self):
        """Speech probability threshold for Silero VAD (0.0-1.0). Default 0.5."""
        return self.get("gemini", "vad", "silero_threshold", default=0.5)

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
    def google_search_enabled(self):
        val = self.get("gemini", "google_search")
        if val is None:
            return not self.is_31_model
        return val

    @property
    def thinking_budget(self):
        return self.get("gemini", "thinking", "budget")

    @property
    def thinking_level(self):
        return self.get("gemini", "thinking", "level")

    @property
    def thinking_include_thoughts(self):
        return self.get("gemini", "thinking", "include_thoughts", default=False)

    @property
    def is_31_model(self):
        """Check if current model is a Gemini 3.1 Live model."""
        return "3.1" in self.model and "live" in self.model.lower()

    @property
    def session_error_threshold(self):
        """How many consecutive errors (e.g. 1007) before clearing session handle. Default 1."""
        return self.get("gemini", "session", "error_threshold", default=1)

    @property
    def session_replay_messages(self):
        """Number of recent user/assistant messages to replay as context on fresh reconnect."""
        return self.get("gemini", "session", "replay_messages", default=10)

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
    def custom_compression_enabled(self):
        return self.get("gemini", "custom_compression", "enabled", default=False)

    @property
    def custom_compression_trigger_tokens(self):
        return self.get("gemini", "custom_compression", "trigger_tokens", default=100000)

    @property
    def custom_compression_model(self):
        return self.get("gemini", "custom_compression", "model", default="gemini-3.1-flash-lite")

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
    def chatbox_rate_limiter_enabled(self):
        return self.get("vrchat", "chatbox_rate_limiter", "enabled", default=True)

    @property
    def chatbox_rate_limit_capacity(self):
        return self.get("vrchat", "chatbox_rate_limiter", "capacity", default=5)

    @property
    def chatbox_rate_limit_window_seconds(self):
        return self.get("vrchat", "chatbox_rate_limiter", "window_seconds", default=5.0)

    @property
    def chatbox_rate_limit_safety_margin_seconds(self):
        return self.get("vrchat", "chatbox_rate_limiter", "safety_margin_seconds", default=0.1)

    @property
    def chatbox_legacy_rate_limit_seconds(self):
        return self.get("vrchat", "chatbox_rate_limiter", "legacy_min_interval_seconds", default=1.27)

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
    def vision_media_resolution(self):
        """Media resolution for Live API vision. Auto-defaults to LOW for 3.1 models to save tokens."""
        val = self.get("vision", "media_resolution")
        if val is not None:
            return val
        return "low" if self.is_31_model else None

    @property
    def vision_pause_on_output(self):
        return self.get("vision", "pause_on_output", default=True)

    @property
    def vision_pause_on_idle(self):
        return self.get("vision", "pause_on_idle", default=True)

    @property
    def vision_idle_interval(self):
        """Seconds between vision frames when idle. Slows down instead of stopping entirely."""
        return self.get("vision", "idle_interval", default=15.0)

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
    def tts_hoppou_enabled(self):
        return self.tts_provider == "hoppou"

    @property
    def tts_chirp3_hd_enabled(self):
        return self.tts_provider == "chirp3_hd"

    @property
    def tts_tiktok_enabled(self):
        return self.tts_provider == "tiktok"

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
    def vrchat_group_id(self):
        return self.get("vrchat_api", "group_id", default="")

    @property
    def tts_switchable_providers(self):
        return self.get("tts", "switchable_providers", default=["gemini"])

    @property
    def emotion_enabled(self):
        return self.get("emotions", "enabled", default=True)

    @property
    def emotion_config(self):
        return self.get("emotions", default={}) or {}

    @property
    def thinking_sound_enabled(self):
        return self.get("audio", "thinking_sound", "enabled", default=False)

    @property
    def thinking_sound_on_thinking(self):
        return self.get("audio", "thinking_sound", "on_thinking", default=True)

    @property
    def thinking_sound_on_recall(self):
        return self.get("audio", "thinking_sound", "on_recall", default=True)

    @property
    def thinking_sound_file(self):
        return self.get("audio", "thinking_sound", "file", default="sfx/thinking.wav")

    @property
    def thinking_sound_volume(self):
        return self.get("audio", "thinking_sound", "volume", default=30)

    @property
    def thinking_sound_fade_in_ms(self):
        return self.get("audio", "thinking_sound", "fade_in_ms", default=500)

    @property
    def thinking_sound_fade_out_ms(self):
        return self.get("audio", "thinking_sound", "fade_out_ms", default=800)

    @property
    def obs_enabled(self):
        return self.get("obs", "enabled", default=False)

    @property
    def discord_bot_enabled(self):
        return self.get("discord_bot", "enabled", default=False)

    @property
    def music_gen_enabled(self):
        return self.get("music_gen", "enabled", default=False)

    @property
    def web_search_enabled(self):
        return self.get("web_search", "enabled", default=False)

    @property
    def social_enabled(self):
        return self.get("social", "enabled", default=False)
