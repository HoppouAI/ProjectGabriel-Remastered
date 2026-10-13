import yaml
import logging
from datetime import datetime
from pathlib import Path

from src.model_caps import (
    TOOL_SCHEDULING,
    TOOL_TYPE_DEFAULT,
    TOOL_TYPES,
    CompatNote,
    check_alpha_features,
    resolve_model,
    resolve_thinking,
    resolve_tool_behavior,
    resolve_tool_scheduling,
)
from src.tool_call_types import default_call_type

logger = logging.getLogger(__name__)


def _summarise(names, limit=4):
    # keep compat notes to one readable line when a model has loads of tools
    names = sorted(names)
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", and {len(names) - limit} more"

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
        self._grounding_disabled = False
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
        self._tool_entry_cache = {}

    def is_tool_enabled(self, name: str) -> bool:
        return self._tool_settings(name)["enabled"]

    def _raw_tool_entry(self, name: str):
        # returns whatever tools.yml holds for this tool: bool, dict, or None. cached, its hot.
        cache = getattr(self, "_tool_entry_cache", None)
        if cache is None:
            cache = {}
            self._tool_entry_cache = cache
        if name in cache:
            return cache[name]

        found = None
        tools_map = self._tools_cfg.get("tools", {}) or {}
        if name in tools_map:
            found = tools_map[name]
        else:
            for _pname, sub in (self._tools_cfg.get("plugin_tools", {}) or {}).items():
                if isinstance(sub, dict) and name in sub:
                    found = sub[name]
                    break
        cache[name] = found
        return found

    def _tool_settings(self, name: str) -> dict:
        """Normalised per-tool settings. takes both the bool shorthand and the long form."""
        raw = self._raw_tool_entry(name)
        if raw is None:
            # unlisted defaults to on so an upgrade never silently drops a tool
            return {"enabled": True, "type": TOOL_TYPE_DEFAULT, "scheduling": None}
        if isinstance(raw, dict):
            return {
                "enabled": bool(raw.get("enabled", True)),
                "type": str(raw.get("type") or TOOL_TYPE_DEFAULT).strip().lower(),
                "scheduling": raw.get("scheduling"),
            }
        return {"enabled": bool(raw), "type": TOOL_TYPE_DEFAULT, "scheduling": None}

    def tool_type(self, name: str) -> str:
        return self._tool_settings(name)["type"]

    def tool_scheduling(self, name: str):
        return self._tool_settings(name)["scheduling"]

    def _effective_call_type(self, name: str, caps):
        # tools.yml wins, built in defaults only fill in where the model leaves it optional
        typ, sched = self.tool_type(name), self.tool_scheduling(name)
        if caps.async_tools != "optional":
            return typ, sched
        default_type, default_sched = default_call_type(name)
        if typ == TOOL_TYPE_DEFAULT and default_type:
            typ = default_type
        if sched is None and typ == default_type:
            sched = default_sched
        return typ, sched

    def behavior_for_tool(self, name: str):
        """FunctionDeclaration.behavior value for this tool, or None to leave alone."""
        caps = self.model_capabilities
        typ, _sched = self._effective_call_type(name, caps)
        behavior, _reason = resolve_tool_behavior(caps, typ)
        return behavior

    def scheduling_for_tool(self, name: str):
        """FunctionResponse.scheduling value for this tool, or None for no hint."""
        caps = self.model_capabilities
        typ, sched = self._effective_call_type(name, caps)
        behavior, _reason = resolve_tool_behavior(caps, typ)
        scheduling, _reason = resolve_tool_scheduling(caps, sched, behavior)
        return scheduling

    def tool_runs_blocking(self, name: str) -> bool:
        """True when the model sits and waits on this tool's result."""
        caps = self.model_capabilities
        if caps.async_tools == "none":
            return True
        return self.behavior_for_tool(name) == "BLOCKING"

    def tool_compat_notes(self):
        """Explain any tools.yml setting the current model cannot honor, grouped per problem."""
        caps = self.model_capabilities
        invalid_types = {}
        invalid_sched = {}

        for name in self._configured_tool_names():
            settings = self._tool_settings(name)
            if not settings["enabled"]:
                # dont nag about settings on a tool the model cant even see
                continue
            cfg = settings
            _behavior, reason = resolve_tool_behavior(caps, cfg["type"])
            if reason and reason.startswith("'"):
                invalid_types.setdefault(str(cfg["type"]), []).append(name)
                continue
            _sched, sreason = resolve_tool_scheduling(caps, cfg["scheduling"], _behavior)
            if sreason and sreason.startswith("'"):
                invalid_sched.setdefault(str(cfg["scheduling"]), []).append(name)

        notes = []
        buckets = self._bucket_tool_names(caps)
        if buckets["non_blocking_unsupported"]:
            names = buckets["non_blocking_unsupported"]
            notes.append(CompatNote(
                "tool type",
                f"{len(names)} tool(s) are set to non_blocking but {caps.name} only "
                f"supports blocking function calls, running them blocking "
                f"({_summarise(names)})",
            ))
        if buckets["forced_async"]:
            names = buckets["forced_async"]
            notes.append(CompatNote(
                "tool type",
                f"{len(names)} tool(s) are set to blocking but {caps.name} rejects "
                f"blocking function calls outright, running them non_blocking "
                f"({_summarise(names)})",
            ))
        if buckets["not_async"]:
            names = buckets["not_async"]
            notes.append(CompatNote(
                "tool scheduling",
                f"{len(names)} tool(s) set a scheduling but run synchronously, so it is "
                f"ignored ({_summarise(names)})",
            ))
        if buckets["no_scheduling"]:
            names = buckets["no_scheduling"]
            notes.append(CompatNote(
                "tool scheduling",
                f"{caps.name} does not support function scheduling, ignoring it for "
                f"{len(names)} tool(s) ({_summarise(names)})",
            ))
        for bad, names in invalid_types.items():
            notes.append(CompatNote(
                "tool type",
                f"'{bad}' is not a valid type, expected one of "
                f"{', '.join(TOOL_TYPES)} ({_summarise(names)})",
            ))
        for bad, names in invalid_sched.items():
            notes.append(CompatNote(
                "tool scheduling",
                f"'{bad}' is not a valid scheduling, expected one of "
                f"{', '.join(TOOL_SCHEDULING)} ({_summarise(names)})",
            ))
        return notes

    def _configured_tool_names(self):
        names = set((self._tools_cfg.get("tools", {}) or {}).keys())
        for _pname, sub in (self._tools_cfg.get("plugin_tools", {}) or {}).items():
            if isinstance(sub, dict):
                names.update(sub.keys())
        return names

    def _bucket_tool_names(self, caps):
        # sort every configured tool into the reason bucket it belongs in
        buckets = {
            "non_blocking_unsupported": [],
            "forced_async": [],
            "not_async": [],
            "no_scheduling": [],
        }
        for name in self._configured_tool_names():
            cfg = self._tool_settings(name)
            if not cfg["enabled"]:
                continue
            behavior, reason = resolve_tool_behavior(caps, cfg["type"])
            if reason == "blocking_only":
                buckets["non_blocking_unsupported"].append(name)
                continue
            if reason == "forced_async":
                buckets["forced_async"].append(name)
                continue
            if reason and reason.startswith("'"):
                continue
            _sched, sreason = resolve_tool_scheduling(caps, cfg["scheduling"], behavior)
            if sreason in ("not_async", "no_scheduling"):
                buckets[sreason].append(name)
        return buckets

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
    def local_llm_concise_reasoning(self):
        # append a brevity instruction so reasoning models stop thinking for
        # ages and keep replies short.
        return bool(self.get("local", "llm", "concise_reasoning", default=False))

    @property
    def local_llm_reasoning_effort(self):
        # optional openai-style reasoning_effort passed to the api for models
        # that support it (minimal/low/medium/high). empty = don't send it.
        return (self.get("local", "llm", "reasoning_effort", default="") or "").strip().lower()

    @property
    def local_llm_dynamic_tools(self):
        # only send a relevant slice of the ~100 tools each turn instead of all
        # of them, the model pulls in more on demand via the findTools meta tool.
        # saves a big chunk of prompt tokens on the local backend.
        return bool(self.get("local", "llm", "dynamic_tools", default=False))

    @property
    def local_llm_dynamic_tools_max(self):
        # how many query-relevant tools to auto-include per turn on top of the
        # always-on core set.
        return int(self.get("local", "llm", "dynamic_tools_max", default=8))

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
        # path to a local .gguf. default is the offline multilingual tdt v3.
        # other options: nemotron-3.5-asr-streaming-0.6b, realtime_eou_120m-v1,
        # ctc-0.6b, rnnt-0.6b.
        return self.get("local", "stt", "model", default="tdt-0.6b-v3")

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

    @property
    def key_count(self):
        return len(self._keys)

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

        # generated body: how to express himself with it, and how it relates
        # to the older canned animation tool when both are on
        if self.motion_enabled:
            try:
                from src.motion_expression import build_instruction
                block = build_instruction(self)
                if block:
                    parts.append(block)
            except Exception as e:
                logger.warning(f"motion instruction failed: {e}")

        # when dynamic tools are on the model only sees the meta tools, so
        # drop in the catalog of what it can reach via searchTools/executeTool
        if self.gemini_dynamic_tools:
            try:
                from src.tools.meta_router import build_meta_prompt
                catalog = build_meta_prompt(self)
                if catalog:
                    parts.append(catalog)
            except Exception as e:
                logger.warning(f"meta tool catalog failed: {e}")

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
    def gemini_dynamic_tools(self):
        return self.get("gemini", "dynamic_tools", default=False)

    @property
    def model_capabilities(self):
        """What the configured Live model actually accepts. See src/model_caps.py."""
        return resolve_model(self.model)

    def resolve_thinking_config(self):
        """(ThinkingConfig kwargs or None, CompatNote list) for the current model.

        Applies the user's thinking settings to whatever this model supports,
        dropping or nudging anything it would reject instead of erroring out.
        """
        return resolve_thinking(
            self.model_capabilities,
            enabled=self.thinking_enabled,
            budget=self.thinking_budget,
            level=self.thinking_level,
            include_thoughts=self.thinking_include_thoughts,
        )

    def alpha_feature_notes(self):
        """CompatNote list for affective dialog / proactivity on this model."""
        return check_alpha_features(
            self.model_capabilities,
            affective_dialog=self.enable_affective_dialog,
            proactivity=self.proactivity,
        )

    def disable_search_grounding(self):
        """Called when grounding 1011s. Drops the built-in search tool so the app's own
        webSearch tool takes over, since that one doesnt touch the grounding quota."""
        self._grounding_disabled = True

    @property
    def google_search_enabled(self):
        if self._grounding_disabled:
            return False
        val = self.get("gemini", "google_search")
        if val is None:
            return self.model_capabilities.google_search
        return bool(val)

    @property
    def thinking_enabled(self):
        return bool(self.get("gemini", "thinking", "enabled", default=True))

    @property
    def thinking_budget(self):
        # 2.5 models only. token count, higher = more reasoning. ignored on 3.x.
        return self.get("gemini", "thinking", "budget")

    @property
    def thinking_level(self):
        # 3.x models only. minimal/low/medium/high, though 3.8 extended thinking
        # rejects minimal and gets nudged up to low automatically.
        return self.get("gemini", "thinking", "level")

    @property
    def thinking_include_thoughts(self):
        return bool(self.get("gemini", "thinking", "include_thoughts", default=False))

    @property
    def uses_realtime_text(self):
        """True when text must go out via send_realtime_input, not send_client_content."""
        return self.model_capabilities.text_input == "realtime"

    @property
    def is_31_model(self):
        # legacy alias kept so plugins and older call sites dont break.
        # true means "new gen live model": realtime text input, no alpha features.
        return self.uses_realtime_text

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
    def motion_enabled(self):
        return self.get("motion", "enabled", default=False)

    @property
    def motion_server_host(self):
        return self.get("motion", "server_host", default="127.0.0.1")

    @property
    def motion_server_port(self):
        return self.get("motion", "server_port", default=8765)

    @property
    def motion_walk_full_speed(self):
        return self.get("motion", "walk_full_speed", default=2.0)

    @property
    def motion_run_full_speed(self):
        return self.get("motion", "run_full_speed", default=4.0)

    @property
    def motion_turn_full_rate(self):
        return self.get("motion", "turn_full_rate", default=1.8)

    @property
    def motion_pose_tracking(self):
        return self.get("motion", "pose_tracking", default=False)

    @property
    def motion_pose_monitor(self):
        return self.get("motion", "pose_monitor", default=1)

    @property
    def motion_navigation(self):
        mode = str(self.get("motion", "navigation", default="pause")).lower()
        return mode if mode in ("pause", "model") else "pause"

    @property
    def motion_expression(self):
        return self.get("motion", "expression", default={}) or {}

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
        """Media resolution for Live API vision. Auto-defaults to LOW on token hungry models."""
        val = self.get("vision", "media_resolution")
        if val is not None:
            return val
        return "low" if self.model_capabilities.vision_token_cap else None

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
