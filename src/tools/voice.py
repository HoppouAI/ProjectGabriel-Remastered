import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class VoiceTools(BaseTool):
    tool_key = "voice"

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="setVoiceBoost",
                description="Set voice boost level for loud distorted bass-boosted yelling effect on your microphone output.\n**Invocation Condition:** Call when asked to get loud or distorted, or for comedic yelling effects.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "level": {"type": "INTEGER", "description": "0=normal voice, 1-10=increasingly loud/distorted/bass-boosted"},
                    },
                    "required": ["level"],
                },
            ),
            types.FunctionDeclaration(
                name="toggleVrchatMic",
                description="Mute or unmute the VRChat microphone.\n**Invocation Condition:** Call when asked to mute or unmute.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "muted": {"type": "STRING", "description": "Set to 'true' to mute, 'false' to unmute"},
                    },
                    "required": ["muted"],
                },
            ),
            types.FunctionDeclaration(
                name="switchTTSProvider",
                description="Switch the text-to-speech voice mid-session. Changes take effect immediately on the next spoken response. NEVER mention provider names, technology names, or internal voice IDs to the user. Just say you switched your voice.\n**Invocation Condition:** Call when asked to change voice, switch TTS, or use a different voice. Use listTTSProviders first to see what is available.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "provider": {"type": "STRING", "description": "Provider ID to switch to (e.g. 'gemini', 'chirp3_hd', 'hoppou', 'qwen3', 'tiktok')"},
                        "voice": {"type": "STRING", "description": "A custom voice name OR a built-in voice name for the provider. Optional -- uses config default if omitted. NOT supported for 'gemini' provider."},
                    },
                    "required": ["provider"],
                },
            ),
            types.FunctionDeclaration(
                name="listTTSProviders",
                description="List available voice providers and the currently active one. Internal use only -- NEVER reveal provider names or IDs to the user.\n**Invocation Condition:** Call when you need to know what providers are available before switching.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="listVoices",
                description="List custom voices available for switching. Each voice has a display name and description. When telling the user about voices, use ONLY the display_name, never the internal ID or provider name.\n**Invocation Condition:** Call when asked what voices are available, or before switching to a custom voice.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="setVoicePitch",
                description="Shift the pitch of the AI voice up or down in semitones, like a voice changer. 0 = normal pitch. Positive = higher, negative = lower. Small values (1-3) sound natural, larger values (4-12) sound dramatic.\n**Invocation Condition:** Call when asked to raise or lower pitch, sound higher or deeper, do a voice changer effect, or reset voice pitch.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "semitones": {"type": "NUMBER", "description": "Pitch shift in semitones. 0=normal, +2=slightly higher, -3=deeper, +5=chipmunk-ish, -5=deep bass. Range limited by config (default -12 to +12)."},
                    },
                    "required": ["semitones"],
                },
            ),
            types.FunctionDeclaration(
                name="toggleLowQualityMic",
                description="Toggle a hilariously bad mic quality effect. When enabled, your voice sounds like it's coming through a dollar store webcam mic from 2005 -- bitcrushed, noisy, telephone-band filtered, with random stuttery glitches. You can also tweak individual parameters to make it worse or better. Great for comedy bits.\n**Invocation Condition:** Call when asked to sound like a bad mic, cheap mic, low quality, crappy audio, discord call from 2010, or to configure/tweak the bad mic effect, or to toggle it off.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "enabled": {"type": "STRING", "description": "'true' to enable garbage mic mode, 'false' to go back to normal"},
                        "downsample": {"type": "INTEGER", "description": "Downsample factor 1-8 (higher = crunchier, default 4). 1=no downsample, 8=extremely crunchy"},
                        "bitcrush": {"type": "NUMBER", "description": "Bitcrush step 16-1024 (lower = harsher bitcrush, default 256). 64=very harsh, 512=mild"},
                        "noise": {"type": "NUMBER", "description": "White noise intensity 0-3000 (default 800). 0=no noise, 3000=overwhelmingly noisy"},
                        "glitch": {"type": "NUMBER", "description": "Glitch/stutter probability 0.0-0.2 per audio chunk (default 0.03). 0=no glitches, 0.2=very glitchy"},
                    },
                    "required": ["enabled"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "setVoiceBoost":
            self.audio.set_boost(args["level"])
            return {"result": "ok", "level": args["level"]}
        elif name == "toggleVrchatMic":
            self.osc.toggle_voice()
            return {"result": "ok"}
        elif name == "setVoicePitch":
            return self._set_pitch(args.get("semitones", 0))
        elif name == "toggleLowQualityMic":
            enabled = str(args.get("enabled", "false")).lower() == "true"
            kwargs = {}
            for param in ("downsample", "bitcrush", "noise", "glitch"):
                if param in args:
                    kwargs[param] = args[param]
            self.audio.set_low_quality(enabled, **kwargs)
            return {"result": "ok", **self.audio.get_low_quality_settings()}
        elif name == "switchTTSProvider":
            return await self._switch_tts(args.get("provider", ""), args.get("voice"))
        elif name == "listTTSProviders":
            return self._list_tts_providers()
        elif name == "listVoices":
            return self._list_voices()
        return None

    async def _switch_tts(self, provider_name, voice=None):
        provider_name = provider_name.strip().lower()
        allowed = [p.strip().lower() for p in (self.config.tts_switchable_providers if self.config else ["gemini"])]
        if provider_name not in allowed:
            return {"result": "error", "message": f"Provider '{provider_name}' not allowed. Allowed: {allowed}"}
        if not self.live_session:
            return {"result": "error", "message": "No active session"}

        voice_override = None
        if voice and self.config:
            voice_def = self.config.get_voice(voice)
            if voice_def and provider_name in voice_def:
                voice_override = voice_def[provider_name]
                logger.info(f"switchTTSProvider: using custom voice '{voice}' for {provider_name}")

        new_provider = None
        if provider_name == "gemini":
            if voice:
                return {"result": "error", "message": "Cannot change Gemini voice mid-session. Gemini voice requires a full session restart."}
        elif provider_name == "qwen3":
            from src.tts import QwenTTSProvider
            new_provider = QwenTTSProvider(self.config, voice_override=voice_override)
        elif provider_name == "hoppou":
            from src.tts import HoppouTTSProvider
            if not voice_override and voice:
                voice_override = {"voice": voice}
            new_provider = HoppouTTSProvider(self.config, voice_override=voice_override)
        elif provider_name == "chirp3_hd":
            from src.tts import Chirp3HDTTSProvider
            if not voice_override and voice:
                voice_override = {"voice": voice}
            new_provider = Chirp3HDTTSProvider(self.config, voice_override=voice_override)
        elif provider_name == "tiktok":
            from src.tts import TikTokTTSProvider
            if not voice_override and voice:
                voice_override = {"voice": voice}
            new_provider = TikTokTTSProvider(self.config, voice_override=voice_override)
        else:
            return {"result": "error", "message": f"Unknown provider: {provider_name}"}

        self.live_session.switch_tts_provider(new_provider)
        result = {"result": "ok", "provider": provider_name}
        if voice:
            result["voice"] = voice
        logger.info(f"switchTTSProvider: switched to '{provider_name}'" + (f" voice='{voice}'" if voice else ""))
        return result

    def _list_tts_providers(self):
        allowed = self.config.tts_switchable_providers if self.config else ["gemini"]
        current_tts = self.live_session._tts if self.live_session else None
        if current_tts is None:
            current = "gemini"
        else:
            name = type(current_tts).__name__
            mapping = {"QwenTTSProvider": "qwen3", "HoppouTTSProvider": "hoppou", "Chirp3HDTTSProvider": "chirp3_hd", "TikTokTTSProvider": "tiktok"}
            current = mapping.get(name, name)
        return {"result": "ok", "providers": allowed, "current": current}

    def _list_voices(self):
        voices = {}
        if self.config:
            for vname, vdef in self.config.list_voices().items():
                voices[vname] = {
                    "display_name": vdef.get("display_name", vname),
                    "description": vdef.get("description", ""),
                    "providers": [p for p in ("qwen3", "hoppou", "chirp3_hd", "tiktok") if p in vdef],
                }
        return {"result": "ok", "voices": voices}

    def _set_pitch(self, semitones):
        if not self.config.get("audio", "pitch_shift", "enabled", default=False):
            return {"result": "error", "message": "Pitch shifting is disabled in config"}
        self.audio.set_pitch(float(semitones))
        current = self.audio.get_pitch()
        return {"result": "ok", "semitones": current}
