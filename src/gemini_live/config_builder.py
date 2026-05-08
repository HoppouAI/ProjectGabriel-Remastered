"""Builds the LiveConnectConfig the session hands to genai client.aio.live.connect.

Pulled out of session.py because this method alone is ~170 lines of pure
config plumbing - VAD, transcription, voice, session resumption, sampling
params, alpha features, context window compression, media resolution,
thinking config, history config. None of it touches I/O, just reads
self.config and produces a config object.
"""

import logging

from google.genai import types

from src.tools import get_tool_declarations

logger = logging.getLogger(__name__)


class ConfigBuilderMixin:
    def _needs_alpha_api(self):
        """Check if any v1alpha-only features are enabled (affective dialog, proactivity).
        3.1 models don't support these features but still use v1alpha API version."""
        if self.config.is_31_model:
            return False
        return (self.config.enable_affective_dialog is not None
                or self.config.proactivity is not None)

    def _build_config(self, skip_alpha_features=False):
        # Build VAD config based on mode
        if self.config.vad_mode == "silero":
            # Client-side Silero VAD: disable server VAD, we handle speech detection ourselves
            vad_config = types.AutomaticActivityDetection(disabled=True)
        else:
            # Server-side auto VAD with configurable sensitivity
            start_sens_map = {
                "START_SENSITIVITY_LOW": types.StartSensitivity.START_SENSITIVITY_LOW,
                "START_SENSITIVITY_HIGH": types.StartSensitivity.START_SENSITIVITY_HIGH,
            }
            end_sens_map = {
                "END_SENSITIVITY_LOW": types.EndSensitivity.END_SENSITIVITY_LOW,
                "END_SENSITIVITY_HIGH": types.EndSensitivity.END_SENSITIVITY_HIGH,
            }
            vad_config = types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=start_sens_map.get(
                    self.config.vad_start_sensitivity,
                    types.StartSensitivity.START_SENSITIVITY_HIGH,
                ),
                end_of_speech_sensitivity=end_sens_map.get(
                    self.config.vad_end_sensitivity,
                    types.EndSensitivity.END_SENSITIVITY_HIGH,
                ),
                prefix_padding_ms=self.config.vad_prefix_padding_ms,
                silence_duration_ms=self.config.vad_silence_duration_ms,
            )

        transcription_config = types.AudioTranscriptionConfig()

        config_kwargs = dict(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part.from_text(
                    text=self.config.build_system_instruction(self.personality)
                )]
            ),
            tools=get_tool_declarations(self.config),
            input_audio_transcription=transcription_config,
            output_audio_transcription=transcription_config,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.config.voice
                    )
                )
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=vad_config
            ),
            session_resumption=types.SessionResumptionConfig(
                handle=self._session_handle
            ) if self._session_handle else types.SessionResumptionConfig(),
        )

        # Skip session resumption if it keeps failing
        if self._resumption_fail_streak >= 3:
            config_kwargs["session_resumption"] = types.SessionResumptionConfig()
            logger.warning(f"Session resumption disabled (failed {self._resumption_fail_streak} times in a row)")
        elif self._session_handle:
            logger.debug(f"Config includes session handle: {self._session_handle[:24]}...")
        else:
            logger.debug("Config requesting new session handle (no existing handle)")

        if self.config.temperature is not None:
            config_kwargs["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            config_kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            config_kwargs["top_k"] = self.config.top_k
        if self.config.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.config.max_output_tokens
        if not skip_alpha_features and not self.config.is_31_model:
            if self.config.enable_affective_dialog is not None:
                config_kwargs["enable_affective_dialog"] = self.config.enable_affective_dialog
            if self.config.proactivity is not None:
                config_kwargs["proactivity"] = self.config.proactivity

        # Context window compression
        # When custom compression is enabled, skip Gemini's built-in sliding window
        # to avoid 1007 errors at the threshold -- we handle it ourselves via summarization
        if self.config.compression_enabled and not self.config.custom_compression_enabled:
            sw_kwargs = {}
            if self.config.compression_target_tokens is not None:
                sw_kwargs["target_tokens"] = self.config.compression_target_tokens
            cw_kwargs = {"sliding_window": types.SlidingWindow(**sw_kwargs)}
            if self.config.compression_trigger_tokens is not None:
                cw_kwargs["trigger_tokens"] = self.config.compression_trigger_tokens
            config_kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(**cw_kwargs)
            trigger = self.config.compression_trigger_tokens or "default"
            target = self.config.compression_target_tokens or "default"
            logger.info(f"Context compression enabled (trigger={trigger}, target={target})")
        elif self.config.custom_compression_enabled:
            # No built-in compression at all -- we handle it via summarization + reconnect
            trigger = self.config.custom_compression_trigger_tokens or "auto"
            logger.info(f"Custom context compression enabled (trigger={trigger} tokens)")

        # Media resolution (reduces image token cost, critical for 3.1 free tier)
        media_res = self.config.vision_media_resolution
        if media_res and self.config.vision_enabled:
            res_map = {
                "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }
            resolved = res_map.get(media_res.lower())
            if resolved:
                config_kwargs["media_resolution"] = resolved
                token_map = {"low": 280, "medium": 560, "high": 1120}
                tokens = token_map.get(media_res.lower(), "?")
                logger.info(f"Media resolution: {media_res} (~{tokens} tokens/frame)")

        # Thinking configuration
        if self.config.is_31_model:
            # 3.1 models use thinking_level (minimal/low/medium/high) instead of budget
            thinking_level = self.config.thinking_level
            include_thoughts = self.config.thinking_include_thoughts
            if thinking_level is not None or include_thoughts:
                thinking_kwargs = {}
                if thinking_level is not None:
                    thinking_kwargs["thinking_level"] = thinking_level
                if include_thoughts:
                    thinking_kwargs["include_thoughts"] = True
                config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)
        else:
            # 2.5 models use thinking_budget (token count)
            thinking_budget = self.config.thinking_budget
            include_thoughts = self.config.thinking_include_thoughts
            if thinking_budget is not None or include_thoughts:
                thinking_kwargs = {}
                if thinking_budget is not None:
                    thinking_kwargs["thinking_budget"] = thinking_budget
                if include_thoughts:
                    thinking_kwargs["include_thoughts"] = True
                config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

        # History config for context replay (3.1 models need this to accept send_client_content)
        # Only needed on fresh sessions (no handle) where we'll replay previous context
        has_replay = self._replay_context or self._compression_summary
        if has_replay and not self._session_handle and self.config.is_31_model:
            config_kwargs["history_config"] = types.HistoryConfig(
                initial_history_in_client_content=True
            )

        return types.LiveConnectConfig(**config_kwargs)
