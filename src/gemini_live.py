import asyncio
import base64
import io
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError
from websockets.exceptions import ConnectionClosed
import mss
from PIL import Image
from src.tools import get_tool_declarations, ToolHandler
from src.emotions import init_emotion_system, get_emotion_system
from src.idle_chatbox import IdleChatbox

logger = logging.getLogger(__name__)

SESSION_HANDLE_FILE = Path("session_handle.txt")
SESSION_EXPIRY_HOURS = 2
IDLE_ENGAGEMENT_SECONDS = 3600  # 1 hour
CONVERSATION_DIR = Path("data/conversations")


class ConversationLogger:
    """Logs conversation history to JSON files per session. Thread-safe, non-blocking."""

    def __init__(self):
        CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)
        self._entries = []
        self._system_instruction = None
        self._session_start = datetime.now()
        self._file_path = CONVERSATION_DIR / f"{self._session_start.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        self._lock = threading.Lock()
        self._pending_user_idx = None

    def set_system_instruction(self, text: str):
        self._system_instruction = text
        self._save_async()

    def stream_user_message(self, text: str):
        """Update user message in-place or create new entry. Called on each transcription event.
        Only updates in-memory entries - disk writes happen at turn boundaries."""
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
        self._pending_user_idx = None

    def add_user_message(self, text: str):
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
        with self._lock:
            self._entries.append({
                "role": "tool_call",
                "name": name,
                "arguments": args,
                "timestamp": datetime.now().isoformat(),
            })
        self._save_async()

    def add_tool_response(self, name: str, response: dict):
        with self._lock:
            self._entries.append({
                "role": "tool_response",
                "name": name,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            })
        self._save_async()

    def _save_async(self):
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


def _broadcast_console(log_type: str, content: str, extra: dict = None):
    """Broadcast a log entry to the control panel console."""
    try:
        from control_server import add_console_log
        add_console_log(log_type, content, extra)
    except ImportError:
        pass
    except Exception:
        pass


class GeminiLiveSession:
    def __init__(self, config, audio_mgr, osc, tracker, personality_mgr, tts_provider=None):
        self.config = config
        self.audio = audio_mgr
        self.osc = osc
        self.personality = personality_mgr
        self.tool_handler = ToolHandler(audio_mgr, osc, tracker, personality_mgr, config)
        self._tts = tts_provider  # External TTS provider or None
        self._tts_audio_task: asyncio.Task | None = None  # Managed separately for hot-swap
        self._speaking = False
        self._thinking_shown = False
        self._transcript_buffer = ""
        self._input_transcript_buffer = ""  # Buffer for user speech
        self._session_handle = None
        self._session_handle_created = None
        self._handle_fail_count = 0
        self._resumption_fail_streak = 0  # Consecutive session resumption failures across fresh sessions
        self._connection_start_time = 0  # When the current session connected
        self._last_connect_succeeded = True  # Track if last connect attempt reached onopen
        self._rate_limit_backoff = 0
        self._tool_call_pending = False
        self._audio_stream_active = False  # True when server knows we're streaming audio
        self._audio_gated = False  # True when we should suppress outbound audio (tool calls, model speaking)
        self._manual_vad_speaking = False  # Client-side VAD: True when user is speaking
        self._manual_vad_silence_start = 0  # When silence started for manual VAD debounce
        self._silero_vad = None  # Lazy-loaded Silero VAD model
        self._out_queue = asyncio.Queue(maxsize=5)
        self._audio_in_queue = asyncio.Queue()
        self._reconnect_requested = False
        self._mic_muted = False
        self._session = None
        self._stream_closing = False  # Flag to stop audio I/O before stream close
        self._chatbox_error_shown = False  # Track if we've shown an error to VRChat chatbox
        self._last_audio_time = 0  # Track when last audio was received
        self._idle_timeout = 15.0  # Stop talking animations after 15s idle
        self._last_interaction_time = time.time()  # Track last user/AI interaction for engagement
        self._idle_engagement_sent = False  # Only send one engagement prompt per idle period
        self._is_idle = False  # True when AI is idle (not speaking, no active tasks)
        self._replay_context = []  # Previous session entries to replay on error reconnect
        self._pending_finalize_task = None
        self._wanderer = None  # Set externally from main.py
        self._save_audio = False  # Set externally via --save-audio flag
        self._audio_recording = bytearray()  # Accumulated audio for WAV export
        self._audio_recording_writer = None
        self._audio_recording_path = None
        self._audio_recording_seconds = 0.0
        self._usage_metadata = {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "tool_calls": 0,
        }
        self._compression_in_progress = False  # Guard to prevent concurrent compression
        self._compression_summary = None  # Summary from custom compression to seed on reconnect
        self._load_session_handle()
        self._conv_logger = ConversationLogger()
        
        # Initialize emotion system
        self._emotion_system = None
        if config.emotion_enabled:
            self._emotion_system = init_emotion_system(config, osc)
            self._emotion_system.start()
            logger.info("Emotion system initialized")

        # Initialize idle chatbox banner
        self._idle_chatbox = IdleChatbox(osc, config)

    def request_reconnect(self):
        """Request a reconnect on next iteration (manual, no context replay)."""
        self._reconnect_requested = True
        self._replay_context = []  # manual reconnect = fresh start
        self._compression_summary = None
        logger.info("Reconnect requested via control panel")

    async def _summarize_conversation_for_compression(self):
        """Summarize current conversation using a lightweight model for context compression.
        Called when token count approaches the threshold to avoid Gemini's built-in
        compression failing with 1007 errors."""
        if self._compression_in_progress:
            return
        self._compression_in_progress = True
        try:
            entries = self._conv_logger.get_recent_entries(50)
            if not entries:
                return
            lines = []
            for e in entries:
                role = "User" if e["role"] == "user" else "AI"
                lines.append(f"{role}: {e['content']}")
            conversation_text = "\n".join(lines)
            system_prompt = (
                "You are a conversation summarizer. Summarize the following conversation between a user and an AI assistant. "
                "Preserve key topics, facts, names, decisions, and ongoing context. "
                "The summary will be used to seed a fresh session so the AI can continue naturally. "
                "Keep the summary concise but include all important details. Under 500 words."
            )
            api_key = self.config.api_key
            if not api_key:
                logger.warning("No API key for compression summary")
                return
            from google.genai import types as gtypes
            client = genai.Client(api_key=api_key)
            model = self.config.custom_compression_model
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=f"Summarize this conversation:\n\n{conversation_text}",
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=2048,
                    ),
                ),
                timeout=20.0,
            )
            summary = response.text if response.text else None
            if summary:
                self._compression_summary = summary
                self._clear_session_handle()
                self._reconnect_requested = True
                prompt_tokens = self._usage_metadata.get("prompt_tokens", 0)
                logger.info(f"Custom compression triggered at {prompt_tokens} tokens, summary ready ({len(summary)} chars)")
                _broadcast_console("info", f"Context compressed via summary ({prompt_tokens} tokens -> fresh session)")
            else:
                logger.warning("Compression summary returned empty result")
        except asyncio.TimeoutError:
            logger.warning("Compression summary timed out after 20s")
        except Exception as e:
            logger.error(f"Compression summary failed: {e}")
        finally:
            self._compression_in_progress = False

    def _check_custom_compression(self, prompt_tokens: int):
        """Check if custom compression should be triggered based on token count."""
        if not self.config.custom_compression_enabled:
            return
        if self._compression_in_progress or self._compression_summary:
            return
        trigger = self.config.custom_compression_trigger_tokens
        if trigger and prompt_tokens >= trigger:
            asyncio.create_task(self._summarize_conversation_for_compression())

    def _notify_chatbox_error(self):
        """Show a one-time error message in VRChat chatbox when connection drops."""
        if not self._chatbox_error_shown and self.osc:
            self._chatbox_error_shown = True
            self.osc.send_chatbox("Please wait, there's been a small issue with the AI service...")

    def _notify_chatbox_resolved(self):
        """Show resolved message in VRChat chatbox after reconnecting."""
        if self._chatbox_error_shown and self.osc:
            self._chatbox_error_shown = False
            self.osc.send_chatbox("Resolved, ready to chat!")

    def save_audio_to_wav(self):
        """Save accumulated audio recording to a WAV file."""
        if self._audio_recording_writer is not None:
            try:
                self._audio_recording_writer.close()
            except Exception:
                pass
            path = self._audio_recording_path
            duration = self._audio_recording_seconds
            self._audio_recording_writer = None
            self._audio_recording_path = None
            self._audio_recording_seconds = 0.0
            logger.info(f"Saved {duration:.1f}s of audio to {path}")
            return
        if not self._audio_recording:
            logger.info("No audio recorded, skipping WAV save")
            return
        import wave
        from datetime import datetime
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wav_path = data_dir / f"gemini_output_{timestamp}.wav"
        sample_rate = self.config.receive_sample_rate
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(self._audio_recording))
        duration = len(self._audio_recording) / (sample_rate * 2)
        logger.info(f"Saved {duration:.1f}s of audio to {wav_path}")

    def _record_output_audio(self, audio_data: bytes):
        """Stream optional debug audio recording to disk without growing RAM forever."""
        if not audio_data:
            return
        if self._audio_recording_writer is None:
            import wave
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._audio_recording_path = data_dir / f"gemini_output_{timestamp}.wav"
            self._audio_recording_writer = wave.open(str(self._audio_recording_path), "wb")
            self._audio_recording_writer.setnchannels(1)
            self._audio_recording_writer.setsampwidth(2)
            self._audio_recording_writer.setframerate(self.config.receive_sample_rate)
            logger.info(f"Recording Gemini output audio to {self._audio_recording_path}")
        self._audio_recording_writer.writeframes(audio_data)
        self._audio_recording_seconds += len(audio_data) / (self.config.receive_sample_rate * 2)

    def set_mic_muted(self, muted: bool):
        """Set mic mute state."""
        self._mic_muted = muted
        logger.info(f"Mic mute set to {muted}")

    async def send_text(self, text: str):
        """Send text to the model via realtime input."""
        if self._session:
            if self._tool_call_pending:
                logger.debug("Skipping text send - tool call pending")
                return
            try:
                await self._session.send_realtime_input(text=text)
                # Signal activity start+end so the model knows the text turn is complete
                # Without this, the model waits for audio silence (VAD) which never comes for text-only input
                await self._session.send_realtime_input(activity_start=types.ActivityStart())
                await self._session.send_realtime_input(activity_end=types.ActivityEnd())
                self._conv_logger.add_user_message(text)
                logger.info(f"Sent text to model: {text[:50]}...")
            except Exception as e:
                logger.error(f"Failed to send text: {e}")

    async def send_client_content_safe(self, turns, turn_complete=True):
        """Send client content, waiting until the model stops speaking to avoid interruptions.
        
        Per Gemini Live best practices: sendClientContent with turnComplete=True
        while the model is speaking will interrupt it mid-speech.
        
        For 3.1 models: send_client_content is only for initial history seeding.
        Mid-session text updates must use send_realtime_input(text=...) instead.
        """
        if not self._session:
            return
        # Wait up to 30s for model to finish speaking
        for _ in range(300):
            if not self._speaking:
                break
            await asyncio.sleep(0.1)
        try:
            if self.config.is_31_model:
                # 3.1 models: extract text from turns and send via realtime input
                text = self._extract_text_from_turns(turns)
                if text:
                    await self._session.send_realtime_input(text=text)
            else:
                await self._session.send_client_content(turns=turns, turn_complete=turn_complete)
        except Exception as e:
            logger.error(f"Failed to send client content: {e}")
            raise

    async def _replay_previous_context(self, session):
        """Replay last few messages as context after an error reconnect.
        Uses send_client_content to seed the session with recent conversation history
        so the model doesnt lose track of what was being discussed."""
        if not self._replay_context:
            return
        try:
            turns = []
            for entry in self._replay_context:
                role = "model" if entry["role"] == "assistant" else "user"
                turns.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=entry["content"])]
                ))
            # Send all turns as context, last one with turn_complete=True
            await session.send_client_content(turns=turns, turn_complete=True)
            count = len(self._replay_context)
            logger.info(f"Replayed {count} messages as session context")
            _broadcast_console("info", f"Replayed {count} messages as context after reconnect")
        except Exception as e:
            logger.warning(f"Failed to replay context (non-fatal): {e}")

    @staticmethod
    def _extract_text_from_turns(turns):
        """Extract text content from turns (Content object or dict) for realtime input."""
        if hasattr(turns, "parts"):
            # types.Content object
            for part in turns.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
        elif isinstance(turns, dict):
            for part in turns.get("parts", []):
                if isinstance(part, dict) and "text" in part:
                    return part["text"]
        elif isinstance(turns, list):
            # List of turns - combine text from all
            texts = []
            for turn in turns:
                if hasattr(turn, "parts"):
                    for part in turn.parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
                elif isinstance(turn, dict):
                    for part in turn.get("parts", []):
                        if isinstance(part, dict) and "text" in part:
                            texts.append(part["text"])
            return " ".join(texts) if texts else None
        return None

    def _schedule_user_finalize(self):
        """Schedule delayed finalization of user transcript to catch late events."""
        if self._pending_finalize_task:
            self._pending_finalize_task.cancel()
        self._pending_finalize_task = asyncio.create_task(self._delayed_user_finalize())

    async def _delayed_user_finalize(self):
        """Wait for late transcription events, then finalize the user entry."""
        try:
            await asyncio.sleep(0.5)
            self._conv_logger.finalize_user_message()
            self._input_transcript_buffer = ""
            if self.config.obs_enabled:
                _broadcast_console("user_turn_complete", "")
        except asyncio.CancelledError:
            pass
        finally:
            self._pending_finalize_task = None

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

    def _load_session_handle(self):
        if not SESSION_HANDLE_FILE.exists():
            return
        try:
            data = json.loads(SESSION_HANDLE_FILE.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(data["created"])
            if datetime.now() - created < timedelta(hours=SESSION_EXPIRY_HOURS):
                self._session_handle = data["handle"]
                self._session_handle_created = created
                logger.info(f"Loaded session handle (created {created.strftime('%H:%M:%S')})")
            else:
                logger.info("Session handle expired, will create new session")
                self._clear_session_handle()
        except Exception as e:
            logger.warning(f"Failed to load session handle: {e}")
            self._clear_session_handle()

    def _save_session_handle(self, handle: str):
        self._session_handle = handle
        self._session_handle_created = datetime.now()
        self._handle_fail_count = 0
        self._resumption_fail_streak = 0
        data = {"handle": handle, "created": self._session_handle_created.isoformat()}
        SESSION_HANDLE_FILE.write_text(json.dumps(data), encoding="utf-8")
        logger.debug("Saved new session handle")

    def _clear_session_handle(self):
        self._session_handle = None
        self._session_handle_created = None
        self._handle_fail_count = 0
        if SESSION_HANDLE_FILE.exists():
            SESSION_HANDLE_FILE.unlink()
            logger.info("Cleared session handle")

    def _is_session_handle_expired(self):
        if not self._session_handle or not self._session_handle_created:
            return False
        return datetime.now() - self._session_handle_created >= timedelta(hours=SESSION_EXPIRY_HOURS)

    async def run(self):
        self._alpha_fallback_failed = False
        self._rate_limit_backoff = 0
        while True:
            # Capture previous conversation for context replay on error reconnects
            # Skip on manual reconnect since the user explicitly wants a fresh session
            # Skip when compression summary is ready (it takes priority over raw replay)
            if not self._reconnect_requested and not self._compression_summary and hasattr(self, '_conv_logger') and self._conv_logger:
                recent = self._conv_logger.get_recent_entries(self.config.session_replay_messages)
                if recent:
                    self._replay_context = recent
            self._reconnect_requested = False
            # Check for expired session handle before each connection attempt
            if self._is_session_handle_expired():
                logger.info("Session handle expired (2h), starting fresh session")
                _broadcast_console("info", "Session handle expired, starting fresh session")
                self._clear_session_handle()
            try:
                use_alpha = self._needs_alpha_api() and not self._alpha_fallback_failed
                # Always use v1alpha API version for Live API (required for video, etc.)
                # The use_alpha flag only controls whether to include alpha-only CONFIG features
                client = genai.Client(
                    api_key=self.config.api_key,
                    http_options={"api_version": "v1alpha"},
                )
                if use_alpha:
                    live_config = self._build_config()
                    logger.info("Using v1alpha API with affective dialog / proactivity")
                else:
                    live_config = self._build_config(skip_alpha_features=self._alpha_fallback_failed)
                # Log model family info on first connect
                if self.config.is_31_model:
                    logger.info("Using 3.1 model (thinkingLevel, realtime text injection, no affective/proactive)")
                    if not self.config.google_search_enabled:
                        logger.info("Google Search auto-disabled for 3.1 model")
                else:
                    logger.info("Using 2.5 model (thinkingBudget, send_client_content, v1alpha features available)")
                # Log VAD mode
                if self.config.vad_mode == "silero":
                    logger.info(f"Silero VAD enabled (threshold={self.config.vad_silero_threshold}, silence={self.config.vad_silence_duration_ms}ms)")
                    logger.info("Audio gating active: outbound suppressed during model speech and tool calls")
                else:
                    logger.info("Using Gemini server-side VAD (auto mode)")
                if self._session_handle:
                    # If last connect attempt failed before reaching onopen, handle is likely bad
                    if not self._last_connect_succeeded:
                        logger.warning("Last connect failed before onopen, clearing stale handle")
                        self._resumption_fail_streak += 1
                        self._clear_session_handle()
                    else:
                        logger.info(f"Connecting to Gemini Live with session resumption...")
                elif self._resumption_fail_streak >= 3:
                    logger.info(f"Connecting to Gemini Live ({self.config.model}) [resumption disabled after {self._resumption_fail_streak} failures]...")
                else:
                    logger.info(f"Connecting to Gemini Live ({self.config.model})...")

                self._last_connect_succeeded = False
                async with client.aio.live.connect(
                    model=self.config.model,
                    config=live_config,
                ) as session:
                    self._last_connect_succeeded = True
                    logger.info("Connected to Gemini Live")
                    _broadcast_console("info", f"Connected to Gemini Live ({self.config.model})")
                    self._notify_chatbox_resolved()
                    # Reset resumption fail streak on successful connection
                    if self._resumption_fail_streak > 0 and self._resumption_fail_streak < 3:
                        self._resumption_fail_streak = 0
                    self._conv_logger = ConversationLogger()
                    self._conv_logger.set_system_instruction(
                        self.config.build_system_instruction(self.personality)
                    )
                    self._session = session
                    self._connection_start_time = time.time()

                    # Replay previous context on fresh sessions (no handle = context was lost)
                    # Skip if session resumption is active since the model already has history
                    if self._compression_summary and not self._session_handle:
                        # Custom compression: send summary as initial context
                        summary_turns = [types.Content(
                            role="user",
                            parts=[types.Part.from_text(
                                text=f"[Context from previous session - continue naturally]\n{self._compression_summary}"
                            )]
                        )]
                        await session.send_client_content(turns=summary_turns, turn_complete=True)
                        logger.info(f"Seeded fresh session with compression summary ({len(self._compression_summary)} chars)")
                        _broadcast_console("info", "Seeded session with conversation summary")
                        self._compression_summary = None
                    elif self._replay_context and not self._session_handle:
                        await self._replay_previous_context(session)
                    self._replay_context = []
                    self._rate_limit_backoff = 0
                    self._last_interaction_time = time.time()
                    self._idle_engagement_sent = False
                    self.tool_handler.session = session
                    self.tool_handler.live_session = self
                    self._out_queue = asyncio.Queue(maxsize=5)
                    self._audio_in_queue = asyncio.Queue()
                    self._stream_closing = False
                    self._playback_interrupted = False
                    self._audio_stream_active = False
                    self._audio_gated = False
                    self._manual_vad_speaking = False
                    self._manual_vad_silence_start = 0
                    # Reset Silero VAD internal state if loaded
                    if self._silero_vad is not None:
                        self._silero_vad.reset_states()
                    input_stream = self.audio.open_input_stream()
                    output_stream = self.audio.open_output_stream()
                    tasks = []
                    try:
                        task_specs = [
                            ("audio-listen", self._listen_audio_loop(input_stream)),
                            ("send-realtime", self._send_realtime_loop(session)),
                            ("receive", self._receive_loop(session)),
                            ("audio-playback", self._play_audio_loop(output_stream)),
                            ("reconnect-monitor", self._reconnect_monitor_loop()),
                            ("now-playing", self._now_playing_loop()),
                            ("idle-check", self._idle_check_loop()),
                        ]
                        # Always start TTS audio loop (supports hot-swap)
                        task_specs.append(("tts-audio", self._tts_audio_loop()))
                        if self.config.vision_enabled:
                            task_specs.append(("screen-capture", self._capture_screen_loop()))
                            logger.info(f"Screen capture enabled (monitor {self.config.vision_monitor})")
                        tasks = [
                            asyncio.create_task(coro, name=f"gemini-{name}")
                            for name, coro in task_specs
                        ]
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            if task.cancelled():
                                continue
                            exc = task.exception()
                            if exc:
                                raise exc
                            raise ConnectionError(f"{task.get_name()} stopped unexpectedly")
                    finally:
                        self._session = None
                        self._stream_closing = True
                        self._idle_chatbox.stop()
                        for task in tasks:
                            task.cancel()
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                        if self._pending_finalize_task:
                            self._pending_finalize_task.cancel()
                            self._pending_finalize_task = None
                        # Drain audio queue and wait for in-flight writes
                        while not self._audio_in_queue.empty():
                            try:
                                self._audio_in_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        while not self._out_queue.empty():
                            try:
                                self._out_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        # Wait for any in-progress to_thread writes to finish
                        await asyncio.sleep(0.5)
                        try:
                            input_stream.close()
                        except Exception:
                            pass
                        try:
                            output_stream.close()
                        except Exception:
                            pass
                        self._stream_closing = False

            except APIError as e:
                err_str = str(e)
                err_lower = err_str.lower()

                # Rate limiting - check expired handle first, then rotate key
                if "429" in err_lower or "quota" in err_lower or "rate" in err_lower:
                    logger.warning(f"Rate limit error details: {err_str[:200]}")
                    # Flush buffered transcript so last messages make it into replay context
                    if self._transcript_buffer.strip():
                        self._conv_logger.add_assistant_message(self._transcript_buffer)
                        self._transcript_buffer = ""
                    if self._input_transcript_buffer.strip():
                        self._conv_logger.finalize_user_message()
                        self._input_transcript_buffer = ""
                    if self._session_handle and self._is_session_handle_expired():
                        logger.warning("Rate limited with expired session handle, clearing handle")
                        _broadcast_console("info", "Expired session handle causing rate limit, clearing")
                        self._clear_session_handle()
                        self._rate_limit_backoff = 0
                        await asyncio.sleep(1)
                        continue
                    old_key = self.config.api_key
                    new_key = self.config.rotate_key()
                    if new_key != old_key:
                        logger.warning("Rate limited - switched API key")
                        _broadcast_console("info", "Rate limited - switched API key")
                        self._rate_limit_backoff = 0
                        # Handle is key-specific, can't resume on a different key
                        if self._session_handle:
                            self._clear_session_handle()
                    else:
                        self._rate_limit_backoff = min(self._rate_limit_backoff + 1, 5)
                        wait = 5 * (2 ** self._rate_limit_backoff)  # 10, 20, 40, 80, 160, 160s
                        logger.warning(f"Rate limited - waiting {wait}s before retry")
                        _broadcast_console("info", f"Rate limited - waiting {wait}s")
                        await asyncio.sleep(wait)
                    continue

                # v1alpha features not supported - fall back
                if not self._alpha_fallback_failed and any(kw in err_str for kw in ("enableAffectiveDialog", "proactivity", "proactive_audio")):
                    logger.warning(f"v1alpha features rejected ({e}), falling back to standard API")
                    _broadcast_console("info", "v1alpha features not supported, falling back")
                    self._alpha_fallback_failed = True
                    await asyncio.sleep(0.5)
                    continue

                # WebSocket close codes - keep handle for transient errors, use fail counter
                if any(code in err_str for code in ("1006", "1007", "1008", "1009", "1011", "1012", "1013", "1014")):
                    logger.warning(f"WebSocket close ({e}), reconnecting...")
                    _broadcast_console("error", f"WebSocket error: {err_str[:100]}")
                    self._notify_chatbox_error()
                    # Flush buffered transcript so last messages make it into replay context
                    if self._transcript_buffer.strip():
                        self._conv_logger.add_assistant_message(self._transcript_buffer)
                        self._transcript_buffer = ""
                    if self._input_transcript_buffer.strip():
                        self._conv_logger.finalize_user_message()
                        self._input_transcript_buffer = ""
                    # Check if session crashed quickly (within 15s) - likely handle issue
                    session_was_short = self._connection_start_time > 0 and (time.time() - self._connection_start_time) < 15
                    err_threshold = self.config.session_error_threshold
                    # 1007 (invalid argument) - clear handle after configurable threshold
                    if "1007" in err_str and self._session_handle:
                        self._handle_fail_count += 1
                        logger.warning(f"1007 invalid argument (attempt {self._handle_fail_count}/{err_threshold})")
                        if self._handle_fail_count >= err_threshold:
                            logger.warning(f"Clearing session handle after {err_threshold} consecutive 1007 error(s)")
                            self._resumption_fail_streak += 1
                            self._clear_session_handle()
                        await asyncio.sleep(3)
                        continue
                    # 1011 (internal error) with short session - handle likely corrupted
                    elif "1011" in err_str and self._session_handle and session_was_short:
                        self._handle_fail_count += 1
                        logger.warning(f"1011 internal error after short session (attempt {self._handle_fail_count}/2)")
                        if self._handle_fail_count >= 2:
                            logger.warning("Clearing session handle - repeated quick crashes")
                            self._resumption_fail_streak += 1
                            self._clear_session_handle()
                    elif "1008" in err_str and self._session_handle:
                        self._handle_fail_count += 1
                        if self._handle_fail_count >= 2:
                            logger.warning("Clearing session handle after 2 consecutive 1008 errors")
                            self._resumption_fail_streak += 1
                            self._clear_session_handle()
                    elif self._session_handle:
                        self._handle_fail_count += 1
                        if self._handle_fail_count >= err_threshold:
                            logger.warning(f"Session handle failed {err_threshold} time(s) after WS errors, clearing")
                            self._clear_session_handle()
                    await asyncio.sleep(0.5)
                    continue

                # Session handle issues (non-WS errors)
                if self._session_handle:
                    self._handle_fail_count += 1
                    if self._handle_fail_count >= 2:
                        logger.warning("Session handle failed twice, clearing and using new session")
                        _broadcast_console("info", "Session handle expired, starting new session")
                        self._clear_session_handle()
                    else:
                        logger.warning(f"Session handle may be invalid (attempt {self._handle_fail_count}/2)")

                logger.error(f"API error: {e}")
                _broadcast_console("error", f"API error: {err_str[:100]}")
                self._notify_chatbox_error()
                await asyncio.sleep(2)
                continue

            except ConnectionClosed as e:
                code = getattr(e, 'code', None)
                reason = getattr(e, 'reason', '') or ''
                logger.warning(f"WebSocket closed (code={code}, reason={reason[:80]}), reconnecting...")
                _broadcast_console("error", f"WebSocket closed: {code} {reason[:60]}")
                self._notify_chatbox_error()
                # Flush any buffered transcript so the last model message is in replay context
                if self._transcript_buffer.strip():
                    self._conv_logger.add_assistant_message(self._transcript_buffer)
                    self._transcript_buffer = ""
                if self._input_transcript_buffer.strip():
                    self._conv_logger.finalize_user_message()
                    self._input_transcript_buffer = ""
                # Check if session crashed quickly (within 15s) - likely handle issue
                session_was_short = self._connection_start_time > 0 and (time.time() - self._connection_start_time) < 15
                # 1007 (invalid argument) - clear handle after configurable threshold
                err_threshold = self.config.session_error_threshold
                if code == 1007 and self._session_handle:
                    self._handle_fail_count += 1
                    logger.warning(f"1007 invalid argument (attempt {self._handle_fail_count}/{err_threshold})")
                    if self._handle_fail_count >= err_threshold:
                        logger.warning(f"Clearing session handle after {err_threshold} consecutive 1007 error(s)")
                        self._resumption_fail_streak += 1
                        self._clear_session_handle()
                    await asyncio.sleep(3)
                    continue
                # 1011 (internal error) with short session - handle likely corrupted
                elif code == 1011 and self._session_handle and session_was_short:
                    self._handle_fail_count += 1
                    logger.warning(f"1011 internal error after short session (attempt {self._handle_fail_count}/2)")
                    if self._handle_fail_count >= 2:
                        logger.warning("Clearing session handle - repeated quick crashes")
                        self._resumption_fail_streak += 1
                        self._clear_session_handle()
                elif code == 1008 and self._session_handle:
                    self._handle_fail_count += 1
                    if self._handle_fail_count >= 2:
                        logger.warning("Clearing session handle after 2 consecutive 1008 errors")
                        self._resumption_fail_streak += 1
                        self._clear_session_handle()
                elif self._session_handle:
                    self._handle_fail_count += 1
                    if self._handle_fail_count >= err_threshold:
                        logger.warning(f"Session handle failed {err_threshold} time(s), clearing")
                        self._clear_session_handle()
                await asyncio.sleep(0.5)
                continue

            except (ConnectionError, OSError, TimeoutError) as e:
                # Network-level errors - keep handle, just retry
                logger.warning(f"Network error: {e}, reconnecting in 3s...")
                _broadcast_console("error", f"Network error: {str(e)[:80]}")
                self._notify_chatbox_error()
                # Flush buffered transcript so last messages make it into replay context
                if self._transcript_buffer.strip():
                    self._conv_logger.add_assistant_message(self._transcript_buffer)
                    self._transcript_buffer = ""
                if self._input_transcript_buffer.strip():
                    self._conv_logger.finalize_user_message()
                    self._input_transcript_buffer = ""
                await asyncio.sleep(3)
                continue

            except (KeyboardInterrupt, SystemExit):
                raise

            except Exception as e:
                err_str = str(e)

                # v1alpha features not supported - fall back
                if not self._alpha_fallback_failed and any(kw in err_str for kw in ("enableAffectiveDialog", "proactivity", "proactive_audio")):
                    logger.warning(f"v1alpha features rejected ({e}), falling back to standard API")
                    _broadcast_console("info", "v1alpha features not supported, falling back")
                    self._alpha_fallback_failed = True
                    await asyncio.sleep(0.5)
                    continue

                # WebSocket close codes (may come wrapped in generic exceptions)
                if any(code in err_str for code in ("1006", "1007", "1008", "1009", "1011", "1012", "1013", "1014")):
                    logger.warning(f"WebSocket error ({e}), reconnecting...")
                    _broadcast_console("error", f"WebSocket error: {err_str[:100]}")
                    self._notify_chatbox_error()
                    if "1007" in err_str or "1008" in err_str:
                        self._handle_fail_count += 1
                        has_handle = "with handle" if self._session_handle else "no handle"
                        logger.warning(f"1007/1008 error ({has_handle}, attempt {self._handle_fail_count}/2)")
                        if self._handle_fail_count >= 2 and self._session_handle:
                            logger.warning("Clearing session handle after 2 consecutive 1007/1008 errors")
                            self._clear_session_handle()
                    elif self._session_handle:
                        self._handle_fail_count += 1
                        if self._handle_fail_count >= 3:
                            self._clear_session_handle()
                    await asyncio.sleep(0.5)
                    continue

                # Reconnect request is not an error
                if "Reconnect requested" in err_str:
                    logger.info("Reconnecting as requested...")
                    continue

                # Session handle issues
                if self._session_handle:
                    self._handle_fail_count += 1
                    if self._handle_fail_count >= 2:
                        logger.warning("Session handle failed twice, clearing and using new session")
                        _broadcast_console("info", "Session handle expired, starting new session")
                        self._clear_session_handle()
                    else:
                        logger.warning(f"Session handle may be invalid (attempt {self._handle_fail_count}/2)")

                logger.error(f"Session error: {type(e).__name__}: {e}")
                _broadcast_console("error", f"Session error: {err_str[:100]}")
                self._notify_chatbox_error()
                await asyncio.sleep(2)
                continue

    async def _reconnect_monitor_loop(self):
        """Monitor for reconnect requests from control panel."""
        while True:
            if self._reconnect_requested:
                logger.info("Processing reconnect request...")
                raise Exception("Reconnect requested")
            await asyncio.sleep(0.5)

    async def _delayed_reconnect(self, delay: float):
        """Wait for delay seconds then trigger a reconnect."""
        await asyncio.sleep(delay)
        self._reconnect_requested = True

    async def _idle_check_loop(self):
        """Monitor for idle state, stop talking animations, and trigger idle animation."""
        while True:
            await asyncio.sleep(1)
            if self._speaking and self._last_audio_time > 0:
                idle_time = time.time() - self._last_audio_time
                if idle_time >= self._idle_timeout:
                    logger.debug(f"AI idle for {idle_time:.1f}s, stopping talking animations")
                    self._speaking = False
                    self._last_audio_time = 0
                    if self._emotion_system:
                        self._emotion_system.stop_speaking()
                    self.osc.set_typing(False)
            # Check music state once per iteration
            music_playing = self.audio.get_music_progress() is not None
            # Check if tracker is following someone
            tracker_active = getattr(self.tool_handler.tracker, 'active', False) if self.tool_handler.tracker else False
            # Check if Lyria music gen is playing
            music_gen = getattr(self.tool_handler, 'music_gen', None)
            music_gen_active = music_gen.is_active if music_gen else False
            # Anything keeping the AI busy suppresses idle
            busy = music_playing or tracker_active or music_gen_active
            # Don't trigger idle during active tasks
            if self._emotion_system:
                self._emotion_system.set_seated(self.osc.seated)
                if busy:
                    self._emotion_system.mark_activity()
                else:
                    self._emotion_system.check_idle()
            # Track idle state for vision pause and chatbox
            idle_now = not self._speaking and not self._manual_vad_speaking and not busy
            self._is_idle = idle_now
            # Start idle chatbox when idle and not busy
            if idle_now:
                emo = self._emotion_system
                if (emo and emo._idle_active) or not emo:
                    self._idle_chatbox.start()
            elif busy:
                self._idle_chatbox.stop()
            # Idle engagement - prompt model to speak after long silence
            if (
                not self._idle_engagement_sent
                and not self._speaking
                and not busy
                and time.time() - self._last_interaction_time >= IDLE_ENGAGEMENT_SECONDS
            ):
                self._idle_engagement_sent = True
                logger.info(f"Idle for {IDLE_ENGAGEMENT_SECONDS}s, sending engagement prompt")
                _broadcast_console("info", "Sending idle engagement prompt")
                await self.send_text(
                    "System update - You have been idle for a while. "
                    "Try to engage nearby people in conversation. "
                    "Say something interesting, ask a question, or make an observation to get someone to talk to you."
                )

    async def _listen_audio_loop(self, input_stream):
        while True:
            try:
                if self._stream_closing:
                    return
                data = await asyncio.to_thread(
                    input_stream.read,
                    self.config.chunk_size,
                    exception_on_overflow=False,
                )
                if self._stream_closing:
                    return
                if not self._mic_muted:
                    try:
                        self._out_queue.put_nowait(("audio", data))
                    except asyncio.QueueFull:
                        try:
                            self._out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self._out_queue.put_nowait(("audio", data))
            except asyncio.CancelledError:
                return
            except OSError:
                return
            except Exception as e:
                logger.error(f"Audio listen error: {e}")
                raise

    async def _send_audio_stream_end(self, session):
        """Send audioStreamEnd to flush server-side buffered audio.
        Per Joe_Hu's production patterns: always flush before gating audio."""
        if self._audio_stream_active:
            try:
                await session.send_realtime_input(audio_stream_end=True)
                self._audio_stream_active = False
                logger.debug("Sent audioStreamEnd")
            except Exception as e:
                logger.debug(f"Failed to send audioStreamEnd: {e}")

    async def _send_activity_start(self, session):
        """Send activityStart signal for manual VAD mode."""
        if not self._manual_vad_speaking:
            try:
                self._manual_vad_speaking = True
                await session.send_realtime_input(activity_start=types.ActivityStart())
                logger.debug("Sent activityStart")
            except Exception as e:
                logger.debug(f"Failed to send activityStart: {e}")

    async def _send_activity_end(self, session):
        """Send activityEnd signal for manual VAD mode."""
        if self._manual_vad_speaking:
            try:
                self._manual_vad_speaking = False
                await session.send_realtime_input(activity_end=types.ActivityEnd())
                logger.debug("Sent activityEnd")
            except Exception as e:
                logger.debug(f"Failed to send activityEnd: {e}")

    async def _gate_audio(self, session):
        """Gate outbound audio, sending audioStreamEnd first to flush server buffer.
        Used when tool calls come in or model starts speaking."""
        if not self._audio_gated:
            self._audio_gated = True
            # If Silero VAD is active, end the activity first
            if self.config.vad_mode == "silero" and self._manual_vad_speaking:
                await self._send_activity_end(session)
            await self._send_audio_stream_end(session)
            logger.debug("Audio gated (outbound suppressed)")

    def _ungate_audio(self):
        """Resume outbound audio. Audio will start flowing on next send."""
        if self._audio_gated:
            self._audio_gated = False
            logger.debug("Audio ungated (outbound resumed)")

    def _load_silero_vad(self):
        """Lazy-load Silero VAD model. Uses torch.hub, cached after first download."""
        if self._silero_vad is not None:
            return self._silero_vad
        import torch
        logger.info("Loading Silero VAD model...")
        torch.set_num_threads(1)  # single thread is fine for VAD
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        model.eval()
        self._silero_vad = model
        logger.info("Silero VAD model loaded")
        return model

    def _silero_detect_speech(self, data: bytes) -> float:
        """Run Silero VAD on a PCM 16-bit 16kHz audio chunk. Returns speech probability 0.0-1.0.
        Silero requires exactly 512 samples at 16kHz, so we split and take the max probability."""
        import torch
        model = self._load_silero_vad()
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        # Silero needs 512-sample chunks at 16kHz
        chunk_size = 512
        max_prob = 0.0
        with torch.no_grad():
            for i in range(0, len(samples) - chunk_size + 1, chunk_size):
                chunk = torch.from_numpy(samples[i:i + chunk_size])
                prob = model(chunk, 16000).item()
                if prob > max_prob:
                    max_prob = prob
        return max_prob

    async def _send_realtime_loop(self, session):
        use_silero = self.config.vad_mode == "silero"
        silero_threshold = self.config.vad_silero_threshold
        silence_ms = self.config.vad_silence_duration_ms
        # Pre-load silero model before the loop starts
        if use_silero:
            await asyncio.to_thread(self._load_silero_vad)
        while True:
            try:
                msg_type, data = await self._out_queue.get()
                # Hard gate: tool calls in progress, drop all audio
                if self._tool_call_pending and msg_type == "audio":
                    continue
                if msg_type == "audio":
                    if use_silero:
                        # Silero VAD: detect speech via ML model
                        prob = await asyncio.to_thread(self._silero_detect_speech, data)
                        if prob >= silero_threshold:
                            # Speech detected
                            self._manual_vad_silence_start = 0
                            if self._audio_gated:
                                # User is speaking while model is talking, interrupt!
                                self._ungate_audio()
                                logger.debug("Silero detected speech while gated, ungating for interruption")
                            if not self._manual_vad_speaking:
                                await self._send_activity_start(session)
                        else:
                            # Silence
                            if self._audio_gated:
                                continue  # still gated and no speech, skip
                            if self._manual_vad_speaking:
                                if self._manual_vad_silence_start == 0:
                                    self._manual_vad_silence_start = time.time()
                                elapsed_silence = (time.time() - self._manual_vad_silence_start) * 1000
                                if elapsed_silence >= silence_ms:
                                    await self._send_activity_end(session)
                                    await self._send_audio_stream_end(session)
                                    self._manual_vad_silence_start = 0
                                    continue  # dont send this silent chunk after activityEnd
                            else:
                                # Not speaking and silence, don't send audio
                                # Joe_Hu: "don't send audio between activityEnd and next activityStart"
                                continue
                    else:
                        # Auto mode: just skip gated audio
                        if self._audio_gated:
                            continue
                    # Send the audio chunk
                    self._audio_stream_active = True
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                elif msg_type == "video":
                    await session.send_realtime_input(
                        video=types.Blob(data=data, mime_type="image/jpeg")
                    )
            except asyncio.CancelledError:
                return
            except ConnectionClosed:
                logger.warning("Send loop: WebSocket closed, stopping")
                return
            except Exception as e:
                logger.error(f"Send realtime error: {e}")
                raise

    async def _play_audio_loop(self, output_stream):
        # Write in small sub-chunks so interrupt can bail out fast
        # ~85ms per chunk at 24kHz 16-bit mono
        CHUNK_BYTES = 4096
        while True:
            try:
                audio_data = await self._audio_in_queue.get()
                if self._stream_closing:
                    return
                audio_data = self.audio.process_output_audio(audio_data)
                if audio_data:
                    if self._save_audio:
                        self._record_output_audio(audio_data)
                    # Chunked write so we can stop quickly on interrupt
                    for i in range(0, len(audio_data), CHUNK_BYTES):
                        if self._playback_interrupted:
                            break
                        await asyncio.to_thread(
                            output_stream.write, audio_data[i:i + CHUNK_BYTES]
                        )
                    if self._playback_interrupted:
                        self._playback_interrupted = False
            except asyncio.CancelledError:
                return
            except OSError:
                return
            except Exception as e:
                logger.error(f"Audio play error: {e}")
                raise

    async def _tts_audio_loop(self):
        """Pull audio from external TTS provider and feed into the audio playback queue.
        
        Runs continuously to support hot-swapping TTS providers mid-session.
        When self._tts is None (gemini mode), this loop simply idles.
        """
        while True:
            try:
                if not self._tts:
                    await asyncio.sleep(0.1)
                    continue
                pcm = await self._tts.get_audio()
                if pcm and self._tts and not self._tts._interrupted:
                    await self._audio_in_queue.put(pcm)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"TTS audio loop error: {e}")
                await asyncio.sleep(0.05)

    def switch_tts_provider(self, new_provider):
        """Hot-swap the TTS provider mid-session.
        
        Stops the old provider, sets the new one (or None for gemini), and starts it.
        Called from ToolHandler.
        """
        old = self._tts
        if old:
            try:
                old.interrupt()
                old.stop()
            except Exception as e:
                logger.warning(f"Error stopping old TTS provider: {e}")
        self._tts = new_provider
        if new_provider:
            new_provider.start()
            logger.info(f"Switched to TTS provider: {type(new_provider).__name__}")
        else:
            logger.info("Switched to Gemini native audio")

    def _capture_screen_frame(self):
        try:
            with mss.mss() as sct:
                monitor_idx = self.config.vision_monitor
                if monitor_idx >= len(sct.monitors):
                    monitor_idx = 0
                monitor = sct.monitors[monitor_idx]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                max_size = self.config.vision_max_size
                # Use smaller resolution for 3.1 models to save tokens
                if self.config.is_31_model and max_size > 768:
                    max_size = 768
                if img.width > max_size or img.height > max_size:
                    img.thumbnail([max_size, max_size])
                buffer = io.BytesIO()
                quality = self.config.vision_quality
                # Use lower JPEG quality for 3.1 models (smaller payload)
                if self.config.is_31_model and quality > 60:
                    quality = 60
                img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                return buffer.read()
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None

    async def _capture_screen_loop(self):
        with mss.mss() as sct:
            monitor_idx = self.config.vision_monitor
            if monitor_idx >= len(sct.monitors):
                logger.warning(f"Monitor {monitor_idx} not found, using monitor 0")
                monitor_idx = 0
            monitor = sct.monitors[monitor_idx]
            logger.info(f"Capturing monitor {monitor_idx}: {monitor['width']}x{monitor['height']}")
        interval = self.config.vision_interval
        # Auto-increase interval for 3.1 models if user hasn't set a higher value
        if self.config.is_31_model and interval < 2.0:
            interval = 2.0
            logger.info(f"Vision interval increased to {interval}s for 3.1 model (token optimization)")
        pause_on_output = self.config.vision_pause_on_output
        pause_on_idle = self.config.vision_pause_on_idle
        idle_interval = self.config.vision_idle_interval
        if pause_on_output:
            logger.info("Vision pause enabled (skips frames during speech/music, not live music)")
        if pause_on_idle:
            logger.info(f"Vision slows to {idle_interval}s interval when idle (normal: {interval}s)")
        try:
            while True:
                current_interval = interval
                # Slow down when AI is idle (nobody talking, no active tasks)
                if pause_on_idle and self._is_idle:
                    current_interval = idle_interval
                # Skip frame when AI is speaking or music is playing (unless live music is active)
                if pause_on_output:
                    music_gen = getattr(self.tool_handler, 'music_gen', None)
                    music_gen_active = music_gen.is_active if music_gen else False
                    if not music_gen_active:
                        if self._speaking or self.audio.is_music_playing():
                            await asyncio.sleep(current_interval)
                            continue
                frame = await asyncio.to_thread(self._capture_screen_frame)
                if frame:
                    try:
                        self._out_queue.put_nowait(("video", frame))
                    except asyncio.QueueFull:
                        try:
                            self._out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            self._out_queue.put_nowait(("video", frame))
                        except asyncio.QueueFull:
                            pass
                await asyncio.sleep(current_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Screen capture loop error: {e}")
            raise

    async def _receive_loop(self, session):
        while True:
            try:
                async for response in session.receive():
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            # Thought summaries (when include_thoughts is enabled)
                            if getattr(part, "thought", False) and part.text:
                                _broadcast_console("thinking", part.text, {"streaming": True})
                                logger.debug(f"Thought: {part.text[:100]}")
                                # Show "Thinking..." in VRChat chatbox (only once per thinking phase)
                                if not self._thinking_shown:
                                    self._thinking_shown = True
                                    self._idle_chatbox.stop()
                                    self.osc.set_typing(True)
                                    self.osc.send_chatbox("Thinking...")
                                    self.audio.start_thinking_sound("thinking")
                                    if self._emotion_system:
                                        self._emotion_system.start_thinking()
                            elif part.inline_data:
                                if self._thinking_shown:
                                    self.audio.stop_thinking_sound()
                                    if self._emotion_system:
                                        self._emotion_system.stop_thinking()
                                self._thinking_shown = False
                                if not self._speaking:
                                    self._speaking = True
                                    self._idle_chatbox.stop()
                                    self.osc.set_typing(True)
                                    # Gate outbound audio while model speaks (prevents echo/barge-in)
                                    if self.config.vad_mode == "silero":
                                        await self._gate_audio(session)
                                # Try to start talking animations (idempotent, handles manual animation blocking)
                                if self._emotion_system:
                                    self._emotion_system.start_speaking()
                                # Track last audio time for idle detection
                                self._last_audio_time = time.time()
                                self._last_interaction_time = time.time()
                                self._idle_engagement_sent = False
                                # Keep wanderer paused while AI is speaking
                                if self._wanderer and self._wanderer._paused:
                                    self._wanderer.on_ai_speaking()
                                # When using external TTS, discard Gemini audio
                                if not self._tts:
                                    await self._audio_in_queue.put(part.inline_data.data)

                    # Handle input transcription (user speech) - cumulative stream
                    if (
                        response.server_content
                        and hasattr(response.server_content, "input_transcription")
                        and response.server_content.input_transcription
                    ):
                        input_trans = response.server_content.input_transcription
                        if hasattr(input_trans, "text") and input_trans.text:
                            # User is speaking - mark activity to cancel idle animation
                            self._last_interaction_time = time.time()
                            self._idle_engagement_sent = False
                            if self._emotion_system:
                                self._emotion_system.mark_activity()
                            # Pause wanderer when someone speaks
                            if self._wanderer and self._wanderer.active:
                                self._wanderer.on_speech_activity()
                            # Input transcription arrives as chunks - accumulate like output
                            self._input_transcript_buffer += input_trans.text
                            # Broadcast chunk to WebUI (it appends via +=)
                            _broadcast_console("transcription", input_trans.text, {"streaming": True})
                            # Update logger with full accumulated text (replaces in-place)
                            self._conv_logger.stream_user_message(self._input_transcript_buffer)

                    # Handle output transcription (AI speech) - real-time
                    if (
                        response.server_content
                        and hasattr(response.server_content, "output_transcription")
                        and response.server_content.output_transcription
                    ):
                        transcription = response.server_content.output_transcription
                        if hasattr(transcription, "text") and transcription.text:
                            self._transcript_buffer += transcription.text
                            self._update_chatbox()
                            # Also stream to console in real-time
                            _broadcast_console("response", transcription.text, {"streaming": True})
                            # Feed text to external TTS provider for synthesis
                            if self._tts:
                                self._tts.feed_text(transcription.text)

                    if response.server_content and response.server_content.turn_complete:
                        self._speaking = False
                        # Ungate audio so mic input can flow again
                        self._ungate_audio()
                        if self._thinking_shown:
                            self.audio.stop_thinking_sound()
                            if self._emotion_system:
                                self._emotion_system.stop_thinking()
                        self._thinking_shown = False
                        if self._emotion_system:
                            self._emotion_system.stop_speaking()
                        await self._finalize_chatbox()
                        if self.config.obs_enabled:
                            _broadcast_console("turn_complete", "")
                        # User message already streamed in-place via stream_user_message
                        if self._transcript_buffer.strip():
                            self._conv_logger.add_assistant_message(self._transcript_buffer)
                        self._transcript_buffer = ""
                        # Flush remaining text to TTS provider
                        if self._tts:
                            self._tts.turn_complete()
                        # Delay user finalization to catch late transcription events
                        self._schedule_user_finalize()

                    if response.server_content and response.server_content.interrupted:
                        self._speaking = False
                        # Ungate audio on interruption so user can speak
                        self._ungate_audio()
                        if self._thinking_shown:
                            self.audio.stop_thinking_sound()
                            if self._emotion_system:
                                self._emotion_system.stop_thinking()
                        self._thinking_shown = False
                        if self._emotion_system:
                            self._emotion_system.stop_speaking()
                        self.osc.set_typing(False)
                        if self._transcript_buffer.strip():
                            self._conv_logger.add_assistant_message(self._transcript_buffer)
                        self._transcript_buffer = ""
                        # Interrupt external TTS provider
                        if self._tts:
                            self._tts.interrupt()
                        self._playback_interrupted = True
                        while not self._audio_in_queue.empty():
                            try:
                                self._audio_in_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                    if response.tool_call:
                        self._tool_call_pending = True
                        # Gate audio and flush server buffer before processing tool calls
                        # Per Joe_Hu: sending audio during tool processing causes 1007/1008 disconnects
                        await self._gate_audio(session)
                        # Interrupt TTS -- model will regenerate after tool response
                        if self._tts:
                            self._tts.interrupt()
                        self._playback_interrupted = True
                        # Finalize user message immediately - model has processed input
                        self._conv_logger.finalize_user_message()
                        if self._pending_finalize_task:
                            self._pending_finalize_task.cancel()
                            self._pending_finalize_task = None
                        self._input_transcript_buffer = ""
                        try:
                            responses = []
                            malformed = False
                            for fc in response.tool_call.function_calls:
                                logger.info(f"Tool call: {fc.name}")
                                try:
                                    args_dict = dict(fc.args) if fc.args else {}
                                    args_str = json.dumps(args_dict)
                                except (TypeError, ValueError) as e:
                                    # Malformed function call -- invalid JSON args from model
                                    logger.warning(f"Malformed tool call args for {fc.name}: {e}")
                                    _broadcast_console("error", f"Malformed tool call: {fc.name} ({e})")
                                    malformed = True
                                    responses.append(types.FunctionResponse(
                                        id=fc.id, name=fc.name,
                                        response={"result": "error", "message": "malformed arguments, please retry"},
                                    ))
                                    continue
                                _broadcast_console("tool_call", f"{fc.name}({args_str})")
                                self._usage_metadata["tool_calls"] += 1
                                self._conv_logger.add_tool_call(fc.name, args_dict)
                                fr = await self.tool_handler.handle(fc)
                                result_dict = fr.response if fr.response else {}
                                result_str = json.dumps(result_dict)
                                _broadcast_console("tool_response", f"{fc.name} → {result_str}")
                                self._conv_logger.add_tool_response(fc.name, result_dict)
                                responses.append(fr)
                            await session.send_tool_response(function_responses=responses)
                            # Nudge model to recover after malformed tool call
                            if malformed:
                                logger.info("Sending recovery nudge after malformed tool call")
                                await session.send_realtime_input(
                                    text="System update - your last tool call had malformed arguments. Please try again."
                                )
                        finally:
                            self._tool_call_pending = False
                            # Ungate audio after tool response sent
                            self._ungate_audio()

                    # Track usage metadata if available
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        um = response.usage_metadata
                        prev_prompt = self._usage_metadata.get("prompt_tokens", 0)
                        if hasattr(um, "prompt_token_count") and um.prompt_token_count:
                            self._usage_metadata["prompt_tokens"] = um.prompt_token_count
                            # Detect context compression (prompt tokens decreased significantly)
                            if prev_prompt > 0 and um.prompt_token_count < prev_prompt * 0.8:
                                logger.info(f"Context compressed: {prev_prompt} -> {um.prompt_token_count} prompt tokens")
                                _broadcast_console("info", f"Context compressed: {prev_prompt} -> {um.prompt_token_count} tokens")
                            # Check if custom compression should trigger
                            self._check_custom_compression(um.prompt_token_count)
                        if hasattr(um, "response_token_count") and um.response_token_count:
                            self._usage_metadata["response_tokens"] = um.response_token_count
                        if hasattr(um, "total_token_count") and um.total_token_count:
                            self._usage_metadata["total_tokens"] = um.total_token_count

                    if response.go_away and not self._reconnect_requested:
                        time_left = response.go_away.time_left
                        logger.warning(
                            f"Server disconnecting in {time_left}"
                        )
                        # Parse seconds from time_left
                        try:
                            if hasattr(time_left, 'total_seconds'):
                                tl_seconds = time_left.total_seconds()
                            elif hasattr(time_left, 'seconds'):
                                tl_seconds = time_left.seconds
                            else:
                                import re
                                m = re.search(r'(\d+)', str(time_left))
                                tl_seconds = int(m.group(1)) if m else 30
                        except Exception:
                            tl_seconds = 30
                        # Wait most of the remaining time, then reconnect fresh
                        wait = max(tl_seconds - 10, 5)
                        logger.info(f"Will reconnect in {wait}s (using remaining session time)")
                        _broadcast_console("info", f"GoAway: reconnecting in {wait}s")
                        # Save current transcript before reconnecting
                        if self._transcript_buffer.strip():
                            self._conv_logger.add_assistant_message(self._transcript_buffer)
                            self._transcript_buffer = ""
                        self._conv_logger.finalize_user_message()
                        self._conv_logger._save_async()
                        # Schedule delayed reconnect so receive loop keeps working
                        asyncio.create_task(self._delayed_reconnect(wait))

                    if (
                        hasattr(response, "session_resumption_update")
                        and response.session_resumption_update
                    ):
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            self._save_session_handle(update.new_handle)

            except asyncio.CancelledError:
                return
            except (ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"Receive loop connection error: {e}")
                raise
            except Exception as e:
                logger.error(f"Receive error: {e}")
                raise

    def _update_chatbox(self):
        """Update chatbox with real-time transcription (truncates if too long)."""
        # Skip chatbox updates if music is playing (Now Playing display takes over)
        if self.audio.is_music_playing():
            return
        music_gen = getattr(self.tool_handler, 'music_gen', None)
        if music_gen and music_gen.is_active:
            return
        
        text = self._strip_audio_tags_for_chatbox(self._transcript_buffer)
        if not text:
            return
        # For real-time updates, just show the last 144 chars (no pagination)
        if len(text) <= 144:
            self.osc.send_chatbox(text)
        else:
            # Show last 141 chars with ellipsis at start
            truncated = "..." + text[-141:]
            self.osc.send_chatbox(truncated)

    async def _finalize_chatbox(self):
        """Finalize chatbox with pagination when AI finishes speaking."""
        # Skip chatbox updates if music is playing
        if self.audio.is_music_playing():
            self.osc.set_typing(False)
            return
        music_gen = getattr(self.tool_handler, 'music_gen', None)
        if music_gen and music_gen.is_active:
            self.osc.set_typing(False)
            return
        
        text = self._strip_audio_tags_for_chatbox(self._transcript_buffer)
        if not text:
            self.osc.set_typing(False)
            return
        pages = self.osc.send_chatbox_paginated(text)
        if len(pages) > 1:
            await self.osc.display_pages(pages, self.config.chatbox_page_delay)
        self.osc.set_typing(False)

    @staticmethod
    def _strip_audio_tags_for_chatbox(text: str) -> str:
        """Remove inline expressive audio tags (for example [whispers]) from chatbox text only."""
        if not text:
            return ""
        cleaned = re.sub(r"\[(?:[A-Za-z][A-Za-z\s,'-]{0,40})\]", " ", text)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        # Convert markdown italics to Unicode small caps
        cleaned = GeminiLiveSession._convert_markdown_italics_to_unicode(cleaned)
        return cleaned

    @staticmethod
    def _convert_markdown_italics_to_unicode(text: str) -> str:
        """Convert *text* markdown italics to Unicode small caps for VRChat chatbox display."""
        if not text or "*" not in text:
            return text
        
        # Unicode small caps mapping
        small_caps_map = {
            "A": "ᴀ", "B": "ʙ", "C": "ᴄ", "D": "ᴅ", "E": "ᴇ", "F": "ꜰ", "G": "ɢ", "H": "ʜ",
            "I": "ɪ", "J": "ᴊ", "K": "ᴋ", "L": "ʟ", "M": "ᴍ", "N": "ɴ", "O": "ᴏ", "P": "ᴘ",
            "Q": "Q", "R": "ʀ", "S": "ꜱ", "T": "ᴛ", "U": "ᴜ", "V": "ᴠ", "W": "ᴡ", "X": "X",
            "Y": "ʏ", "Z": "ᴢ",
            "a": "ᴀ", "b": "ʙ", "c": "ᴄ", "d": "ᴅ", "e": "ᴇ", "f": "ꜰ", "g": "ɢ", "h": "ʜ",
            "i": "ɪ", "j": "ᴊ", "k": "ᴋ", "l": "ʟ", "m": "ᴍ", "n": "ɴ", "o": "ᴏ", "p": "ᴘ",
            "q": "q", "r": "ʀ", "s": "ꜱ", "t": "ᴛ", "u": "ᴜ", "v": "ᴠ", "w": "ᴡ", "x": "x",
            "y": "ʏ", "z": "ᴢ",
        }
        
        def convert_to_small_caps(match):
            content = match.group(1)
            return "".join(small_caps_map.get(ch, ch) for ch in content)
        
        # Replace all *text* with small caps version
        result = re.sub(r"\*([^*]+)\*", convert_to_small_caps, text)
        return result

    @staticmethod
    def _normalize_song_name(name: str) -> str:
        """Clean up a filename-based song name for display."""
        import re
        name = name.replace("_", " ").replace("-", " ")
        name = re.sub(r"\s+", " ", name).strip()
        name = name.title()
        return name

    def _format_now_playing(self, progress_info: dict) -> str:
        """Format Now Playing display for chatbox."""
        name = self._normalize_song_name(progress_info["song_name"])
        position = progress_info["position"]
        duration = progress_info["duration"]
        progress = progress_info["progress"]
        
        # Format times
        pos_min, pos_sec = divmod(int(position), 60)
        dur_min, dur_sec = divmod(int(duration), 60)
        time_str = f"{pos_min}:{pos_sec:02d} / {dur_min}:{dur_sec:02d}"
        
        # Create progress bar with sub-block transitions
        bar_width = 14
        exact = progress * bar_width
        filled = int(exact)
        fraction = exact - filled
        if filled >= bar_width:
            bar = "\u2588" * bar_width
        else:
            if fraction < 0.25:
                transition = "\u2591"
            elif fraction < 0.5:
                transition = "\u2592"
            elif fraction < 0.75:
                transition = "\u2593"
            else:
                transition = "\u2588"
            bar = "\u2588" * filled + transition + "\u2591" * (bar_width - filled - 1)
        
        # Get current lyric
        lyric = self.audio.get_current_lyric()
        
        lines = []
        if lyric:
            max_lyric = 100
            if len(lyric) > max_lyric:
                lyric = lyric[:max_lyric - 3] + "..."
            lines.append("LYRICS")
            lines.append(lyric)
            lines.append("────────────")
        
        # Truncate song name if needed
        max_name = 100
        if len(name) > max_name:
            name = name[:max_name - 3] + "..."
        
        lines.append(name)
        lines.append(bar)
        lines.append(time_str)
        
        return "\n".join(lines)

    def _format_music_gen_display(self, music_gen) -> str:
        """Format Lyria music gen display for chatbox."""
        elapsed = int(music_gen.elapsed)
        prompts = music_gen.current_prompts

        # Format tags like [Acoustic Guitar] [Solo]
        tags = " ".join(f"[{p['text']}]" for p in prompts)

        # Format elapsed time
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        if hours > 0:
            time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        elif minutes > 0:
            time_str = f"{minutes}:{seconds:02d}"
        else:
            time_str = f"0:{seconds:02d}"

        divider_char = self.config.get("vrchat", "idle_chatbox", "divider", default="\u2500")
        divider_length = self.config.get("vrchat", "idle_chatbox", "divider_length", default=14)
        divider = str(divider_char) * int(divider_length)

        lines = []
        if music_gen.is_paused:
            lines.append("\u23f8 PAUSED")
        else:
            lines.append("\u266b Live Music")
        if tags:
            lines.append(tags)
        lines.append(divider)
        lines.append(time_str)

        text = "\n".join(lines)
        if len(text) > 144:
            # Truncate tags if needed to fit chatbox limit
            max_tags = 144 - len(lines[0]) - len(divider) - len(time_str) - 3
            if max_tags > 3:
                tags = tags[:max_tags - 3] + "..."
            lines = [lines[0], tags, divider, time_str]
            text = "\n".join(lines)
            if len(text) > 144:
                text = text[:144]
        return text

    async def _now_playing_loop(self):
        """Background task that updates chatbox with Now Playing when music plays."""
        while True:
            try:
                progress = self.audio.get_music_progress()
                if progress:
                    display = self._format_now_playing(progress)
                    self.osc.send_chatbox(display)
                else:
                    music_gen = getattr(self.tool_handler, 'music_gen', None)
                    if music_gen and music_gen.is_active:
                        display = self._format_music_gen_display(music_gen)
                        self.osc.send_chatbox(display)
                await asyncio.sleep(1.3)
            except Exception as e:
                logger.error(f"Now Playing loop error: {e}")
                await asyncio.sleep(1.0)
