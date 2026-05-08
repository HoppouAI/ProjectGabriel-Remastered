import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import APIError
from websockets.exceptions import ConnectionClosed
from src.tools import get_tool_declarations, ToolHandler
from src.emotions import init_emotion_system, get_emotion_system
from src.idle_chatbox import IdleChatbox
from .conversation_logger import ConversationLogger, CONVERSATION_DIR
from .chatbox import ChatboxFormattersMixin
from .config_builder import ConfigBuilderMixin
from .audio import AudioLoopsMixin
from .receive import ReceiveLoopMixin

logger = logging.getLogger(__name__)

SESSION_HANDLE_FILE = Path("session_handle.txt")
SESSION_EXPIRY_HOURS = 2
IDLE_ENGAGEMENT_SECONDS = 3600  # 1 hour


def _broadcast_console(log_type: str, content: str, extra: dict = None):
    """Broadcast a log entry to the control panel console."""
    try:
        from control_server import add_console_log
        add_console_log(log_type, content, extra)
    except ImportError:
        pass
    except Exception:
        pass


class GeminiLiveSession(ReceiveLoopMixin, AudioLoopsMixin, ConfigBuilderMixin, ChatboxFormattersMixin):
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
        self._conv_logger = ConversationLogger(enabled=config.conversation_logging_enabled)
        
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
                try:
                    from src.plugins import emit_event
                    emit_event("message_in", text, "text")
                except Exception as e:
                    logger.debug(f"plugin message_in dispatch failed: {e}")
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
            final_text = self._input_transcript_buffer.strip()
            self._conv_logger.finalize_user_message()
            self._input_transcript_buffer = ""
            if self.config.obs_enabled:
                _broadcast_console("user_turn_complete", "")
            if final_text:
                try:
                    from src.plugins import emit_event
                    emit_event("message_in", final_text, "vrchat")
                except Exception as e:
                    logger.debug(f"plugin message_in dispatch failed: {e}")
        except asyncio.CancelledError:
            pass
        finally:
            self._pending_finalize_task = None

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
                    logger.debug(f"Silero VAD enabled (threshold={self.config.vad_silero_threshold}, silence={self.config.vad_silence_duration_ms}ms)")
                    logger.debug("Audio gating active: outbound suppressed during model speech and tool calls")
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
                    self._conv_logger = ConversationLogger(enabled=self.config.conversation_logging_enabled)
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
                            logger.debug(f"Screen capture enabled (monitor {self.config.vision_monitor})")
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
            # Check if Suno song is streaming
            suno = getattr(self.tool_handler, 'suno', None)
            suno_active = suno.is_active if suno else False
            # Anything keeping the AI busy suppresses idle
            busy = music_playing or tracker_active or music_gen_active or suno_active
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

    async def _now_playing_loop(self):
        """Background task that updates chatbox with Now Playing when music plays."""
        while True:
            try:
                progress = self.audio.get_music_progress()
                if progress:
                    display = self._format_now_playing(progress)
                    self.osc.send_chatbox(display)
                else:
                    suno = getattr(self.tool_handler, 'suno', None)
                    suno_progress = suno.get_progress() if suno else None
                    if suno_progress:
                        display = self._format_suno_now_playing(suno_progress)
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
