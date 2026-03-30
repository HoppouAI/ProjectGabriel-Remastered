import asyncio
import base64
import io
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
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
        self._rate_limit_backoff = 0
        self._tool_call_pending = False
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
        self._pending_finalize_task = None
        self._wanderer = None  # Set externally from main.py
        self._usage_metadata = {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "tool_calls": 0,
        }
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
        """Request a reconnect on next iteration."""
        self._reconnect_requested = True
        logger.info("Reconnect requested via control panel")

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
                self._conv_logger.add_user_message(text)
                logger.info(f"Sent text to model: {text[:50]}...")
            except Exception as e:
                logger.error(f"Failed to send text: {e}")

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
        """Check if any v1alpha-only features are enabled."""
        return (self.config.enable_affective_dialog is not None 
                or self.config.proactivity is not None)

    def _build_config(self, skip_alpha_features=False):
        start_sens_map = {
            "START_SENSITIVITY_LOW": types.StartSensitivity.START_SENSITIVITY_LOW,
            "START_SENSITIVITY_HIGH": types.StartSensitivity.START_SENSITIVITY_HIGH,
        }
        end_sens_map = {
            "END_SENSITIVITY_LOW": types.EndSensitivity.END_SENSITIVITY_LOW,
            "END_SENSITIVITY_HIGH": types.EndSensitivity.END_SENSITIVITY_HIGH,
        }

        vad_config = types.AutomaticActivityDetection(
            disabled=self.config.vad_disabled,
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
        if not skip_alpha_features:
            if self.config.enable_affective_dialog is not None:
                config_kwargs["enable_affective_dialog"] = self.config.enable_affective_dialog
            if self.config.proactivity is not None:
                config_kwargs["proactivity"] = self.config.proactivity

        # Context window compression
        if self.config.compression_enabled:
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

        # Thinking configuration
        thinking_budget = self.config.thinking_budget
        include_thoughts = self.config.thinking_include_thoughts
        if thinking_budget is not None or include_thoughts:
            thinking_kwargs = {}
            if thinking_budget is not None:
                thinking_kwargs["thinking_budget"] = thinking_budget
            if include_thoughts:
                thinking_kwargs["include_thoughts"] = True
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

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
            self._reconnect_requested = False
            # Check for expired session handle before each connection attempt
            if self._is_session_handle_expired():
                logger.info("Session handle expired (2h), starting fresh session")
                _broadcast_console("info", "Session handle expired, starting fresh session")
                self._clear_session_handle()
            try:
                use_alpha = self._needs_alpha_api() and not self._alpha_fallback_failed
                if use_alpha:
                    client = genai.Client(
                        api_key=self.config.api_key,
                        http_options={"api_version": "v1alpha"},
                    )
                    live_config = self._build_config()
                    logger.info("Using v1alpha API for affective dialog / proactivity")
                else:
                    client = genai.Client(api_key=self.config.api_key)
                    live_config = self._build_config(skip_alpha_features=self._alpha_fallback_failed)
                if self._session_handle:
                    logger.info(f"Connecting to Gemini Live with session resumption...")
                elif self._resumption_fail_streak >= 3:
                    logger.info(f"Connecting to Gemini Live ({self.config.model}) [resumption disabled after {self._resumption_fail_streak} failures]...")
                else:
                    logger.info(f"Connecting to Gemini Live ({self.config.model})...")

                async with client.aio.live.connect(
                    model=self.config.model,
                    config=live_config,
                ) as session:
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
                    self._rate_limit_backoff = 0
                    self._last_interaction_time = time.time()
                    self._idle_engagement_sent = False
                    self.tool_handler.session = session
                    self.tool_handler.live_session = self
                    self._out_queue = asyncio.Queue(maxsize=5)
                    self._audio_in_queue = asyncio.Queue()
                    self._stream_closing = False
                    input_stream = self.audio.open_input_stream()
                    output_stream = self.audio.open_output_stream()
                    try:
                        tasks = [
                            self._listen_audio_loop(input_stream),
                            self._send_realtime_loop(session),
                            self._receive_loop(session),
                            self._play_audio_loop(output_stream),
                            self._reconnect_monitor_loop(),
                            self._now_playing_loop(),
                            self._idle_check_loop(),
                        ]
                        # Always start TTS audio loop (supports hot-swap)
                        tasks.append(self._tts_audio_loop())
                        if self.config.vision_enabled:
                            tasks.append(self._capture_screen_loop())
                            logger.info(f"Screen capture enabled (monitor {self.config.vision_monitor})")
                        await asyncio.gather(*tasks)
                    finally:
                        self._session = None
                        self._stream_closing = True
                        self._idle_chatbox.stop()
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
                    # Check if session crashed quickly (within 15s) - likely handle issue
                    session_was_short = self._connection_start_time > 0 and (time.time() - self._connection_start_time) < 15
                    # 1007 (invalid argument) - retry with handle, only clear after 3 failures
                    if "1007" in err_str and self._session_handle:
                        self._handle_fail_count += 1
                        logger.warning(f"1007 invalid argument (attempt {self._handle_fail_count}/3, keeping handle)")
                        if self._handle_fail_count >= 3:
                            logger.warning("Clearing session handle after 3 consecutive 1007 errors")
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
                        if self._handle_fail_count >= 3:
                            logger.warning("Session handle failed 3 times after WS errors, clearing")
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
                # Check if session crashed quickly (within 15s) - likely handle issue
                session_was_short = self._connection_start_time > 0 and (time.time() - self._connection_start_time) < 15
                # 1007 (invalid argument) - retry with handle, only clear after 3 failures
                if code == 1007 and self._session_handle:
                    self._handle_fail_count += 1
                    logger.warning(f"1007 invalid argument (attempt {self._handle_fail_count}/3, keeping handle)")
                    if self._handle_fail_count >= 3:
                        logger.warning("Clearing session handle after 3 consecutive 1007 errors")
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
                    if self._handle_fail_count >= 3:
                        logger.warning("Session handle failed 3 times, clearing")
                        self._clear_session_handle()
                await asyncio.sleep(0.5)
                continue

            except (ConnectionError, OSError, TimeoutError) as e:
                # Network-level errors - keep handle, just retry
                logger.warning(f"Network error: {e}, reconnecting in 3s...")
                _broadcast_console("error", f"Network error: {str(e)[:80]}")
                self._notify_chatbox_error()
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
            # Don't trigger idle during music playback
            if self._emotion_system:
                if music_playing:
                    self._emotion_system.mark_activity()
                else:
                    self._emotion_system.check_idle()
            # Start idle chatbox when idle and no music playing
            if not self._speaking and not music_playing:
                emo = self._emotion_system
                if (emo and emo._idle_active) or not emo:
                    self._idle_chatbox.start()
            elif music_playing:
                self._idle_chatbox.stop()
            # Idle engagement - prompt model to speak after long silence
            if (
                not self._idle_engagement_sent
                and not self._speaking
                and not music_playing
                and time.time() - self._last_interaction_time >= IDLE_ENGAGEMENT_SECONDS
            ):
                self._idle_engagement_sent = True
                logger.info(f"Idle for {IDLE_ENGAGEMENT_SECONDS}s, sending engagement prompt")
                _broadcast_console("info", "Sending idle engagement prompt")
                await self.send_text(
                    "[System: You have been idle for a while. "
                    "Try to engage nearby people in conversation - "
                    "say something interesting, ask a question, or make an observation to get someone to talk to you.]"
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

    async def _send_realtime_loop(self, session):
        while True:
            try:
                msg_type, data = await self._out_queue.get()
                if self._tool_call_pending:
                    continue
                if msg_type == "audio":
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                elif msg_type == "video":
                    await session.send_realtime_input(
                        media=types.Blob(data=data, mime_type="image/jpeg")
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
        while True:
            try:
                audio_data = await self._audio_in_queue.get()
                if self._stream_closing:
                    return
                audio_data = self.audio.process_output_audio(audio_data)
                if audio_data:
                    await asyncio.to_thread(output_stream.write, audio_data)
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
                if img.width > max_size or img.height > max_size:
                    img.thumbnail([max_size, max_size])
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=self.config.vision_quality)
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
        try:
            while True:
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
                await asyncio.sleep(interval)
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
                        while not self._audio_in_queue.empty():
                            try:
                                self._audio_in_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                    if response.tool_call:
                        self._tool_call_pending = True
                        # Interrupt TTS -- model will regenerate after tool response
                        if self._tts:
                            self._tts.interrupt()
                        # Finalize user message immediately - model has processed input
                        self._conv_logger.finalize_user_message()
                        if self._pending_finalize_task:
                            self._pending_finalize_task.cancel()
                            self._pending_finalize_task = None
                        self._input_transcript_buffer = ""
                        try:
                            responses = []
                            for fc in response.tool_call.function_calls:
                                logger.info(f"Tool call: {fc.name}")
                                args_dict = dict(fc.args) if fc.args else {}
                                args_str = json.dumps(args_dict)
                                _broadcast_console("tool_call", f"{fc.name}({args_str[:100]})")
                                self._usage_metadata["tool_calls"] += 1
                                self._conv_logger.add_tool_call(fc.name, args_dict)
                                fr = await self.tool_handler.handle(fc)
                                result_dict = fr.response if fr.response else {}
                                result_str = json.dumps(result_dict)
                                _broadcast_console("tool_response", f"{fc.name} → {result_str[:150]}")
                                self._conv_logger.add_tool_response(fc.name, result_dict)
                                responses.append(fr)
                            await session.send_tool_response(function_responses=responses)
                        finally:
                            self._tool_call_pending = False

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
                        if hasattr(um, "response_token_count") and um.response_token_count:
                            self._usage_metadata["response_tokens"] = um.response_token_count
                        if hasattr(um, "total_token_count") and um.total_token_count:
                            self._usage_metadata["total_tokens"] = um.total_token_count

                    if response.go_away:
                        logger.warning(
                            f"Server disconnecting in {response.go_away.time_left}"
                        )
                        _broadcast_console("info", f"GoAway received, reconnecting in {response.go_away.time_left}")
                        self._reconnect_requested = True

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
        
        text = self._transcript_buffer.strip()
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
        
        text = self._transcript_buffer.strip()
        if not text:
            self.osc.set_typing(False)
            return
        pages = self.osc.send_chatbox_paginated(text)
        if len(pages) > 1:
            await self.osc.display_pages(pages, self.config.chatbox_page_delay)
        self.osc.set_typing(False)

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

    async def _now_playing_loop(self):
        """Background task that updates chatbox with Now Playing when music plays."""
        while True:
            try:
                progress = self.audio.get_music_progress()
                if progress:
                    display = self._format_now_playing(progress)
                    self.osc.send_chatbox(display)
                await asyncio.sleep(1.3)  # Same as chatbox rate limit
            except Exception as e:
                logger.error(f"Now Playing loop error: {e}")
                await asyncio.sleep(1.0)
