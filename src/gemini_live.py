import asyncio
import base64
import io
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import APIError
import mss
from PIL import Image
from src.tools import get_tool_declarations, ToolHandler

logger = logging.getLogger(__name__)

SESSION_HANDLE_FILE = Path("session_handle.txt")
SESSION_EXPIRY_HOURS = 2


class GeminiLiveSession:
    def __init__(self, config, audio_mgr, osc, tracker, personality_mgr):
        self.config = config
        self.audio = audio_mgr
        self.osc = osc
        self.personality = personality_mgr
        self.tool_handler = ToolHandler(audio_mgr, osc, tracker, personality_mgr)
        self._speaking = False
        self._transcript_buffer = ""
        self._session_handle = None
        self._handle_fail_count = 0
        self._out_queue = asyncio.Queue(maxsize=5)
        self._audio_in_queue = asyncio.Queue()
        self._reconnect_requested = False
        self._mic_muted = False
        self._session = None
        self._usage_metadata = {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "tool_calls": 0,
        }
        self._load_session_handle()

    def request_reconnect(self):
        """Request a reconnect on next iteration."""
        self._reconnect_requested = True
        logger.info("Reconnect requested via control panel")

    def set_mic_muted(self, muted: bool):
        """Set mic mute state."""
        self._mic_muted = muted
        logger.info(f"Mic mute set to {muted}")

    async def send_text(self, text: str):
        """Send text to the model via client content."""
        if self._session:
            try:
                await self._session.send_client_content(
                    turns=[types.Content(role="user", parts=[types.Part.from_text(text=text)])]
                )
                logger.info(f"Sent text to model: {text[:50]}...")
            except Exception as e:
                logger.error(f"Failed to send text: {e}")

    def _build_config(self):
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
        if self.config.language:
            transcription_config = types.AudioTranscriptionConfig(language=self.config.language)

        config_kwargs = dict(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part.from_text(
                    text=self.config.build_system_instruction(self.personality)
                )]
            ),
            tools=get_tool_declarations(),
            output_audio_transcription=transcription_config,
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
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

        if self.config.temperature is not None:
            config_kwargs["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            config_kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            config_kwargs["top_k"] = self.config.top_k
        if self.config.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.config.max_output_tokens
        if self.config.enable_affective_dialog is not None:
            config_kwargs["enable_affective_dialog"] = self.config.enable_affective_dialog
        if self.config.proactivity is not None:
            config_kwargs["proactivity"] = self.config.proactivity

        return types.LiveConnectConfig(**config_kwargs)

    def _load_session_handle(self):
        if not SESSION_HANDLE_FILE.exists():
            return
        try:
            data = json.loads(SESSION_HANDLE_FILE.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(data["created"])
            if datetime.now() - created < timedelta(hours=SESSION_EXPIRY_HOURS):
                self._session_handle = data["handle"]
                logger.info(f"Loaded session handle (created {created.strftime('%H:%M:%S')})")
            else:
                logger.info("Session handle expired, will create new session")
                self._clear_session_handle()
        except Exception as e:
            logger.warning(f"Failed to load session handle: {e}")
            self._clear_session_handle()

    def _save_session_handle(self, handle: str):
        self._session_handle = handle
        self._handle_fail_count = 0
        data = {"handle": handle, "created": datetime.now().isoformat()}
        SESSION_HANDLE_FILE.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Saved new session handle")

    def _clear_session_handle(self):
        self._session_handle = None
        self._handle_fail_count = 0
        if SESSION_HANDLE_FILE.exists():
            SESSION_HANDLE_FILE.unlink()
            logger.info("Cleared session handle")

    async def run(self):
        while True:
            self._reconnect_requested = False
            try:
                client = genai.Client(api_key=self.config.api_key)
                live_config = self._build_config()
                if self._session_handle:
                    logger.info(f"Connecting to Gemini Live with session resumption...")
                else:
                    logger.info(f"Connecting to Gemini Live ({self.config.model})...")

                async with client.aio.live.connect(
                    model=self.config.model,
                    config=live_config,
                ) as session:
                    logger.info("Connected to Gemini Live")
                    self._session = session
                    self._handle_fail_count = 0
                    self.tool_handler.session = session
                    self._out_queue = asyncio.Queue(maxsize=5)
                    self._audio_in_queue = asyncio.Queue()
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
                        ]
                        if self.config.vision_enabled:
                            tasks.append(self._capture_screen_loop())
                            logger.info(f"Screen capture enabled (monitor {self.config.vision_monitor})")
                        await asyncio.gather(*tasks)
                    finally:
                        self._session = None
                        input_stream.close()
                        output_stream.close()

            except APIError as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    old_key = self.config.api_key
                    new_key = self.config.rotate_key()
                    if new_key != old_key:
                        logger.warning("Rate limited — switched API key")
                    else:
                        logger.warning("Rate limited — waiting 5s before retry")
                        await asyncio.sleep(5)
                    continue
                
                # 1007 is WebSocket protocol error - reconnect immediately
                if "1007" in str(e):
                    logger.warning("WebSocket protocol error (1007), reconnecting immediately...")
                    continue
                
                if self._session_handle:
                    self._handle_fail_count += 1
                    if self._handle_fail_count >= 2:
                        logger.warning("Session handle failed twice, clearing and using new session")
                        self._clear_session_handle()
                    else:
                        logger.warning(f"Session handle may be invalid (attempt {self._handle_fail_count}/2)")
                logger.error(f"API error: {e}")
                await asyncio.sleep(2)
            except Exception as e:
                err_str = str(e)
                
                # 1007 WebSocket errors - reconnect immediately
                if "1007" in err_str:
                    logger.warning("WebSocket protocol error (1007), reconnecting immediately...")
                    continue
                
                if self._session_handle:
                    self._handle_fail_count += 1
                    if self._handle_fail_count >= 2:
                        logger.warning("Session handle failed twice, clearing and using new session")
                        self._clear_session_handle()
                    else:
                        logger.warning(f"Session handle may be invalid (attempt {self._handle_fail_count}/2)")
                logger.error(f"Session error: {e}")
                await asyncio.sleep(2)

    async def _reconnect_monitor_loop(self):
        """Monitor for reconnect requests from control panel."""
        while True:
            if self._reconnect_requested:
                logger.info("Processing reconnect request...")
                raise Exception("Reconnect requested")
            await asyncio.sleep(0.5)

    async def _listen_audio_loop(self, input_stream):
        while True:
            try:
                data = await asyncio.to_thread(
                    input_stream.read,
                    self.config.chunk_size,
                    exception_on_overflow=False,
                )
                # Only send audio if mic is not muted
                if not self._mic_muted:
                    await self._out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except Exception as e:
                logger.error(f"Audio listen error: {e}")
                raise

    async def _send_realtime_loop(self, session):
        while True:
            try:
                msg = await self._out_queue.get()
                if msg["mime_type"] == "audio/pcm":
                    await session.send_realtime_input(audio=msg)
                else:
                    await session.send_realtime_input(media=msg)
            except Exception as e:
                logger.error(f"Send realtime error: {e}")
                raise

    async def _play_audio_loop(self, output_stream):
        while True:
            try:
                audio_data = await self._audio_in_queue.get()
                # process_output_audio handles boost/distortion AND music fade
                audio_data = self.audio.process_output_audio(audio_data)
                if audio_data:  # Only write if not completely muted
                    await asyncio.to_thread(output_stream.write, audio_data)
            except Exception as e:
                logger.error(f"Audio play error: {e}")
                raise

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
                return {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buffer.read()).decode(),
                }
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
                        self._out_queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        try:
                            self._out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            self._out_queue.put_nowait(frame)
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
                            if part.inline_data:
                                self._speaking = True
                                self.osc.set_typing(True)
                                # Audio processing (boost + music fade) happens in _play_audio_loop
                                await self._audio_in_queue.put(part.inline_data.data)

                    if (
                        response.server_content
                        and hasattr(response.server_content, "output_transcription")
                        and response.server_content.output_transcription
                    ):
                        transcription = response.server_content.output_transcription
                        if hasattr(transcription, "text") and transcription.text:
                            self._transcript_buffer += transcription.text
                            self._update_chatbox()

                    if response.server_content and response.server_content.turn_complete:
                        self._speaking = False
                        await self._finalize_chatbox()
                        self._transcript_buffer = ""

                    if response.server_content and response.server_content.interrupted:
                        self._speaking = False
                        self.osc.set_typing(False)
                        self._transcript_buffer = ""
                        while not self._audio_in_queue.empty():
                            try:
                                self._audio_in_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                    if response.tool_call:
                        responses = []
                        for fc in response.tool_call.function_calls:
                            logger.info(f"Tool call: {fc.name}")
                            self._usage_metadata["tool_calls"] += 1
                            fr = await self.tool_handler.handle(fc)
                            responses.append(fr)
                        await session.send_tool_response(function_responses=responses)

                    # Track usage metadata if available
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        um = response.usage_metadata
                        if hasattr(um, "prompt_token_count"):
                            self._usage_metadata["prompt_tokens"] = um.prompt_token_count
                        if hasattr(um, "response_token_count"):
                            self._usage_metadata["response_tokens"] = um.response_token_count
                        if hasattr(um, "total_token_count"):
                            self._usage_metadata["total_tokens"] = um.total_token_count

                    if response.go_away:
                        logger.warning(
                            f"Server disconnecting in {response.go_away.time_left}"
                        )

                    if (
                        hasattr(response, "session_resumption_update")
                        and response.session_resumption_update
                    ):
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            self._save_session_handle(update.new_handle)

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

    def _format_now_playing(self, progress_info: dict) -> str:
        """Format Now Playing display for chatbox.
        
        Format:
        ♪ Song Name ♪
        ──────•───────────
        0:45 / 3:21
        """
        name = progress_info["name"]
        position = progress_info["position"]
        duration = progress_info["duration"]
        progress = progress_info["progress"]
        
        # Format times
        pos_min, pos_sec = divmod(int(position), 60)
        dur_min, dur_sec = divmod(int(duration), 60)
        time_str = f"{pos_min}:{pos_sec:02d} / {dur_min}:{dur_sec:02d}"
        
        # Create progress bar (use remaining space for max width)
        # Title line: ♪ Song Name ♪
        # Leave room for ball position
        bar_width = 12
        filled = int(progress * bar_width)
        
        # Build progress bar: | at edges, ─ for track, • for position
        bar = "|" + "─" * filled + "•" + "─" * (bar_width - filled - 1) + "|"
        
        # Truncate song name if too long (leave room for ♪ symbols)
        max_name_len = 140 - len(time_str) - len(bar) - 10
        if len(name) > max_name_len:
            name = name[:max_name_len-3] + "..."
        
        return f"♪ {name} ♪\n{bar}\n{time_str}"

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
