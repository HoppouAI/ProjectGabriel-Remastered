import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

SESSION_HANDLE_FILE = Path("discord_bot/data/session_handle.txt")
SESSION_EXPIRY_HOURS = 2


class GeminiTextSession:
    """Gemini Live session for Discord bot using AUDIO modality with transcription.

    Uses AUDIO response modality (required by Gemini Live) but captures the
    output_audio_transcription for text. Audio data is discarded since we only
    need the transcription for Discord messages.
    """

    def __init__(self, config, tool_handler, personality_mgr=None):
        self.config = config
        self.tool_handler = tool_handler
        self.personality = personality_mgr
        self._session = None
        self._session_handle = None
        self._session_handle_created = None
        self._handle_fail_count = 0
        self._rate_limit_backoff = 0
        self._connected = asyncio.Event()
        self._response_queue = asyncio.Queue()
        self._pending_responses = {}  # request_id -> asyncio.Future
        self._request_counter = 0
        self._receive_task = None
        self._reconnect_lock = asyncio.Lock()
        self._closing = False
        self._load_session_handle()

    def _build_config(self):
        transcription_config = types.AudioTranscriptionConfig()

        config_kwargs = dict(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part.from_text(
                    text=self.config.build_system_instruction(self.personality)
                )]
            ),
            tools=self.tool_handler.get_declarations(),
            output_audio_transcription=transcription_config,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.config.voice
                    )
                )
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

        # Context window compression
        if self.config.compression_enabled:
            sw_kwargs = {}
            if self.config.compression_target_tokens is not None:
                sw_kwargs["target_tokens"] = self.config.compression_target_tokens
            cw_kwargs = {"sliding_window": types.SlidingWindow(**sw_kwargs)}
            if self.config.compression_trigger_tokens is not None:
                cw_kwargs["trigger_tokens"] = self.config.compression_trigger_tokens
            config_kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(**cw_kwargs)

        # Thinking
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
                logger.info(f"Loaded Discord session handle (created {created.strftime('%H:%M:%S')})")
            else:
                self._clear_session_handle()
        except Exception as e:
            logger.warning(f"Failed to load Discord session handle: {e}")
            self._clear_session_handle()

    def _save_session_handle(self, handle):
        self._session_handle = handle
        self._session_handle_created = datetime.now()
        self._handle_fail_count = 0
        SESSION_HANDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"handle": handle, "created": self._session_handle_created.isoformat()}
        SESSION_HANDLE_FILE.write_text(json.dumps(data), encoding="utf-8")

    def _clear_session_handle(self):
        self._session_handle = None
        self._session_handle_created = None
        self._handle_fail_count = 0
        if SESSION_HANDLE_FILE.exists():
            SESSION_HANDLE_FILE.unlink()

    def _is_handle_expired(self):
        if not self._session_handle or not self._session_handle_created:
            return False
        return datetime.now() - self._session_handle_created >= timedelta(hours=SESSION_EXPIRY_HOURS)

    async def connect(self):
        """Connect to Gemini Live and start the receive loop."""
        if self._is_handle_expired():
            self._clear_session_handle()

        client = genai.Client(api_key=self.config.api_key)
        live_config = self._build_config()

        logger.info(f"Connecting Discord bot to Gemini Live ({self.config.model})...")
        self._session = await client.aio.live.connect(
            model=self.config.model,
            config=live_config,
        ).__aenter__()
        logger.info("Discord bot connected to Gemini Live")
        self._connected.set()
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self):
        """Disconnect from Gemini Live."""
        self._closing = True
        self._connected.clear()
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None

    async def send_message(self, text, images=None):
        """Send a message and wait for the complete text response.

        Args:
            text: The text message to send
            images: Optional list of (bytes, mime_type) tuples for image content

        Returns:
            The complete text response from the model
        """
        await self._connected.wait()
        if not self._session:
            raise RuntimeError("Not connected to Gemini Live")

        # Build content parts
        parts = []
        if images:
            for img_data, mime_type in images:
                parts.append(types.Part.from_bytes(data=img_data, mime_type=mime_type))
        parts.append(types.Part.from_text(text=text))

        # Send as client content (conversation turn)
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=parts),
            turn_complete=True,
        )

        # Wait for complete response
        response_text = await self._response_queue.get()
        return response_text

    async def inject_context(self, text):
        """Inject context into the session without expecting a response.
        Used for loading conversation history on startup."""
        await self._connected.wait()
        if not self._session:
            return
        await self._session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            ),
            turn_complete=False,
        )

    async def _receive_loop(self):
        """Continuously receive responses from Gemini Live."""
        transcript_buffer = ""
        while not self._closing:
            try:
                async for response in self._session.receive():
                    # Audio data from model_turn is discarded (we only need transcription)

                    # Capture output transcription (text version of audio response)
                    if (
                        response.server_content
                        and hasattr(response.server_content, "output_transcription")
                        and response.server_content.output_transcription
                    ):
                        transcription = response.server_content.output_transcription
                        if hasattr(transcription, "text") and transcription.text:
                            transcript_buffer += transcription.text

                    # Thinking/thought parts (for logging)
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if getattr(part, "thought", False) and part.text:
                                logger.debug(f"Discord bot thinking: {part.text[:100]}")

                    # Turn complete - deliver accumulated transcription
                    if response.server_content and response.server_content.turn_complete:
                        if transcript_buffer.strip():
                            await self._response_queue.put(transcript_buffer.strip())
                        transcript_buffer = ""

                    # Tool calls
                    if response.tool_call:
                        try:
                            responses = []
                            for fc in response.tool_call.function_calls:
                                logger.info(f"Discord tool call: {fc.name}")
                                fr = await self.tool_handler.handle(fc)
                                responses.append(fr)
                            await self._session.send_tool_response(function_responses=responses)
                            # If a personality switch happened, inject the prompt
                            if self.tool_handler._personality_prompt:
                                prompt = self.tool_handler._personality_prompt
                                self.tool_handler._personality_prompt = None
                                await self._session.send_client_content(
                                    turns=types.Content(
                                        role="user",
                                        parts=[types.Part.from_text(text=prompt)],
                                    ),
                                    turn_complete=False,
                                )
                        except Exception as e:
                            logger.error(f"Discord tool dispatch error: {e}")

                    # Session resumption
                    if (
                        hasattr(response, "session_resumption_update")
                        and response.session_resumption_update
                    ):
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            self._save_session_handle(update.new_handle)

                    # Go away
                    if response.go_away:
                        logger.warning(f"Discord session: server disconnecting in {response.go_away.time_left}")

            except asyncio.CancelledError:
                return
            except Exception as e:
                if self._closing:
                    return
                logger.error(f"Discord receive loop error: {e}")
                # Signal any waiting send_message calls
                await self._response_queue.put(f"[Error: {str(e)[:100]}]")
                raise

    async def run_forever(self):
        """Main loop - connects, handles errors, reconnects automatically."""
        while not self._closing:
            if self._is_handle_expired():
                self._clear_session_handle()
            try:
                await self.connect()
                # Wait for receive loop to end (error or disconnect)
                await self._receive_task
            except APIError as e:
                err_str = str(e)
                if "429" in err_str.lower() or "quota" in err_str.lower():
                    old_key = self.config.api_key
                    new_key = self.config.rotate_key()
                    if new_key != old_key:
                        logger.warning("Discord bot: rate limited, switched API key")
                        self._rate_limit_backoff = 0
                    else:
                        self._rate_limit_backoff = min(self._rate_limit_backoff + 1, 5)
                        wait = 5 * (2 ** self._rate_limit_backoff)
                        logger.warning(f"Discord bot: rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)
                    continue
                if self._session_handle:
                    self._handle_fail_count += 1
                    if self._handle_fail_count >= 2:
                        self._clear_session_handle()
                logger.error(f"Discord bot API error: {e}")
                await asyncio.sleep(2)
            except Exception as e:
                if self._closing:
                    return
                logger.error(f"Discord bot session error: {e}")
                await asyncio.sleep(3)
            finally:
                self._connected.clear()
                if self._session:
                    try:
                        await self._session.__aexit__(None, None, None)
                    except Exception:
                        pass
                    self._session = None
