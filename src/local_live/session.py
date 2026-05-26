"""Local LLM session - drop-in replacement for GeminiLiveSession.

Public surface (matches what main.py / control_server / plugins reach for):
    s = LocalLiveSession(config, audio, osc, tracker, personality, tts_provider)
    s.tool_handler            - ToolHandler instance
    s._speaking / _is_idle    - state flags read by tracker / face_tracker
    s._wanderer               - set externally by main.py
    s._save_audio             - debug wav export toggle
    s.set_mic_muted(bool)
    s.request_reconnect()
    await s.send_text(text)
    await s.send_client_content_safe(turns, turn_complete=True)
    s.save_audio_to_wav()
    await s.run()

Flow:
    mic -> AudioManager input stream -> Silero VAD (inside MoonshineSTT)
        -> Moonshine transcript -> LLM turn -> LM Studio stream
        -> text deltas: TTS provider feed_text + chatbox stream
        -> tool calls: ToolHandler.handle_by_name -> append result -> reloop
        -> finish: turn_complete, finalize chatbox
    TTS audio queue -> AudioManager output stream
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.idle_chatbox import IdleChatbox
from src.emotions import init_emotion_system, get_emotion_system
from src.tools import ToolHandler

from .llm import LMStudioClient
from .stt import MoonshineSTT
from .tools_adapter import collect_openai_tools

logger = logging.getLogger(__name__)

try:
    from src.gemini_live.conversation_logger import ConversationLogger as _ConvLogger
except Exception:
    _ConvLogger = None

try:
    from src.gemini_live.chatbox_orchestrator import ChatboxOrchestrator
except Exception:
    ChatboxOrchestrator = None  # type: ignore


def _broadcast_console(log_type: str, content: str, extra: dict | None = None):
    try:
        from control_server import add_console_log
        add_console_log(log_type, content, extra)
    except ImportError:
        pass
    except Exception:
        pass


CHATBOX_LIMIT = 144


class LocalLiveSession:
    def __init__(self, config, audio_mgr, osc, tracker, personality_mgr, tts_provider=None):
        self.config = config
        self.audio = audio_mgr
        self.osc = osc
        self.personality = personality_mgr
        self.tool_handler = ToolHandler(audio_mgr, osc, tracker, personality_mgr, config)
        self._tts = tts_provider
        if self._tts is None:
            raise RuntimeError(
                "local backend requires an external TTS provider. enable one of "
                "tts.qwen3 / tts.hoppou / tts.chirp3_hd / tts.tiktok or set "
                "tts.external_provider to a plugin-supplied name."
            )

        # state flags read by other subsystems
        self._speaking = False
        self._is_idle = False
        self._mic_muted = False
        self._wanderer = None
        self._save_audio = False
        self._reconnect_requested = False

        # streaming buffers
        self._transcript_buffer = ""        # current AI utterance for chatbox
        self._playback_interrupted = False
        self._audio_in_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._last_audio_time = 0.0
        self._last_interaction_time = time.time()
        self._idle_engagement_sent = False

        # rolling chat history (excluding system instruction, which is rebuilt)
        self._history: list[dict] = []
        self._pending_inputs: asyncio.Queue[dict] = asyncio.Queue()
        # waiting-for-LLM lock so concurrent send_text calls serialize
        self._turn_lock = asyncio.Lock()

        # audio recording for --save-audio
        self._audio_recording_writer = None
        self._audio_recording_path: Optional[Path] = None
        self._audio_recording_seconds = 0.0

        # conversation logger (reuse the gemini one)
        self._conv_logger = (
            _ConvLogger(enabled=config.conversation_logging_enabled)
            if _ConvLogger else None
        )

        # emotion system
        self._emotion_system = None
        if config.emotion_enabled:
            self._emotion_system = init_emotion_system(config, osc)
            self._emotion_system.start()
            logger.info("emotion system initialized")

        # idle chatbox banner
        self._idle_chatbox = IdleChatbox(osc, config)

        # chatbox orchestrator (plugin sources, music, etc)
        self._chatbox = None
        if ChatboxOrchestrator is not None:
            self._chatbox = ChatboxOrchestrator(send_chatbox=osc.send_chatbox)
            self._chatbox.register_builtin("local_music", self._builtin_local_music)
            self._chatbox.register_builtin("music_gen", self._builtin_music_gen)

        # stt + llm clients
        self._stt = MoonshineSTT(config)
        self._llm = LMStudioClient(config)

        # cached prompt + tools so LM Studio's prefix cache actually hits
        # across turns. rebuilt on personality switch or reconnect, same idea
        # as a gemini live session_handle lifetime.
        self._system_text: Optional[str] = None
        self._system_personality_id: Optional[str] = None
        self._cached_tools: Optional[list] = None
        # last partial we already broadcast, so we don't spam the WebUI
        self._last_partial_broadcast: str = ""

    # ── chatbox builtins (parity with gemini session) ────────────────────

    def _builtin_local_music(self):
        progress = self.audio.get_music_progress()
        if not progress:
            return None
        try:
            from src.gemini_live.chatbox import ChatboxFormattersMixin
            text = ChatboxFormattersMixin._format_now_playing(self, progress)
        except Exception:
            text = f"Now playing: {progress.get('title', '?')}"
        return ("local_music", text)

    def _builtin_music_gen(self):
        music_gen = getattr(self.tool_handler, "music_gen", None)
        if music_gen and music_gen.is_active:
            try:
                from src.gemini_live.chatbox import ChatboxFormattersMixin
                text = ChatboxFormattersMixin._format_music_gen_display(self, music_gen)
            except Exception:
                text = "Generating music..."
            return ("music_gen", text)
        return None

    # ── public api ───────────────────────────────────────────────────────

    def set_mic_muted(self, muted: bool):
        self._mic_muted = muted
        logger.info(f"mic mute set to {muted}")

    def request_reconnect(self):
        """In local mode there's no websocket; this just clears the in-flight
        turn so the next iteration starts fresh, and forces the system prompt
        + tools list to be rebuilt (mirrors a gemini session_handle reset)."""
        self._reconnect_requested = True
        self._system_text = None
        self._cached_tools = None
        logger.info("local: reconnect requested - resetting turn state")

    def _current_personality_id(self) -> Optional[str]:
        try:
            cur = self.personality.get_current() if self.personality else None
            if isinstance(cur, dict):
                return cur.get("id") or cur.get("name")
        except Exception:
            return None
        return None

    def _get_system_text(self) -> str:
        """Return the cached system instruction, rebuilding only on personality
        switch or explicit reconnect. Keeping this stable across turns is what
        lets LM Studio's prompt prefix cache hit instead of reprocessing the
        whole context every time the {date} placeholder changes by a second.
        """
        pid = self._current_personality_id()
        if self._system_text is None or pid != self._system_personality_id:
            self._system_text = self.config.build_system_instruction(self.personality)
            self._system_personality_id = pid
        return self._system_text

    def _get_tools(self):
        if self._cached_tools is None:
            self._cached_tools = collect_openai_tools(self.config)
        return self._cached_tools

    async def send_text(self, text: str):
        """Inject a user text message into the next LLM turn."""
        if not text:
            return
        await self._pending_inputs.put({"role": "user", "content": text, "_source": "text"})
        try:
            from src.plugins import emit_event
            emit_event("message_in", text, "text")
        except Exception:
            pass

    async def send_client_content_safe(self, turns, turn_complete=True):
        """Plugin-compatible: extract text from gemini-style turns and inject
        as a system note for the next turn. Not interruptive in local mode."""
        text = self._extract_text_from_turns(turns)
        if not text:
            return
        await self._pending_inputs.put({"role": "system", "content": text, "_source": "system"})

    @staticmethod
    def _extract_text_from_turns(turns):
        if hasattr(turns, "parts"):
            for part in turns.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
        elif isinstance(turns, dict):
            for part in turns.get("parts", []):
                if isinstance(part, dict) and "text" in part:
                    return part["text"]
        elif isinstance(turns, list):
            # list of Content
            for t in turns:
                txt = LocalLiveSession._extract_text_from_turns(t)
                if txt:
                    return txt
        return None

    def save_audio_to_wav(self):
        if self._audio_recording_writer is not None:
            try:
                self._audio_recording_writer.close()
            except Exception:
                pass
            logger.info(
                f"saved {self._audio_recording_seconds:.1f}s of audio to {self._audio_recording_path}"
            )
            self._audio_recording_writer = None
            self._audio_recording_path = None
            self._audio_recording_seconds = 0.0

    def _record_output_audio(self, audio_data: bytes):
        if not audio_data:
            return
        if self._audio_recording_writer is None:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._audio_recording_path = data_dir / f"local_output_{timestamp}.wav"
            self._audio_recording_writer = wave.open(str(self._audio_recording_path), "wb")
            self._audio_recording_writer.setnchannels(1)
            self._audio_recording_writer.setsampwidth(2)
            self._audio_recording_writer.setframerate(self.config.receive_sample_rate)
            logger.info(f"recording local output audio to {self._audio_recording_path}")
        self._audio_recording_writer.writeframes(audio_data)
        self._audio_recording_seconds += len(audio_data) / (self.config.receive_sample_rate * 2)

    # ── run loop ─────────────────────────────────────────────────────────

    async def run(self):
        """Open mic/speaker streams, launch loops, supervise."""
        # boot stt + llm
        self._stt.start()
        if not self._stt.ready:
            err = self._stt.load_error or "unknown error"
            logger.error(f"local: STT failed to start: {err}")
            _broadcast_console("error", f"STT init failed: {err}")
            raise RuntimeError(f"STT init failed: {err}")
        await self._llm.start()

        logger.info("local backend connected (LM Studio + Moonshine)")
        _broadcast_console("info",
            f"Local backend ready (model={self.config.local_llm_model}, "
            f"stt={self.config.local_stt_model})")

        self.tool_handler.session = None
        self.tool_handler.live_session = self

        input_stream = self.audio.open_input_stream()
        output_stream = self.audio.open_output_stream()

        tasks: list[asyncio.Task] = []
        try:
            task_specs = [
                ("mic", self._mic_loop(input_stream)),
                ("turn-pump", self._turn_pump_loop()),
                ("stt-partials", self._stt_partial_loop()),
                ("tts-audio", self._tts_audio_loop()),
                ("audio-playback", self._play_audio_loop(output_stream)),
                ("idle-check", self._idle_check_loop()),
                ("now-playing", self._now_playing_loop()),
                ("reconnect-monitor", self._reconnect_monitor_loop()),
            ]
            tasks = [
                asyncio.create_task(coro, name=f"local-{name}")
                for name, coro in task_specs
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.cancelled():
                    continue
                exc = t.exception()
                if exc:
                    raise exc
        finally:
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                input_stream.close()
            except Exception:
                pass
            try:
                output_stream.close()
            except Exception:
                pass
            self._stt.stop()
            await self._llm.stop()
            try:
                self._idle_chatbox.stop()
            except Exception:
                pass

    # ── loops ────────────────────────────────────────────────────────────

    async def _mic_loop(self, input_stream):
        """Read mic chunks, feed STT (which runs Silero VAD internally)."""
        chunk = self.config.chunk_size
        while True:
            try:
                data = await asyncio.to_thread(
                    input_stream.read, chunk, exception_on_overflow=False,
                )
                if self._mic_muted:
                    continue
                self._stt.feed_audio(data)
                # mirror speaking flag to the gemini-style _manual_vad_speaking
                # so anything reading session._is_idle behaves the same.
                if self._stt.speaking:
                    self._last_interaction_time = time.time()
                    self._idle_engagement_sent = False
                    if self._emotion_system:
                        self._emotion_system.mark_activity()
            except asyncio.CancelledError:
                return
            except OSError:
                return
            except Exception as e:
                logger.error(f"mic loop error: {e}")
                await asyncio.sleep(0.1)

    async def _stt_partial_loop(self):
        """Watch Moonshine's in-flight partial and push it to the WebUI as a
        streaming transcription event, so the user sees text appear while
        they're still talking instead of only after VAD ends."""
        while True:
            try:
                await asyncio.sleep(0.15)
                partial = (self._stt.partial_text or "").strip()
                if partial and partial != self._last_partial_broadcast:
                    self._last_partial_broadcast = partial
                    _broadcast_console(
                        "transcription", partial, {"streaming": True, "partial": True}
                    )
                elif not self._stt.speaking and self._last_partial_broadcast:
                    self._last_partial_broadcast = ""
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug(f"stt partial loop: {e}")
                await asyncio.sleep(0.5)

    async def _turn_pump_loop(self):
        """Pull either a new STT transcript or a queued send_text, then run
        one LLM turn end-to-end (including tool call iterations)."""
        while True:
            try:
                # 1) drain queued text/system inputs first (no blocking)
                msg = None
                if not self._pending_inputs.empty():
                    msg = await self._pending_inputs.get()
                else:
                    # 2) wait briefly for STT
                    transcript = await self._stt.next_transcript(timeout=0.3)
                    if transcript:
                        msg = {"role": "user", "content": transcript, "_source": "voice"}
                if msg is None:
                    continue
                async with self._turn_lock:
                    await self._run_turn(msg)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception(f"turn pump error: {e}")
                _broadcast_console("error", f"turn error: {str(e)[:120]}")
                await asyncio.sleep(0.5)

    async def _run_turn(self, msg: dict):
        """Run one full user turn: build messages, stream LLM, handle tools,
        push deltas to TTS + chatbox."""
        # log + broadcast user message
        content = msg["content"]
        source = msg.pop("_source", "voice")
        if msg["role"] == "user":
            _broadcast_console("transcription", content, {"streaming": False})
            if self._conv_logger:
                self._conv_logger.stream_user_message(content)
                self._conv_logger.finalize_user_message()
        else:
            _broadcast_console("info", f"[system note] {content}")

        # append to rolling history
        self._history.append({"role": msg["role"], "content": content})
        self._trim_history()

        # build messages with system instruction + optional image. system
        # prompt is cached for the life of the session (see _get_system_text)
        # so LM Studio can reuse the kv-cache prefix.
        system_text = self._get_system_text()
        messages: list[dict] = [{"role": "system", "content": system_text}]
        messages.extend(self._history[:-1])

        last_user_msg = self._history[-1]
        if (
            msg["role"] == "user"
            and self.config.local_vision_enabled
        ):
            image_part = await asyncio.to_thread(self._capture_screen_part)
            if image_part is not None:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        image_part,
                    ],
                })
            else:
                messages.append(last_user_msg)
        else:
            messages.append(last_user_msg)

        tools = self._get_tools()
        max_iter = self.config.local_llm_max_tool_iterations

        # prep chatbox / speaking state
        self._transcript_buffer = ""
        self._speaking = False
        self._idle_chatbox.stop()
        self.osc.set_typing(True)

        full_assistant_text = ""

        for iteration in range(max_iter):
            tool_calls_this_iter: list[dict] = []
            saw_finish_reason = None
            try:
                async for event in self._llm.stream_turn(messages, tools=tools):
                    if self._reconnect_requested:
                        self._reconnect_requested = False
                        break
                    etype = event.get("type")
                    if etype == "text":
                        delta = event["delta"]
                        if not self._speaking and delta.strip():
                            self._speaking = True
                            if self._emotion_system:
                                self._emotion_system.start_speaking()
                        self._transcript_buffer += delta
                        full_assistant_text += delta
                        self._last_audio_time = time.time()
                        _broadcast_console("response", delta, {"streaming": True})
                        if self._tts:
                            try:
                                self._tts.feed_text(delta)
                            except Exception as e:
                                logger.debug(f"tts feed failed: {e}")
                        self._update_chatbox(self._transcript_buffer)
                    elif etype == "tool_call":
                        tool_calls_this_iter = event["calls"]
                    elif etype == "finish":
                        saw_finish_reason = event.get("reason")
                    elif etype == "error":
                        logger.error(f"LLM error: {event['message']}")
                        _broadcast_console("error", event["message"][:200])
                        await self._inject_chatbox_error()
                        return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"LLM stream crashed: {e}")
                _broadcast_console("error", f"LLM stream crashed: {e}")
                return

            # if there were tool calls, dispatch them, append messages, loop
            if tool_calls_this_iter:
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": full_assistant_text or None,
                    "tool_calls": [
                        {
                            "id": tc.get("id") or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc.get("arguments_json") or "{}",
                            },
                        }
                        for i, tc in enumerate(tool_calls_this_iter)
                    ],
                }
                messages.append(assistant_msg)
                self._history.append(assistant_msg)

                for tc in tool_calls_this_iter:
                    name = tc.get("name") or ""
                    args_raw = tc.get("arguments_json") or "{}"
                    try:
                        args = json.loads(args_raw) if args_raw.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                    logger.info(f"tool call: {name}({args})")
                    _broadcast_console("info", f"tool call: {name}")
                    result = await self.tool_handler.handle_by_name(name, args)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or f"call_{name}",
                        "name": name,
                        "content": json.dumps(result),
                    }
                    messages.append(tool_msg)
                    self._history.append(tool_msg)

                # reset visible turn buffers but keep aggregated text for history
                self._transcript_buffer = ""
                full_assistant_text = ""
                continue

            # no tool calls; this turn is done
            break

        # finalize speech + chatbox
        self._speaking = False
        if self._emotion_system:
            self._emotion_system.stop_speaking()
        if self._tts:
            try:
                if hasattr(self._tts, "turn_complete"):
                    self._tts.turn_complete()
            except Exception:
                pass

        ai_text = (full_assistant_text or self._transcript_buffer).strip()
        if ai_text:
            self._history.append({"role": "assistant", "content": ai_text})
            if self._conv_logger:
                self._conv_logger.add_assistant_message(ai_text)
            try:
                from src.plugins import emit_event
                emit_event("message_out", ai_text)
            except Exception:
                pass
        await self._finalize_chatbox(ai_text)
        self._trim_history()

    def _trim_history(self):
        keep = max(2, self.config.local_llm_history_messages)
        # never drop a tool_calls/tool pair, so trim conservatively
        if len(self._history) > keep:
            self._history = self._history[-keep:]

    def _capture_screen_part(self):
        try:
            import mss
            from PIL import Image
        except ImportError:
            logger.warning("vision enabled but mss/PIL missing")
            return None
        try:
            with mss.mss() as sct:
                idx = self.config.vision_monitor
                if idx >= len(sct.monitors):
                    idx = 0
                screenshot = sct.grab(sct.monitors[idx])
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                max_size = self.config.local_vision_max_size
                if img.width > max_size or img.height > max_size:
                    img.thumbnail([max_size, max_size])
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=self.config.local_vision_quality)
                jpeg = buf.getvalue()
                return self._llm.encode_image(jpeg)
        except Exception as e:
            logger.warning(f"screen capture failed: {e}")
            return None

    # ── chatbox helpers (slimmed from gemini receive.py) ─────────────────

    def _update_chatbox(self, text: str):
        if self.audio.is_music_playing():
            return
        music_gen = getattr(self.tool_handler, "music_gen", None)
        if music_gen and music_gen.is_active:
            return
        if not text:
            return
        if len(text) <= CHATBOX_LIMIT:
            self.osc.send_chatbox(text)
        else:
            self.osc.send_chatbox("..." + text[-(CHATBOX_LIMIT - 3):])

    async def _finalize_chatbox(self, text: str):
        if self.audio.is_music_playing():
            self.osc.set_typing(False)
            return
        music_gen = getattr(self.tool_handler, "music_gen", None)
        if music_gen and music_gen.is_active:
            self.osc.set_typing(False)
            return
        if not text:
            self.osc.set_typing(False)
            return
        pages = self.osc.send_chatbox_paginated(text)
        if pages and len(pages) > 1:
            await self.osc.display_pages(pages, self.config.chatbox_page_delay)
        self.osc.set_typing(False)

    async def _inject_chatbox_error(self):
        try:
            self.osc.send_chatbox("local LLM hiccup, trying again...")
        except Exception:
            pass
        self.osc.set_typing(False)

    # ── tts audio + playback ─────────────────────────────────────────────

    async def _tts_audio_loop(self):
        while True:
            try:
                pcm = await self._tts.get_audio()
                if pcm and not getattr(self._tts, "_interrupted", False):
                    await self._audio_in_queue.put(pcm)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug(f"tts audio loop: {e}")
                await asyncio.sleep(0.05)

    async def _play_audio_loop(self, output_stream):
        CHUNK = 4096
        while True:
            try:
                audio_data = await self._audio_in_queue.get()
                audio_data = self.audio.process_output_audio(audio_data)
                if not audio_data:
                    continue
                if self._save_audio:
                    self._record_output_audio(audio_data)
                for i in range(0, len(audio_data), CHUNK):
                    if self._playback_interrupted:
                        break
                    await asyncio.to_thread(
                        output_stream.write, audio_data[i:i + CHUNK]
                    )
                if self._playback_interrupted:
                    self._playback_interrupted = False
            except asyncio.CancelledError:
                return
            except OSError:
                return
            except Exception as e:
                logger.error(f"play audio loop error: {e}")
                await asyncio.sleep(0.1)

    # ── idle / housekeeping ──────────────────────────────────────────────

    async def _idle_check_loop(self):
        IDLE_TIMEOUT = 15.0
        IDLE_ENGAGEMENT_SECONDS = 3600
        while True:
            await asyncio.sleep(1)
            if self._speaking and self._last_audio_time > 0:
                idle_t = time.time() - self._last_audio_time
                if idle_t >= IDLE_TIMEOUT:
                    self._speaking = False
                    self._last_audio_time = 0
                    if self._emotion_system:
                        self._emotion_system.stop_speaking()
                    self.osc.set_typing(False)

            music_playing = self.audio.get_music_progress() is not None
            tracker_active = getattr(self.tool_handler.tracker, "active", False) if self.tool_handler.tracker else False
            music_gen = getattr(self.tool_handler, "music_gen", None)
            music_gen_active = music_gen.is_active if music_gen else False
            plugin_busy = False
            if self._chatbox is not None:
                try:
                    plugin_busy = self._chatbox.has_active_source()
                except Exception:
                    plugin_busy = False

            busy = music_playing or tracker_active or music_gen_active or plugin_busy
            if self._emotion_system:
                self._emotion_system.set_seated(self.osc.seated)
                if busy:
                    self._emotion_system.mark_activity()
                else:
                    self._emotion_system.check_idle()

            idle_now = (
                not self._speaking
                and not self._stt.speaking
                and not busy
            )
            self._is_idle = idle_now
            if idle_now:
                emo = self._emotion_system
                if (emo and emo._idle_active) or not emo:
                    self._idle_chatbox.start()
            elif busy:
                self._idle_chatbox.stop()

            if (
                not self._idle_engagement_sent
                and not self._speaking
                and not busy
                and time.time() - self._last_interaction_time >= IDLE_ENGAGEMENT_SECONDS
            ):
                self._idle_engagement_sent = True
                logger.info("idle engagement prompt")
                await self.send_text(
                    "System update - You have been idle for a while. "
                    "Try to engage nearby people in conversation. "
                    "Say something interesting, ask a question, or make an observation to get someone to talk to you."
                )

    async def _now_playing_loop(self):
        """Chatbox orchestrator tick - keeps now playing / plugin sources fresh."""
        if self._chatbox is None:
            while True:
                await asyncio.sleep(60)
        while True:
            try:
                if self._speaking or self._stt.speaking:
                    await asyncio.sleep(0.5)
                    continue
                self._chatbox.tick()
            except Exception as e:
                logger.debug(f"chatbox tick: {e}")
            await asyncio.sleep(1.5)

    async def _reconnect_monitor_loop(self):
        while True:
            if self._reconnect_requested:
                # local mode just resets history+turn buffers, no reconnect needed
                logger.info("local: reset state on reconnect request")
                self._history.clear()
                self._transcript_buffer = ""
                self._reconnect_requested = False
            await asyncio.sleep(0.5)
