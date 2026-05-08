"""Receive loop for the Gemini Live session.

This is by far the chunkiest piece of session logic, it has to demux
every event Gemini emits in a single async for: thoughts, audio inline_data,
input/output transcription, turn_complete, interruptions, tool_call,
usage_metadata, go_away, session_resumption_update.

Plus the two chatbox update helpers that are only called from this loop.

Pulled out as a mixin to keep session.py readable.
"""

import asyncio
import json
import logging
import time

from google.genai import types
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


def _broadcast_console(log_type: str, content: str, extra: dict = None):
    try:
        from control_server import add_console_log
        add_console_log(log_type, content, extra)
    except ImportError:
        pass
    except Exception:
        pass


class ReceiveLoopMixin:
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
                        ai_text = self._transcript_buffer.strip()
                        if ai_text:
                            self._conv_logger.add_assistant_message(self._transcript_buffer)
                        self._transcript_buffer = ""
                        if ai_text:
                            try:
                                from src.plugins import emit_event
                                emit_event("message_out", ai_text)
                            except Exception as e:
                                logger.debug(f"plugin message_out dispatch failed: {e}")
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
                        ai_text = self._transcript_buffer.strip()
                        if ai_text:
                            self._conv_logger.add_assistant_message(self._transcript_buffer)
                        self._transcript_buffer = ""
                        if ai_text:
                            try:
                                from src.plugins import emit_event
                                emit_event("message_out", ai_text)
                            except Exception as e:
                                logger.debug(f"plugin message_out dispatch failed: {e}")
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
