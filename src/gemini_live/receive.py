"""Receive loop for the Gemini Live session, demuxes every event type Gemini emits."""

import asyncio
import json
import logging
import time

from google.genai import types
from websockets.exceptions import ConnectionClosed

from src.gemini_live.leak_filter import strip_tool_call_leaks
from src.model_caps import interaction_status

logger = logging.getLogger(__name__)

MALFORMED_NUDGE = "System update - your last tool call had malformed arguments. Please try again."


def _broadcast_console(log_type: str, content: str, extra: dict = None):
    try:
        from control_server import add_console_log
        add_console_log(log_type, content, extra)
    except ImportError:
        pass
    except Exception:
        pass


class ReceiveLoopMixin:
    def _mark_model_active(self):
        self._interaction_busy = True
        self._last_model_activity = time.time()

    async def _finish_spoken_turn(self):
        # model stopped talking. on newer models the interaction can keep going after this
        if self._thinking_shown:
            self.audio.stop_thinking_sound()
            if self._emotion_system:
                self._emotion_system.stop_thinking()
        self._thinking_shown = False
        self._speaking = False
        if self._emotion_system:
            self._emotion_system.stop_speaking()
        self._start_chatbox_finalize(self._transcript_buffer)
        if self.config.obs_enabled:
            _broadcast_console("turn_complete", "")
        # User message already streamed in-place via stream_user_message
        ai_text, leaked = strip_tool_call_leaks(self._transcript_buffer.strip())
        if leaked:
            logger.warning("stripped tool-call leak from transcript before logging")
        if ai_text:
            self._conv_logger.add_assistant_message(ai_text)
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
        # people can talk again even if it is still working in the background
        if not self._tool_call_pending:
            self._ungate_audio()

    def _end_model_turn(self):
        # model is fully done with this interaction
        if self._thinking_shown:
            self.audio.stop_thinking_sound()
            if self._emotion_system:
                self._emotion_system.stop_thinking()
            self._thinking_shown = False
        self._interaction_busy = False
        self._speaking = False
        if not self._tool_call_pending:
            self._ungate_audio()
        if not self._chatbox_task:
            self.osc.set_typing(False)

    def _start_chatbox_finalize(self, text):
        # paging sleeps between pages, never let that hold up the receive loop
        self._cancel_chatbox_finalize()
        self._chatbox_task = asyncio.create_task(self._finalize_chatbox(text))
        self._chatbox_task.add_done_callback(self._chatbox_task_done)

    def _chatbox_task_done(self, task):
        if self._chatbox_task is task:
            self._chatbox_task = None

    def _cancel_chatbox_finalize(self):
        task = self._chatbox_task
        self._chatbox_task = None
        if task and not task.done():
            task.cancel()

    def _track_interaction(self, server_content):
        # the server says when its done if it sends interaction_status, else turn_complete is the end
        status = interaction_status(server_content)
        if status:
            self._uses_interaction_status = True
            self._last_model_activity = time.time()
        if status == "IDLE":
            return True
        if status:
            self._interaction_busy = True
            return False
        return bool(server_content.turn_complete) and not self._uses_interaction_status

    async def _receive_loop(self, session):
        while True:
            try:
                async for response in session.receive():
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            # Thought summaries (when include_thoughts is enabled)
                            if getattr(part, "thought", False) and part.text:
                                self._mark_model_active()
                                _broadcast_console("thinking", part.text, {"streaming": True})
                                logger.debug(f"Thought: {part.text[:100]}")
                                # Show "Thinking..." in VRChat chatbox (only once per thinking phase)
                                if not self._thinking_shown and not self._speaking:
                                    self._thinking_shown = True
                                    self._idle_chatbox.stop()
                                    self.osc.set_typing(True)
                                    self.osc.send_chatbox("Thinking...")
                                    self.audio.start_thinking_sound("thinking")
                                    if self._emotion_system:
                                        self._emotion_system.start_thinking()
                            elif part.inline_data:
                                self._mark_model_active()
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
                            # new speech owns the chatbox, drop any paging left from the last turn
                            self._cancel_chatbox_finalize()
                            self._transcript_buffer += transcription.text
                            self._update_chatbox()
                            # Also stream to console in real-time
                            _broadcast_console("response", transcription.text, {"streaming": True})
                            # Feed text to external TTS provider for synthesis
                            if self._tts:
                                self._tts.feed_text(transcription.text)

                    if response.server_content:
                        sc = response.server_content
                        if sc.turn_complete:
                            await self._finish_spoken_turn()
                        if self._track_interaction(sc):
                            self._end_model_turn()
                        elif sc.turn_complete and self._interaction_busy:
                            logger.debug("turn_complete but the model is still working in the background")
                            self.osc.set_typing(True)

                    if response.server_content and response.server_content.interrupted:
                        # an interruption ends the turn regardless of interaction_status
                        self._end_model_turn()
                        # unlike a normal turn end, the model was cut off mid speech
                        if self._emotion_system:
                            self._emotion_system.stop_speaking()
                        ai_text, leaked = strip_tool_call_leaks(self._transcript_buffer.strip())
                        if leaked:
                            logger.warning("stripped tool-call leak from transcript before logging (interrupted turn)")
                        if ai_text:
                            self._conv_logger.add_assistant_message(ai_text)
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
                        await self._start_tool_calls(session, response.tool_call.function_calls or [])

                    if response.tool_call_cancellation:
                        self._cancel_tool_calls(response.tool_call_cancellation.ids or [])

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
                            _cleaned, _ = strip_tool_call_leaks(self._transcript_buffer)
                            if _cleaned:
                                self._conv_logger.add_assistant_message(_cleaned)
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

    async def _start_tool_calls(self, session, calls):
        """Run every call in its own task so the receive loop keeps reading audio meanwhile."""
        if not calls:
            return
        self._mark_model_active()
        # model has the users input by now, close out their message
        self._conv_logger.finalize_user_message()
        if self._pending_finalize_task:
            self._pending_finalize_task.cancel()
            self._pending_finalize_task = None
        self._input_transcript_buffer = ""

        blocking = [fc for fc in calls if self.config.tool_runs_blocking(fc.name)]
        background = [fc for fc in calls if not self.config.tool_runs_blocking(fc.name)]

        if blocking:
            # the model is paused waiting on these, mic audio now can 1007/1008 the socket
            self._blocking_tool_calls += 1
            self._tool_call_pending = True
            await self._gate_audio(session)
            if self.config.model_capabilities.async_tools == "none":
                # old models regenerate after the response, so drop what was queued before the call
                if self._tts:
                    self._tts.interrupt()
                self._playback_interrupted = True
                while not self._audio_in_queue.empty():
                    try:
                        self._audio_in_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            self._spawn_tool_task(self._run_blocking_calls(session, blocking), blocking)

        for fc in background:
            logger.info(f"Tool call in background: {fc.name}")
            self._spawn_tool_task(self._run_background_call(session, fc), [fc])

    def _spawn_tool_task(self, coro, calls):
        task = asyncio.create_task(coro, name=f"tool-{calls[0].name}")
        ids = [fc.id for fc in calls if fc.id]
        for call_id in ids:
            self._tool_tasks[call_id] = task

        def _done(t, ids=ids):
            for call_id in ids:
                if self._tool_tasks.get(call_id) is t:
                    del self._tool_tasks[call_id]
            if not t.cancelled() and t.exception():
                logger.error(f"tool task {t.get_name()} crashed: {t.exception()}")

        task.add_done_callback(_done)
        return task

    def _cancel_tool_calls(self, ids):
        # let the tool finish so its side effects stay whole, just never answer a call the server dropped
        for call_id in ids:
            self._cancelled_tool_ids.add(call_id)
            if call_id in self._tool_tasks:
                logger.info(f"Tool call {call_id} cancelled by the server, its result wont be sent")
                _broadcast_console("info", f"Tool call cancelled: {call_id}")

    async def _run_blocking_calls(self, session, calls):
        try:
            results = await asyncio.gather(*(self._execute_tool_call(fc) for fc in calls))
            responses = [fr for fr, _bad in results]
            await self._send_tool_responses(session, responses)
            if any(bad for _fr, bad in results):
                await self._send_malformed_nudge(session)
        finally:
            # after a reconnect the new socket has its own gate state
            if session is self._session:
                self._release_blocking_gate()

    def _release_blocking_gate(self):
        self._blocking_tool_calls = max(0, self._blocking_tool_calls - 1)
        if self._blocking_tool_calls:
            return
        self._tool_call_pending = False
        # reset before ungating or the first chunk of the reply gets swallowed
        self._playback_interrupted = False
        if not (self._speaking and self.config.vad_mode == "silero"):
            self._ungate_audio()

    async def _run_background_call(self, session, fc):
        fr, malformed = await self._execute_tool_call(fc)
        await self._send_tool_responses(session, [fr])
        if malformed:
            await self._send_malformed_nudge(session)

    async def _execute_tool_call(self, fc):
        """(FunctionResponse, malformed) for one call. Never raises, errors become the response."""
        logger.info(f"Tool call: {fc.name}")
        try:
            args_dict = dict(fc.args) if fc.args else {}
            args_str = json.dumps(args_dict)
        except (TypeError, ValueError) as e:
            logger.warning(f"Malformed tool call args for {fc.name}: {e}")
            _broadcast_console("error", f"Malformed tool call: {fc.name} ({e})")
            return self.tool_handler.apply_response_hints(types.FunctionResponse(
                id=fc.id, name=fc.name,
                response={"result": "error", "message": "malformed arguments, please retry"},
            )), True
        _broadcast_console("tool_call", f"{fc.name}({args_str})")
        self._usage_metadata["tool_calls"] += 1
        self._conv_logger.add_tool_call(fc.name, args_dict)
        try:
            fr = await self.tool_handler.handle(fc)
        except Exception as e:
            logger.error(f"Tool {fc.name} failed: {e}")
            fr = self.tool_handler.apply_response_hints(types.FunctionResponse(
                id=fc.id, name=fc.name, response={"result": "error", "message": str(e)},
            ))
        result_dict = fr.response if fr.response else {}
        try:
            result_str = json.dumps(result_dict)
        except (TypeError, ValueError):
            result_str = str(result_dict)
        _broadcast_console("tool_response", f"{fc.name} → {result_str}")
        self._conv_logger.add_tool_response(fc.name, result_dict)
        return fr, False

    async def _send_tool_responses(self, session, responses):
        live = [fr for fr in responses if fr.id not in self._cancelled_tool_ids]
        for fr in responses:
            self._cancelled_tool_ids.discard(fr.id)
        if not live:
            return
        if session is not self._session:
            # reconnected while it ran, those call ids died with the old socket
            logger.debug(f"dropping {len(live)} tool response(s) for a closed session")
            return
        try:
            await session.send_tool_response(function_responses=live)
            self._last_model_activity = time.time()
        except Exception as e:
            logger.warning(f"Could not send tool response ({', '.join(fr.name for fr in live)}): {e}")

    async def _send_malformed_nudge(self, session):
        if session is not self._session:
            return
        logger.info("Sending recovery nudge after malformed tool call")
        try:
            await session.send_realtime_input(text=MALFORMED_NUDGE)
        except Exception as e:
            logger.debug(f"malformed nudge failed: {e}")

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

    async def _finalize_chatbox(self, raw_text):
        """Finalize chatbox with pagination when AI finishes speaking."""
        try:
            await self._page_chatbox(raw_text)
        except asyncio.CancelledError:
            # newer speech took over the chatbox, it owns the typing state now
            return
        self.osc.set_typing(self._interaction_busy)

    async def _page_chatbox(self, raw_text):
        # Skip chatbox updates if music is playing
        if self.audio.is_music_playing():
            return
        music_gen = getattr(self.tool_handler, 'music_gen', None)
        if music_gen and music_gen.is_active:
            return
        text = self._strip_audio_tags_for_chatbox(raw_text)
        if not text:
            return
        pages = self.osc.send_chatbox_paginated(text)
        if len(pages) > 1:
            await self.osc.display_pages(pages, self.config.chatbox_page_delay)
