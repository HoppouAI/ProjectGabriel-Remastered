"""Audio + video streaming loops for the Gemini Live session.

All the realtime I/O ferrying lives here:
- mic input loop reading PyAudio chunks into the send queue
- send-realtime loop with optional client side Silero VAD
- speaker output loop applying boost/distortion and chunked writes
- TTS audio loop pulling from external providers
- screen capture loop for the vision modality
- the manual VAD activity start/end + audioStreamEnd flush helpers

Pulled out as a mixin so session.py stops being a wall of loops.
"""

import asyncio
import logging
import time

import numpy as np
from google.genai import types
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class AudioLoopsMixin:
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
        logger.debug("Loading Silero VAD model...")
        torch.set_num_threads(1)  # single thread is fine for VAD
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        model.eval()
        self._silero_vad = model
        logger.debug("Silero VAD model loaded")
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
