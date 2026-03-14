import asyncio
import base64
import io
import json
import logging
import re
import struct
import threading
import time
from collections import deque

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Sentence-ending punctuation for splitting transcription chunks
SENTENCE_ENDS = re.compile(r'[.!?]\s+')
# Clause boundaries for fallback splitting when buffer gets long
CLAUSE_BREAKS = re.compile(r'[,;:]\s+')

BUFFER_MIN_SPLIT = 20       # Minimum chars before attempting sentence split
BUFFER_CLAUSE_SPLIT = 100   # Chars before falling back to clause-boundary splits
BUFFER_HARD_FLUSH = 200     # Chars before force-flushing regardless of punctuation
BUFFER_MAX_AGE = 3.0        # Seconds before force-flushing stale buffer


class QwenTTSProvider:
    """Streams output transcription text to a Qwen3 TTS server and produces PCM audio.

    Architecture:
    - Gemini Live stays in AUDIO mode (required by Live API).
    - When active, Gemini audio is discarded by the caller.
    - Output transcription text is fed in via `feed_text()`.
    - Text is buffered and split on sentence boundaries.
    - Each sentence is sent to the Qwen3 TTS server (streaming endpoint).
    - Returned audio chunks are resampled to the target sample rate (24 kHz)
      and queued for playback through the existing PyAudio output.
    """

    def __init__(self, config):
        self._base_url = config.get("tts", "qwen3", "base_url", default="http://localhost:7860").rstrip("/")
        self._mode = config.get("tts", "qwen3", "mode", default="voice_clone")
        self._ref_preset = config.get("tts", "qwen3", "ref_preset", default="")
        self._ref_audio = config.get("tts", "qwen3", "ref_audio", default="")
        self._ref_text = config.get("tts", "qwen3", "ref_text", default="")
        self._speaker = config.get("tts", "qwen3", "speaker", default="")
        self._instruct = config.get("tts", "qwen3", "instruct", default="")
        self._language = config.get("tts", "qwen3", "language", default="English")
        self._xvec_only = config.get("tts", "qwen3", "xvec_only", default=True)
        self._chunk_size = config.get("tts", "qwen3", "chunk_size", default=8)
        self._temperature = config.get("tts", "qwen3", "temperature", default=0.9)
        self._top_k = config.get("tts", "qwen3", "top_k", default=50)
        self._repetition_penalty = config.get("tts", "qwen3", "repetition_penalty", default=1.05)
        self._target_sr = config.get("audio", "receive_sample_rate", default=24000)

        self._text_buffer = ""
        self._buffer_first_add = 0  # Timestamp when text first entered the buffer
        self._sentence_queue = deque()
        self._audio_queue = asyncio.Queue()
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread = None
        self._interrupted = False
        self._current_request = None  # Track active HTTP request for cancellation

    # ── Public API ───────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._interrupted = False
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info(f"Qwen3 TTS started (mode={self._mode}, url={self._base_url})")

    def stop(self):
        self._running = False
        self._interrupted = True
        if self._worker_thread:
            self._worker_thread.join(timeout=3)
            self._worker_thread = None

    def feed_text(self, text: str):
        """Feed a chunk of output transcription text. Called from receive_loop."""
        if not text:
            return
        self._interrupted = False  # New speech started, clear interrupt flag
        with self._lock:
            if not self._text_buffer:
                self._buffer_first_add = time.monotonic()
            self._text_buffer += text
            self._flush_sentences()

    def turn_complete(self):
        """Called when Gemini signals turn_complete. Flushes any remaining text."""
        with self._lock:
            remaining = self._text_buffer.strip()
            if remaining:
                self._sentence_queue.append(remaining)
                self._text_buffer = ""
                self._buffer_first_add = 0

    def interrupt(self):
        """Called on interruption. Clears buffer and queue, cancels in-flight request.
        
        Sets _interrupted=True which stays set until feed_text() is called again
        with new speech. This prevents the worker from processing stale data."""
        self._interrupted = True
        with self._lock:
            self._text_buffer = ""
            self._sentence_queue.clear()
        # Drain audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Cancel in-flight HTTP request
        if self._current_request:
            try:
                self._current_request.close()
            except Exception:
                pass
            self._current_request = None

    async def get_audio(self) -> bytes | None:
        """Get next audio chunk (PCM 16-bit mono at target_sr). Async-safe."""
        try:
            return await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    # ── Internal ─────────────────────────────────────────────────────────

    def _flush_sentences(self):
        """Split buffered text on sentence boundaries and queue complete sentences.
        
        Strategy:
        1. Primary: split on . ? ! (sentence ends) when buffer >= 20 chars
        2. Fallback: split on , ; : (clause breaks) when buffer >= 100 chars
        3. Hard flush: force-send when buffer >= 200 chars
        """
        buf_len = len(self._text_buffer)
        if buf_len < BUFFER_MIN_SPLIT:
            return

        # Try sentence-end splitting first
        if self._try_split(SENTENCE_ENDS):
            return

        # Fallback to clause-boundary splitting for long buffers
        if buf_len >= BUFFER_CLAUSE_SPLIT and self._try_split(CLAUSE_BREAKS):
            return

        # Hard flush for very long buffers with no punctuation
        if buf_len >= BUFFER_HARD_FLUSH:
            self._sentence_queue.append(self._text_buffer.strip())
            self._text_buffer = ""
            self._buffer_first_add = 0

    def _check_stale_buffer(self):
        """Called from worker thread: flush buffer if it's been sitting too long."""
        with self._lock:
            if (self._text_buffer
                    and self._buffer_first_add > 0
                    and time.monotonic() - self._buffer_first_add >= BUFFER_MAX_AGE):
                text = self._text_buffer.strip()
                if text:
                    self._sentence_queue.append(text)
                self._text_buffer = ""
                self._buffer_first_add = 0

    def _try_split(self, pattern) -> bool:
        """Try to split buffer on the given regex pattern. Returns True if split occurred."""
        splits = list(pattern.finditer(self._text_buffer))
        if not splits:
            return False
        for match in splits:
            end_pos = match.end()
            sentence = self._text_buffer[:end_pos].strip()
            if sentence:
                self._sentence_queue.append(sentence)
            self._text_buffer = self._text_buffer[end_pos:]
        self._text_buffer = self._text_buffer.lstrip()
        if self._text_buffer:
            self._buffer_first_add = time.monotonic()
        else:
            self._buffer_first_add = 0
        return True

    def _worker_loop(self):
        """Background thread: pulls sentences from queue, sends to TTS, queues audio."""
        while self._running:
            # Check for stale buffer that needs flushing
            self._check_stale_buffer()
            sentence = None
            with self._lock:
                if self._sentence_queue:
                    sentence = self._sentence_queue.popleft()
            if not sentence:
                time.sleep(0.02)
                continue
            if self._interrupted:
                continue
            try:
                self._synthesize_and_queue(sentence)
            except Exception as e:
                logger.error(f"Qwen3 TTS synthesis error: {e}")

    def _synthesize_and_queue(self, text: str):
        """Send text to Qwen3 TTS streaming endpoint and queue resampled PCM chunks."""
        form = {
            "text": text,
            "language": self._language,
            "mode": self._mode,
            "chunk_size": str(self._chunk_size),
            "temperature": str(self._temperature),
            "top_k": str(self._top_k),
            "repetition_penalty": str(self._repetition_penalty),
        }

        if self._mode == "voice_clone":
            form["xvec_only"] = str(self._xvec_only)
            if self._ref_preset:
                form["ref_preset"] = self._ref_preset
            if self._ref_text:
                form["ref_text"] = self._ref_text

        elif self._mode == "custom":
            if self._speaker:
                form["speaker"] = self._speaker
            if self._instruct:
                form["instruct"] = self._instruct

        elif self._mode == "voice_design":
            if self._instruct:
                form["instruct"] = self._instruct

        files = {}
        if self._mode == "voice_clone" and self._ref_audio:
            try:
                files["ref_audio"] = open(self._ref_audio, "rb")
            except FileNotFoundError:
                logger.warning(f"Ref audio not found: {self._ref_audio}")

        try:
            resp = requests.post(
                f"{self._base_url}/generate/stream",
                data=form,
                files=files if files else None,
                stream=True,
                timeout=30,
            )
            self._current_request = resp
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if self._interrupted or not self._running:
                    break
                if not line or not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])

                if payload["type"] == "error":
                    logger.error(f"Qwen3 TTS error: {payload['message']}")
                    break
                if payload["type"] == "done":
                    break
                if payload["type"] == "queued":
                    continue
                if payload["type"] == "chunk":
                    audio_b64 = payload["audio_b64"]
                    audio_bytes = base64.b64decode(audio_b64)
                    pcm = self._decode_and_resample(audio_bytes)
                    if pcm and not self._interrupted:
                        # Put into asyncio queue from sync thread
                        try:
                            self._audio_queue.put_nowait(pcm)
                        except asyncio.QueueFull:
                            pass
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Qwen3 TTS server at {self._base_url}")
        except requests.exceptions.Timeout:
            logger.error("Qwen3 TTS request timed out")
        except Exception as e:
            if not self._interrupted:
                logger.error(f"Qwen3 TTS stream error: {e}")
        finally:
            self._current_request = None
            for f in files.values():
                f.close()

    def _decode_and_resample(self, audio_bytes: bytes) -> bytes | None:
        """Decode WAV/PCM from TTS server and resample to target sample rate."""
        try:
            import soundfile as sf
            audio_np, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

            # Convert stereo to mono if needed
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            # Resample if sample rates differ
            if src_sr != self._target_sr:
                audio_np = self._resample(audio_np, src_sr, self._target_sr)

            # Convert to int16 PCM
            pcm = (audio_np * 32767).clip(-32767, 32767).astype(np.int16)
            return pcm.tobytes()
        except Exception as e:
            logger.error(f"Audio decode/resample error: {e}")
            return None

    def _resample(self, audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Simple linear interpolation resample."""
        if src_sr == dst_sr:
            return audio
        ratio = dst_sr / src_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
