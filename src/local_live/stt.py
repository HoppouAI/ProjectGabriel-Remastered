"""Moonshine v2 STT with Silero VAD utterance segmentation.

Design:
- A background thread reads raw PCM chunks fed by the session's mic loop.
- Each chunk runs through Silero VAD (already used by the gemini_live mixin).
- On speech onset we start buffering, including a small pre-roll.
- On `silence_duration_ms` of trailing silence (or max_utterance_ms reached)
  we lock the buffer and run Moonshine to get a transcript.
- The transcript is dropped onto an asyncio.Queue the session consumes.

Moonshine is loaded lazily on first feed so import-time isn't blocked by the
ONNX runtime spinning up. If the model fails to load the STT instance reports
itself dead so the session can refuse to run instead of looping forever.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SILERO_CHUNK = 512  # samples, fixed by silero


class MoonshineSTT:
    def __init__(self, config):
        self.config = config
        self._model_name = config.local_stt_model
        self._min_speech_ms = config.local_stt_min_speech_ms
        self._max_utterance_ms = config.local_stt_max_utterance_ms
        self._pre_roll_ms = config.local_stt_pre_roll_ms
        self._silence_ms = config.vad_silence_duration_ms
        self._vad_threshold = config.vad_silero_threshold

        self._model = None
        self._tokenizer = None
        self._silero = None

        self._chunk_queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=200)
        self._transcript_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._ready = False
        self._load_error: Optional[str] = None

        # speech-level flag for the session (mirrors gemini's _manual_vad_speaking)
        self.speaking = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def _load(self):
        if self._model is not None:
            return
        logger.info(f"loading moonshine model '{self._model_name}'...")
        # primary: useful-moonshine-onnx, package name moonshine_onnx
        try:
            from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        except ImportError as e:
            self._load_error = (
                "moonshine-onnx not installed. run: pip install useful-moonshine-onnx"
            )
            logger.error(self._load_error)
            raise RuntimeError(self._load_error) from e
        try:
            self._model = MoonshineOnnxModel(model_name=self._model_name)
            self._tokenizer = load_tokenizer()
        except Exception as e:
            self._load_error = f"moonshine load failed for '{self._model_name}': {e}"
            logger.error(self._load_error)
            raise

        # silero vad - reuse the same torch hub model the gemini path uses
        import torch
        torch.set_num_threads(1)
        sm, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        sm.eval()
        self._silero = sm
        logger.info(f"moonshine + silero ready (model={self._model_name})")

    def start(self):
        if self._running:
            return
        try:
            self._load()
        except Exception:
            return
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._worker_thread = threading.Thread(
            target=self._worker, name="moonshine-stt", daemon=True
        )
        self._worker_thread.start()
        self._ready = True

    def stop(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ── feeders / consumers ──────────────────────────────────────────────

    def feed_audio(self, pcm16_chunk: bytes):
        """Called from the session's mic loop (async). Non-blocking, drops
        the oldest chunk if the worker is falling behind."""
        try:
            self._chunk_queue.put_nowait(pcm16_chunk)
        except asyncio.QueueFull:
            try:
                self._chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._chunk_queue.put_nowait(pcm16_chunk)
            except asyncio.QueueFull:
                pass

    async def next_transcript(self, timeout: float = 0.5) -> Optional[str]:
        try:
            return await asyncio.wait_for(self._transcript_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ── worker ───────────────────────────────────────────────────────────

    def _vad_prob(self, samples_f32: np.ndarray) -> float:
        import torch
        max_prob = 0.0
        n = len(samples_f32)
        with torch.no_grad():
            for i in range(0, n - SILERO_CHUNK + 1, SILERO_CHUNK):
                t = torch.from_numpy(samples_f32[i:i + SILERO_CHUNK])
                p = self._silero(t, SAMPLE_RATE).item()
                if p > max_prob:
                    max_prob = p
        return max_prob

    def _drain_chunks(self) -> list[bytes]:
        """Pull whatever is sitting in the asyncio.Queue right now into a list,
        called from the worker thread via run_coroutine_threadsafe."""
        # we go through the loop because asyncio.Queue is not thread-safe.
        if not self._loop:
            return []
        fut = asyncio.run_coroutine_threadsafe(self._collect(), self._loop)
        try:
            return fut.result(timeout=0.2)
        except Exception:
            return []

    async def _collect(self) -> list[bytes]:
        out = []
        while not self._chunk_queue.empty():
            try:
                out.append(self._chunk_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not out:
            try:
                first = await asyncio.wait_for(self._chunk_queue.get(), timeout=0.1)
                out.append(first)
            except asyncio.TimeoutError:
                pass
        return out

    def _push_transcript(self, text: str):
        if not text or not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._transcript_queue.put(text), self._loop,
            )
        except Exception as e:
            logger.debug(f"push transcript failed: {e}")

    def _worker(self):
        pre_roll_bytes = int(self._pre_roll_ms / 1000 * SAMPLE_RATE) * 2
        max_bytes = int(self._max_utterance_ms / 1000 * SAMPLE_RATE) * 2
        min_bytes = int(self._min_speech_ms / 1000 * SAMPLE_RATE) * 2

        # ring buffer for pre-roll, only used when not speaking
        pre_roll = deque(maxlen=max(1, pre_roll_bytes))

        utterance = bytearray()
        speaking = False
        silence_started = 0.0
        utterance_started = 0.0

        while self._running:
            chunks = self._drain_chunks()
            if not chunks:
                continue

            for raw in chunks:
                if not raw:
                    continue
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                try:
                    prob = self._vad_prob(samples)
                except Exception as e:
                    logger.debug(f"vad failed: {e}")
                    prob = 0.0
                is_speech = prob >= self._vad_threshold

                if is_speech:
                    if not speaking:
                        # speech onset: prepend pre-roll
                        speaking = True
                        self.speaking = True
                        utterance_started = time.time()
                        utterance = bytearray(pre_roll)
                        pre_roll.clear()
                    utterance.extend(raw)
                    silence_started = 0.0
                else:
                    if speaking:
                        utterance.extend(raw)
                        if silence_started == 0.0:
                            silence_started = time.time()
                        elapsed = (time.time() - silence_started) * 1000
                        if elapsed >= self._silence_ms:
                            self._finalize(utterance, min_bytes)
                            utterance = bytearray()
                            speaking = False
                            self.speaking = False
                            silence_started = 0.0
                    else:
                        # idle silence, accumulate pre-roll
                        pre_roll.extend(raw)

                if speaking and len(utterance) >= max_bytes:
                    logger.debug("utterance hit max length, force finalizing")
                    self._finalize(utterance, min_bytes)
                    utterance = bytearray()
                    speaking = False
                    self.speaking = False
                    silence_started = 0.0

    def _finalize(self, audio: bytearray, min_bytes: int):
        if len(audio) < min_bytes:
            logger.debug(
                f"dropping short utterance ({len(audio)} bytes < {min_bytes})"
            )
            return
        try:
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            # moonshine expects shape (1, N)
            batch = samples.reshape(1, -1)
            tokens = self._model.generate(batch)
            text = self._tokenizer.decode_batch(tokens)[0].strip()
        except Exception as e:
            logger.error(f"moonshine transcribe failed: {e}")
            return
        if text:
            logger.info(f"stt: {text}")
            self._push_transcript(text)
