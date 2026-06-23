"""Silero VAD + parakeet.cpp STT for the local backend.

parakeet.cpp (an NVIDIA NeMo Parakeet port on ggml) does the actual speech
recognition. It loads a GGUF model through a small ctypes binding (see
parakeet_capi.py) and runs on CPU or any GPU via the prebuilt vulkan build,
both auto downloaded on first use (see parakeet_assets.py).

Two paths, picked automatically from the model:

  streaming  (nemotron-*-streaming, realtime_eou_*): mic audio is fed straight
             into parakeet's cache aware streaming decoder, which surfaces text
             as it finalizes plus end of utterance (<EOU>) / backchannel (<EOB>)
             events. We take the turn on <EOU>, or on a Silero silence timeout
             for streaming models that don't emit one.

  offline    (tdt / ctc / rnnt / hybrid): Silero VAD segments the mic stream
             into utterances and each finished segment is batch transcribed.

Silero VAD (the same model gemini_live uses) drives the .speaking flag for
barge-in in both modes, and the utterance segmentation in offline mode.

Public surface (kept stable for the orchestrator):

    start() / stop()
    feed_audio(pcm16_bytes)
    await next_transcript(timeout)
    .speaking          true while the user is mid utterance
    .partial_text      latest partial (best-effort)
    .ready / .load_error
"""

from __future__ import annotations

import asyncio
import collections
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SILERO_CHUNK = 512  # silero requires exactly 512 samples at 16khz per inference

# offline decoder selection passed to parakeet_capi_transcribe_pcm_lang.
_DECODER_MAP = {"default": 0, "ctc": 1, "tdt": 2, "rnnt": 2}

# substrings that mark a model as a cache-aware streaming model.
_STREAMING_HINTS = ("streaming", "eou", "realtime")


class BaseSTTProvider:
    """Interface a custom STT / ASR provider has to implement to drop into
    the local backend.

    Plugins register one with `ctx.register_stt("my_asr", factory)` and
    point `local.stt.external_provider: my_asr` at it in config.yml. The
    host then builds it with `factory(config)` and drives it from the
    local session run loop exactly like the built in parakeet pipeline.

    Subclass this for a head start (the optional bits already have sane
    defaults), or just duck-type the same surface. The orchestrator only
    ever touches the members documented below, so anything matching this
    shape works.

    Lifecycle / threading contract:
      - `start()` is called once on the asyncio loop thread before audio
        flows. Do blocking model loads here. Set `ready` False and put a
        message in `load_error` if you fail, the host aborts cleanly.
      - `feed_audio(pcm16)` is called for every mic chunk: raw int16 mono
        PCM at 16 kHz. It runs on the loop thread and must stay cheap, so
        push heavy transcription onto your own worker thread.
      - `next_transcript()` is awaited by the host to pull finished text.
        Return a finalized utterance string, or None on timeout.
      - flip `speaking` True while the user is mid-utterance so barge-in
        and idle detection work. Leave it False otherwise.
    """

    # True between speech-start and the silence timeout. The session reads
    # this for barge-in and idle suppression, keep it honest.
    speaking: bool = False
    # Best-effort latest partial transcript. Fine to leave as "".
    partial_text: str = ""

    def __init__(self, config):
        self.config = config

    def start(self) -> None:
        """Load models and begin. Called once before audio is fed."""
        pass

    def stop(self) -> None:
        """Release models / threads. Called once on shutdown."""
        pass

    @property
    def ready(self) -> bool:
        """True once `start()` succeeded and audio can be fed."""
        return True

    @property
    def load_error(self):
        """Human readable reason `start()` failed, else None."""
        return None

    def feed_audio(self, pcm16_chunk: bytes) -> None:
        """Consume a raw int16 mono 16 kHz PCM chunk from the mic. Keep it
        cheap, this fires ~16x a second on the loop thread."""
        raise NotImplementedError

    async def next_transcript(self, timeout: float = 0.5):
        """Return the next finalized transcript string, or None on timeout."""
        raise NotImplementedError

    def drain_pending(self) -> list[str]:
        """Pop every transcript currently queued, in order. Used to coalesce
        backlogged utterances and to flush stale ones after a barge-in.
        Return [] when nothing is waiting."""
        return []


class ParakeetSTT(BaseSTTProvider):
    """Silero VAD + parakeet.cpp transcription. Auto picks a streaming or
    offline path based on the configured model."""

    def __init__(self, config):
        self.config = config
        self._model = (config.local_stt_model or "nemotron-3.5-asr-streaming-0.6b").strip()
        self._quant = (config.local_stt_quant or "q8_0").strip()
        self._compute = (config.local_stt_compute or "auto").strip().lower()
        self._language = (config.local_stt_language or "auto").strip()
        self._mode_cfg = (config.local_stt_mode or "auto").strip().lower()
        self._decoder = _DECODER_MAP.get(
            (config.local_stt_decoder or "default").strip().lower(), 0,
        )

        # VAD knobs. read from local.stt.* first, fall back to the shared
        # gemini.vad.* keys so a single value can drive both backends.
        self._vad_threshold = float(
            config.get("local", "stt", "vad_threshold", default=None)
            or config.vad_silero_threshold
        )
        self._silence_ms = int(
            config.get("local", "stt", "silence_ms", default=None)
            or config.vad_silence_duration_ms
            or 600
        )
        self._min_speech_ms = int(config.local_stt_min_speech_ms)
        self._max_utterance_ms = int(config.local_stt_max_utterance_ms)
        self._pre_roll_ms = int(config.local_stt_pre_roll_ms)

        # parakeet handles
        self._lib = None
        self._ctx = None       # parakeet_capi.ParakeetContext
        self._stream = None    # parakeet_capi.ParakeetStream (streaming mode)
        self._variant = None
        self._abi = 0
        self._streaming = False

        # silero
        self._silero = None
        self._torch = None

        self._transcript_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # silero windowing state
        self._silero_buf = np.zeros(0, dtype=np.float32)
        self._pre_roll = collections.deque(
            maxlen=max(1, int(self._pre_roll_ms / 1000 * SAMPLE_RATE / SILERO_CHUNK)),
        )
        # offline utterance buffering
        self._utterance: list[np.ndarray] = []
        self._utterance_samples = 0
        self._silence_samples = 0
        self._speech_start_ts = 0.0

        # streaming worker
        self._feed_queue: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stream_text = ""
        self._silence_finalize = threading.Event()

        self._transcribe_lock = threading.Lock()
        self._lock = threading.Lock()
        self._running = False
        self._ready = False
        self._load_error: Optional[str] = None

        # session-visible
        self.speaking = False
        self.partial_text = ""

    # ── lifecycle ────────────────────────────────────────────────────────

    def _resolve_streaming(self) -> bool:
        if self._mode_cfg == "streaming":
            return True
        if self._mode_cfg == "offline":
            return False
        name = self._model.lower()
        return any(h in name for h in _STREAMING_HINTS)

    def _load(self):
        if self._ctx is not None:
            return

        from .parakeet_assets import ensure_model, ensure_runtime, resolve_variants
        from .parakeet_capi import ParakeetContext, ParakeetError, load_library

        # 1. runtime dll: try each variant (vulkan then cpu under 'auto') until
        # one actually loads. a machine without a vulkan driver fails the CDLL
        # load and we drop to cpu.
        lib = None
        last_err: Optional[Exception] = None
        for variant in resolve_variants(self._compute):
            try:
                dll = ensure_runtime(variant)
            except Exception as e:
                last_err = e
                logger.warning(f"parakeet runtime '{variant}' download failed: {e}")
                continue
            try:
                lib = load_library(str(dll))
                self._variant = variant
                logger.info(f"loaded parakeet runtime ({variant}) from {dll}")
                break
            except OSError as e:
                last_err = e
                logger.warning(
                    f"parakeet '{variant}' dll did not load ({e}), trying next variant"
                )
        if lib is None:
            self._load_error = f"could not load any parakeet runtime: {last_err}"
            logger.error(self._load_error)
            raise RuntimeError(self._load_error)
        self._lib = lib

        # 2. model gguf
        try:
            gguf = ensure_model(self._model, self._quant)
        except Exception as e:
            self._load_error = (
                f"parakeet model download failed ({self._model}-{self._quant}): {e}"
            )
            logger.error(self._load_error)
            raise

        # 3. load the model
        try:
            self._ctx = ParakeetContext(lib, str(gguf))
        except ParakeetError as e:
            self._load_error = f"parakeet model load failed: {e}"
            logger.error(self._load_error)
            raise
        self._abi = self._ctx.abi_version

        # 4. pick mode and open a stream for streaming models
        self._streaming = self._resolve_streaming()
        if self._streaming:
            try:
                self._stream = self._ctx.begin_stream(self._language)
            except ParakeetError as e:
                logger.warning(
                    f"streaming session failed to start ({e}), using offline mode"
                )
                self._streaming = False

        # 5. silero vad
        try:
            import torch
            torch.set_num_threads(1)
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            model.eval()
            self._silero = model
            self._torch = torch
        except Exception as e:
            self._load_error = f"silero vad load failed: {e}"
            logger.error(self._load_error)
            raise

        logger.info(
            f"local stt ready (parakeet {self._model}-{self._quant} [{self._variant}], "
            f"mode={'streaming' if self._streaming else 'offline'}, abi={self._abi})"
        )

    def start(self):
        if self._running:
            return
        try:
            self._load()
        except Exception:
            return
        self._loop = asyncio.get_event_loop()
        self._running = True
        self._ready = True
        if self._streaming:
            self._worker = threading.Thread(
                target=self._stream_worker, name="parakeet-stream", daemon=True,
            )
            self._worker.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._ready = False
        if self._worker is not None:
            try:
                self._feed_queue.put_nowait(None)
            except Exception:
                pass
            self._worker.join(timeout=2.0)
            self._worker = None
        try:
            if self._stream is not None:
                self._stream.free()
                self._stream = None
        except Exception as e:
            logger.debug(f"stream free: {e}")
        try:
            if self._ctx is not None:
                self._ctx.free()
                self._ctx = None
        except Exception as e:
            logger.debug(f"context free: {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ── feed / consume ───────────────────────────────────────────────────

    def feed_audio(self, pcm16_chunk: bytes):
        """Push raw int16 mono PCM from the mic at 16kHz."""
        if not self._running or not pcm16_chunk:
            return
        try:
            samples = (
                np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )
        except Exception as e:
            logger.debug(f"pcm decode failed: {e}")
            return

        if self._streaming:
            self._streaming_vad(samples)
            # hand the whole chunk to parakeet's streaming decoder
            try:
                self._feed_queue.put_nowait(samples)
            except Exception as e:
                logger.debug(f"feed queue put: {e}")
            return

        # offline: silero segments the stream into utterances
        with self._lock:
            self._silero_buf = np.concatenate([self._silero_buf, samples])
            while self._silero_buf.size >= SILERO_CHUNK:
                window = self._silero_buf[:SILERO_CHUNK]
                self._silero_buf = self._silero_buf[SILERO_CHUNK:]
                try:
                    self._process_window(window.copy())
                except Exception as e:
                    logger.debug(f"vad window processing failed: {e}")

    def _silero_prob(self, window: np.ndarray) -> float:
        with self._torch.no_grad():
            return self._silero(self._torch.from_numpy(window), SAMPLE_RATE).item()

    def _streaming_vad(self, samples: np.ndarray):
        """Lightweight Silero pass for the streaming path: only updates the
        speaking flag and the silence timeout that finalizes a turn. parakeet
        keeps its own audio buffer, so we don't accumulate utterance audio."""
        self._silero_buf = np.concatenate([self._silero_buf, samples])
        while self._silero_buf.size >= SILERO_CHUNK:
            window = self._silero_buf[:SILERO_CHUNK]
            self._silero_buf = self._silero_buf[SILERO_CHUNK:]
            try:
                is_speech = self._silero_prob(window.copy()) >= self._vad_threshold
            except Exception as e:
                logger.debug(f"vad window failed: {e}")
                continue
            if is_speech:
                self._silence_samples = 0
                if not self.speaking:
                    self.speaking = True
            elif self.speaking:
                self._silence_samples += SILERO_CHUNK
                if self._silence_samples / SAMPLE_RATE * 1000 >= self._silence_ms:
                    self.speaking = False
                    self._silence_finalize.set()

    def _stream_worker(self):
        """Pull mic chunks and drive parakeet's streaming decoder, emitting a
        finished utterance on <EOU> or a Silero silence timeout."""
        from .parakeet_capi import PARAKEET_EVENT_EOU, ParakeetError

        while self._running:
            try:
                chunk = self._feed_queue.get(timeout=0.2)
            except queue.Empty:
                if self._silence_finalize.is_set():
                    self._silence_finalize.clear()
                    self._emit_stream_turn()
                continue
            if chunk is None:
                break
            try:
                text, events = self._stream.feed(chunk)
            except ParakeetError as e:
                logger.debug(f"stream feed error: {e}")
                continue
            if text:
                self._stream_text += text
                self.partial_text = self._stream_text.strip()
            # v5 reports EOU as bit 0 of the mask; v4 was an any-event 0/1 that
            # also lines up with bit 0. EOB-only (bit 1) keeps us listening.
            eou = bool(events & PARAKEET_EVENT_EOU) if self._abi >= 5 else bool(events)
            if eou or self._silence_finalize.is_set():
                self._silence_finalize.clear()
                self._emit_stream_turn()

    def _emit_stream_turn(self):
        utter = self._stream_text.strip()
        self._stream_text = ""
        self.partial_text = ""
        if utter:
            self._push_transcript(utter)

    def _process_window(self, window: np.ndarray):
        # silero inference (cheap, ~1-2ms cpu per 512-sample window)
        is_speech = self._silero_prob(window) >= self._vad_threshold

        if is_speech:
            self._silence_samples = 0
            if not self.speaking:
                self.speaking = True
                self._speech_start_ts = time.time()
                # prepend pre-roll so we don't clip the first phoneme
                if self._pre_roll:
                    self._utterance.append(np.concatenate(list(self._pre_roll)))
                    self._utterance_samples = sum(a.size for a in self._utterance)
                    self._pre_roll.clear()
            self._utterance.append(window)
            self._utterance_samples += window.size
            # hard cap on utterance length
            if self._utterance_samples >= self._max_utterance_ms * SAMPLE_RATE / 1000:
                self._finalize_utterance(reason="max_utterance")
        else:
            if self.speaking:
                # still buffer trailing silence so transcription has context,
                # but count it toward the silence timeout.
                self._utterance.append(window)
                self._utterance_samples += window.size
                self._silence_samples += window.size
                silence_ms = self._silence_samples / SAMPLE_RATE * 1000
                if silence_ms >= self._silence_ms:
                    self._finalize_utterance(reason="silence")
            else:
                # idle: keep window as pre-roll
                self._pre_roll.append(window)

    def _finalize_utterance(self, reason: str = "silence"):
        if not self._utterance:
            self.speaking = False
            return
        audio = np.concatenate(self._utterance)
        duration_ms = audio.size / SAMPLE_RATE * 1000
        self._utterance = []
        self._utterance_samples = 0
        self._silence_samples = 0
        self.speaking = False
        self.partial_text = ""

        if duration_ms < self._min_speech_ms:
            logger.debug(
                f"dropped short utterance ({duration_ms:.0f}ms < "
                f"{self._min_speech_ms}ms min, reason={reason})"
            )
            return

        logger.debug(
            f"finalizing utterance {duration_ms:.0f}ms (reason={reason})"
        )
        threading.Thread(
            target=self._transcribe_worker, args=(audio,), daemon=True,
        ).start()

    def _transcribe_worker(self, audio: np.ndarray):
        # serialize parakeet calls so spammed utterances don't run concurrently
        # on a context that isn't built for it.
        with self._transcribe_lock:
            try:
                text = self._ctx.transcribe(
                    audio, decoder=self._decoder, lang=self._language,
                ).strip()
                if text:
                    self._push_transcript(text)
                else:
                    logger.debug("parakeet returned empty transcript")
            except Exception as e:
                logger.warning(f"parakeet transcribe failed: {e}")

    async def next_transcript(self, timeout: float = 0.5) -> Optional[str]:
        try:
            return await asyncio.wait_for(self._transcript_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def drain_pending(self) -> list[str]:
        """Pop every transcript currently sitting in the queue, in order.
        Used to coalesce backlogged utterances or to flush stale ones after
        a barge-in. Returns [] when the queue is empty.
        """
        out: list[str] = []
        while True:
            try:
                out.append(self._transcript_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            except Exception:
                break
        return out

    def _push_transcript(self, text: str):
        if not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._transcript_queue.put(text), self._loop,
            )
        except Exception as e:
            logger.debug(f"push transcript: {e}")
