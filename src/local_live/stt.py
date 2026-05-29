"""Silero VAD + Moonshine batch STT.

Why not use Moonshine's built-in streaming VAD? Its threshold isn't exposed
through the Python API, so we can't tune sensitivity. Instead we run Silero
VAD ourselves on the mic stream (same model gemini_live uses), and only
hand finalized speech segments to Moonshine for batch transcription.

Public surface (kept stable for the orchestrator):

    start() / stop()
    feed_audio(pcm16_bytes)
    await next_transcript(timeout)
    .speaking          true between silero speech-start and silence-timeout
    .partial_text      latest partial (best-effort, may stay empty)
    .ready / .load_error
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SILERO_CHUNK = 512  # silero requires exactly 512 samples at 16khz per inference

# config string -> ModelArch attr on moonshine_voice.transcriber.ModelArch.
# we use the non-streaming weights since we batch-transcribe per segment, but
# the streaming weights also accept batch input so we keep accepting both.
_ARCH_MAP = {
    "tiny": "TINY",
    "base": "BASE",
    "small": "SMALL",
    "medium": "MEDIUM",
    "tiny_streaming": "TINY_STREAMING",
    "small_streaming": "SMALL_STREAMING",
    "medium_streaming": "MEDIUM_STREAMING",
    "base_streaming": "BASE_STREAMING",
    "moonshine/tiny": "TINY_STREAMING",
    "moonshine/base": "SMALL_STREAMING",
    "moonshine/small": "SMALL_STREAMING",
}


class MoonshineSTT:
    """Name kept for backward compat with session.py imports; under the hood
    it's Silero VAD + Moonshine batch transcription."""

    def __init__(self, config):
        self.config = config
        self._model_name = (config.local_stt_model or "small_streaming").lower()
        self._language = getattr(config, "local_stt_language", "en") or "en"

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

        self._transcriber = None
        self._silero = None

        self._transcript_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # silero state
        self._silero_buf = np.zeros(0, dtype=np.float32)
        self._pre_roll = collections.deque(
            maxlen=max(1, int(self._pre_roll_ms / 1000 * SAMPLE_RATE / SILERO_CHUNK)),
        )
        self._utterance: list[np.ndarray] = []
        self._utterance_samples = 0
        self._silence_samples = 0
        self._speech_start_ts = 0.0

        self._transcribe_lock = threading.Lock()
        self._lock = threading.Lock()
        self._running = False
        self._ready = False
        self._load_error: Optional[str] = None

        # session-visible
        self.speaking = False
        self.partial_text = ""

    # lifecycle

    def _load(self):
        if self._transcriber is not None:
            return
        try:
            from moonshine_voice import get_model_for_language
            from moonshine_voice.transcriber import ModelArch, Transcriber
        except ImportError as e:
            self._load_error = (
                "moonshine-voice not installed (it's an optional dep for the local "
                "backend). run: pip install .[local]   or   "
                "pip install moonshine-voice==0.0.59"
            )
            logger.error(self._load_error)
            raise RuntimeError(self._load_error) from e

        arch_key = _ARCH_MAP.get(self._model_name)
        if arch_key is None:
            self._load_error = (
                f"unknown local_stt_model '{self._model_name}'. "
                f"valid: {sorted(set(_ARCH_MAP.keys()))}"
            )
            logger.error(self._load_error)
            raise RuntimeError(self._load_error)
        # try the configured arch, then fall back to non-streaming variant
        # since we batch-transcribe (streaming weights still accept batch but
        # the plain SMALL/BASE/etc are tuned for it).
        try:
            arch = getattr(ModelArch, arch_key)
        except AttributeError:
            fallback = arch_key.replace("_STREAMING", "")
            try:
                arch = getattr(ModelArch, fallback)
                arch_key = fallback
            except AttributeError as e:
                self._load_error = f"ModelArch has neither {arch_key} nor {fallback}"
                logger.error(self._load_error)
                raise RuntimeError(self._load_error) from e

        logger.info(
            f"loading moonshine model lang={self._language} arch={arch_key}"
        )
        try:
            model_path, model_arch = get_model_for_language(
                wanted_language=self._language,
                wanted_model_arch=arch,
            )
        except Exception as e:
            self._load_error = f"get_model_for_language failed: {e}"
            logger.error(self._load_error)
            raise

        try:
            self._transcriber = Transcriber(
                model_path=str(model_path),
                model_arch=model_arch,
            )
        except Exception as e:
            self._load_error = f"failed to construct Transcriber: {e}"
            logger.error(self._load_error)
            raise

        # load silero
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
            logger.info(
                f"silero vad loaded (threshold={self._vad_threshold:.2f}, "
                f"silence={self._silence_ms}ms)"
            )
        except Exception as e:
            self._load_error = f"silero vad load failed: {e}"
            logger.error(self._load_error)
            raise

        logger.info("local stt ready (silero vad + moonshine batch)")

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

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._ready = False
        try:
            if self._transcriber is not None:
                self._transcriber.close()
        except Exception as e:
            logger.debug(f"transcriber close: {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # feed / consume

    def feed_audio(self, pcm16_chunk: bytes):
        """Push raw int16 mono PCM from the mic at 16kHz. We chunk it into
        512-sample windows for silero, then either buffer (speaking) or
        retain as pre-roll (silence)."""
        if not self._running or not pcm16_chunk:
            return
        try:
            samples = (
                np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )
        except Exception as e:
            logger.debug(f"pcm decode failed: {e}")
            return

        with self._lock:
            self._silero_buf = np.concatenate([self._silero_buf, samples])
            while self._silero_buf.size >= SILERO_CHUNK:
                window = self._silero_buf[:SILERO_CHUNK]
                self._silero_buf = self._silero_buf[SILERO_CHUNK:]
                try:
                    self._process_window(window.copy())
                except Exception as e:
                    logger.debug(f"vad window processing failed: {e}")

    def _process_window(self, window: np.ndarray):
        # silero inference (cheap, ~1-2ms cpu per 512-sample window)
        with self._torch.no_grad():
            prob = self._silero(self._torch.from_numpy(window), SAMPLE_RATE).item()
        is_speech = prob >= self._vad_threshold

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
        # serialize moonshine calls to keep memory bounded if the user spams.
        with self._transcribe_lock:
            try:
                transcript = self._transcriber.transcribe_without_streaming(
                    audio.tolist(), sample_rate=SAMPLE_RATE,
                )
                text = " ".join(
                    (line.text or "").strip()
                    for line in transcript.lines
                    if (line.text or "").strip()
                ).strip()
                if text:
                    self._push_transcript(text)
                else:
                    logger.debug("moonshine returned empty transcript")
            except Exception as e:
                logger.warning(f"moonshine transcribe failed: {e}")

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
