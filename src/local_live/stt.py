"""Moonshine Voice streaming STT (moonshine-voice 0.0.59 windows wheel).

The package's Transcriber + Stream pair gives us proper incremental decoding
with built-in VAD/segmentation. We feed mic chunks into a Stream and a
TranscriptEventListener receives line-level events as the user talks.

The streaming weights (SMALL_STREAMING / MEDIUM_STREAMING / TINY_STREAMING)
emit partial text via on_line_text_changed and a final completed text on
on_line_completed, so we get sub-200ms latency between end-of-speech and
the final transcript.

Public surface kept stable for the orchestrator:

    start() / stop()
    feed_audio(pcm16_bytes)
    await next_transcript(timeout)
    .speaking          true between line start and line completed
    .partial_text      latest in-progress text (for UI)
    .ready / .load_error
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# config string -> ModelArch attr on moonshine_voice.transcriber.ModelArch
_ARCH_MAP = {
    "tiny_streaming": "TINY_STREAMING",
    "small_streaming": "SMALL_STREAMING",
    "medium_streaming": "MEDIUM_STREAMING",
    "base_streaming": "BASE_STREAMING",
    # legacy aliases
    "tiny": "TINY_STREAMING",
    "small": "SMALL_STREAMING",
    "medium": "MEDIUM_STREAMING",
    "moonshine/tiny": "TINY_STREAMING",
    "moonshine/base": "SMALL_STREAMING",
    "moonshine/small": "SMALL_STREAMING",
}


class MoonshineSTT:
    def __init__(self, config):
        self.config = config
        self._model_name = (config.local_stt_model or "small_streaming").lower()
        self._language = getattr(config, "local_stt_language", "en") or "en"
        self._max_segment_ms = config.local_stt_max_utterance_ms
        self._vad_threshold = float(config.vad_silero_threshold)
        self._update_interval = 0.3

        self._transcriber = None
        self._stream = None
        self._listener = None

        self._transcript_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._lock = threading.Lock()
        self._running = False
        self._ready = False
        self._load_error: Optional[str] = None

        # session-visible
        self.speaking = False
        self.partial_text = ""

    # ── lifecycle ────────────────────────────────────────────────────────

    def _load(self):
        if self._transcriber is not None:
            return
        try:
            from moonshine_voice import get_model_for_language
            from moonshine_voice.transcriber import (
                ModelArch,
                Transcriber,
                TranscriptEventListener,
            )
        except ImportError as e:
            self._load_error = (
                "moonshine-voice not installed. run: pip install moonshine-voice==0.0.59"
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
        try:
            arch = getattr(ModelArch, arch_key)
        except AttributeError as e:
            self._load_error = (
                f"ModelArch has no '{arch_key}' (installed moonshine-voice may be too old)"
            )
            logger.error(self._load_error)
            raise RuntimeError(self._load_error) from e

        logger.info(
            f"loading moonshine voice model lang={self._language} arch={arch_key}"
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

        # moonshine option keys are stringy on this version
        options = {
            "vad_threshold": f"{self._vad_threshold:.3f}",
            "vad_max_segment_duration": f"{self._max_segment_ms / 1000:.2f}",
            "identify_speakers": "false",
            "return_audio_data": "false",
        }
        try:
            self._transcriber = Transcriber(
                model_path=str(model_path),
                model_arch=model_arch,
                update_interval=self._update_interval,
                options=options,
            )
            self._stream = self._transcriber.create_stream(self._update_interval)
        except Exception as e:
            self._load_error = f"failed to construct Transcriber: {e}"
            logger.error(self._load_error)
            raise

        outer = self

        class _Listener(TranscriptEventListener):
            def on_line_started(self, event):
                outer.speaking = True
                outer.partial_text = ""

            def on_line_text_changed(self, event):
                txt = getattr(event.line, "text", "") or ""
                outer.partial_text = txt

            def on_line_completed(self, event):
                outer.speaking = False
                txt = (getattr(event.line, "text", "") or "").strip()
                outer.partial_text = ""
                if txt:
                    outer._push_transcript(txt)

            def on_error(self, error):
                logger.warning(f"moonshine stream error: {error}")

        self._listener = _Listener()
        self._stream.add_listener(self._listener)
        logger.info(f"moonshine voice ready (arch={arch_key})")

    def start(self):
        if self._running:
            return
        try:
            self._load()
        except Exception:
            return
        self._loop = asyncio.get_event_loop()
        try:
            self._stream.start()
        except Exception as e:
            self._load_error = f"stream.start() failed: {e}"
            logger.error(self._load_error)
            return
        self._running = True
        self._ready = True

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._ready = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception as e:
            logger.debug(f"stream stop: {e}")
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

    # ── feed / consume ───────────────────────────────────────────────────

    def feed_audio(self, pcm16_chunk: bytes):
        """Push raw int16 mono PCM from the mic at 16kHz. Converts to float32
        in [-1, 1] and hands to the moonshine Stream which runs its own VAD
        and incremental decode."""
        if not self._running or self._stream is None or not pcm16_chunk:
            return
        try:
            samples = (
                np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )
            self._stream.add_audio(samples, SAMPLE_RATE)
        except Exception as e:
            logger.debug(f"add_audio failed: {e}")

    async def next_transcript(self, timeout: float = 0.5) -> Optional[str]:
        try:
            return await asyncio.wait_for(self._transcript_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def _push_transcript(self, text: str):
        if not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._transcript_queue.put(text), self._loop,
            )
        except Exception as e:
            logger.debug(f"push transcript: {e}")
