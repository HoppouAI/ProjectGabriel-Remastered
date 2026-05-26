"""Moonshine Voice streaming STT.

Uses the moonshine-voice package (pip install moonshine-voice) which wraps
the Moonshine streaming models with built-in VAD, segmentation, and an
event listener API. We pick the model arch from config (small / medium /
tiny streaming variants).

The Transcriber runs incremental decoding while the user is still talking
so completed-line latency is sub-200ms in practice. We expose:

- start() / stop()       lifecycle
- feed_audio(pcm16)      push raw int16 mono mic chunks
- await next_transcript() pop the next completed line text
- .speaking              true between line-start and line-complete
- .partial_text          most-recent in-progress transcript (for UI)
- .ready / .load_error   load status

Public surface matches the previous (Silero+ONNX) implementation so the
session orchestrator doesn't care which backend is wired up.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# config string -> moonshine_voice.ModelArch attribute name
_ARCH_MAP = {
    "tiny_streaming": "TINY_STREAMING",
    "small_streaming": "SMALL_STREAMING",
    "medium_streaming": "MEDIUM_STREAMING",
    # legacy aliases for users with old config
    "small": "SMALL_STREAMING",
    "medium": "MEDIUM_STREAMING",
    "tiny": "TINY_STREAMING",
    "moonshine/small": "SMALL_STREAMING",
    "moonshine/base": "SMALL_STREAMING",
}


class MoonshineSTT:
    def __init__(self, config):
        self.config = config
        self._model_name = (config.local_stt_model or "small_streaming").lower()
        self._language = getattr(config, "local_stt_language", "en") or "en"
        self._max_segment_ms = config.local_stt_max_utterance_ms
        self._vad_threshold = config.vad_silero_threshold

        self._transcriber = None
        self._listener = None

        self._transcript_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._lock = threading.Lock()
        self._running = False
        self._ready = False
        self._load_error: Optional[str] = None

        # session-visible state
        self.speaking = False
        self.partial_text = ""

    # ── lifecycle ────────────────────────────────────────────────────────

    def _load(self):
        if self._transcriber is not None:
            return
        try:
            import moonshine_voice as mv
        except ImportError as e:
            self._load_error = (
                "moonshine-voice not installed. run: pip install moonshine-voice"
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
            arch = getattr(mv.ModelArch, arch_key)
        except AttributeError as e:
            self._load_error = (
                f"moonshine_voice.ModelArch has no '{arch_key}' "
                f"(installed version may not ship streaming weights)"
            )
            logger.error(self._load_error)
            raise RuntimeError(self._load_error) from e

        logger.info(f"downloading/loading moonshine voice model arch={arch_key} lang={self._language}")
        try:
            model_path, model_arch = mv.download_model(
                language=self._language, model_arch=arch,
            )
        except TypeError:
            # older signatures may not accept model_arch
            model_path, model_arch = mv.download_model(language=self._language)

        options = {
            "vad_threshold": f"{self._vad_threshold:.3f}",
            "vad_max_segment_duration": f"{self._max_segment_ms / 1000:.2f}",
            "identify_speakers": "false",
            "return_audio_data": "false",
        }
        try:
            self._transcriber = mv.Transcriber(
                model_path=model_path,
                model_arch=model_arch,
                update_interval=0.3,
                options=options,
            )
        except Exception as e:
            self._load_error = f"failed to construct Transcriber: {e}"
            logger.error(self._load_error)
            raise

        # listener bridges moonshine events -> our async queue
        outer = self

        class _Listener(mv.TranscriptEventListener):
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

        self._listener = _Listener()
        self._transcriber.add_listener(self._listener)
        logger.info("moonshine voice streaming transcriber ready")

    def start(self):
        if self._running:
            return
        try:
            self._load()
        except Exception:
            return
        self._loop = asyncio.get_event_loop()
        try:
            self._transcriber.start()
        except Exception as e:
            self._load_error = f"transcriber.start() failed: {e}"
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
            if self._transcriber is not None:
                self._transcriber.stop()
        except Exception as e:
            logger.debug(f"transcriber stop: {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ── feed / consume ───────────────────────────────────────────────────

    def feed_audio(self, pcm16_chunk: bytes):
        """Push raw int16 mono PCM bytes at 16kHz from the mic loop. Convert
        to float32 [-1, 1] and hand to the moonshine transcriber, which
        runs its own VAD + incremental decoder."""
        if not self._running or self._transcriber is None or not pcm16_chunk:
            return
        try:
            samples = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            # add_audio releases the GIL inside the native lib so this is cheap
            self._transcriber.add_audio(samples, sample_rate=SAMPLE_RATE)
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
