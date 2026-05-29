import asyncio
import logging
import queue
import threading
import time

import httpx
import numpy as np
from stream2sentence import generate_sentences

from ._helpers import _strip_audio_tags, _strip_emojis
from .qwen import QwenTTSProvider

logger = logging.getLogger(__name__)


class HoppouTTSProvider:
    """Streams output transcription text to the Hoppou AI cloud TTS API.

    OpenAI-compatible endpoint returning raw int16 PCM at 24kHz.
    Same sentence splitting and pre-synthesis pipeline as QwenTTSProvider.
    """

    _SAMPLE_RATE = 24000

    def __init__(self, config, voice_override=None):
        self._api_url = config.get("tts", "hoppou", "api_url", default="https://api.hoppou.ai/tts").rstrip("/")
        self._api_key = config.get("tts", "hoppou", "api_key", default="")
        vo = voice_override or {}
        self._voice = vo.get("voice", config.get("tts", "hoppou", "voice", default="alba"))
        self._model = vo.get("model", config.get("tts", "hoppou", "model", default="tts-1"))
        self._target_sr = config.get("audio", "receive_sample_rate", default=24000)

        self._text_queue = queue.Queue()
        self._sentence_queue = queue.Queue()
        self._ready_queue: asyncio.Queue[asyncio.Queue[bytes | None]] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        self._running = False
        self._interrupted = False
        self._splitter_thread: threading.Thread | None = None
        self._client: httpx.AsyncClient | None = None
        self._async_tasks: list[asyncio.Task] = []
        self._synth_tasks: set[asyncio.Task] = set()
        self._synth_semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- Public API -------------------------------------------------------

    def start(self):
        if self._running:
            return
        if not self._api_key:
            logger.error("Hoppou TTS requires an API key (tts.hoppou.api_key)")
            return
        self._running = True
        self._interrupted = False
        import nltk
        nltk.download('punkt_tab', quiet=True)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            follow_redirects=True,
        )
        self._splitter_thread = threading.Thread(target=self._splitter_loop, daemon=True)
        self._splitter_thread.start()
        logger.info("Hoppou TTS started (voice=%s, model=%s, url=%s)", self._voice, self._model, self._api_url)

    def stop(self):
        self._running = False
        self._interrupted = True
        self._text_queue.put(None)
        if self._splitter_thread:
            self._splitter_thread.join(timeout=3)
            self._splitter_thread = None
        for task in self._async_tasks:
            task.cancel()
        self._async_tasks.clear()
        for task in self._synth_tasks:
            task.cancel()
        self._synth_tasks.clear()
        if self._client:
            client = self._client
            self._client = None
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    lambda c=client: asyncio.ensure_future(c.aclose())
                )

    def feed_text(self, text: str):
        if not text:
            return
        self._interrupted = False
        self._text_queue.put(text)

    def turn_complete(self):
        self._text_queue.put(None)

    def interrupt(self):
        self._interrupted = True
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break
        self._text_queue.put(None)
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except queue.Empty:
                break
        for task in self._synth_tasks:
            task.cancel()
        self._synth_tasks.clear()
        while not self._ready_queue.empty():
            try:
                self._ready_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_audio(self) -> bytes | None:
        self._ensure_async_tasks()
        try:
            return await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    # -- Async task management --------------------------------------------

    def _ensure_async_tasks(self):
        if self._async_tasks:
            return
        self._loop = asyncio.get_running_loop()
        self._synth_semaphore = asyncio.Semaphore(2)
        self._async_tasks = [
            asyncio.create_task(self._dispatch_task()),
            asyncio.create_task(self._feeder_task()),
        ]

    # -- Splitter (same as QwenTTSProvider) -------------------------------

    def _text_generator(self):
        last_text_time = time.monotonic()
        while True:
            try:
                chunk = self._text_queue.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() - last_text_time > 1.5:
                    return
                if self._interrupted or not self._running:
                    return
                continue
            if chunk is None:
                return
            last_text_time = time.monotonic()
            yield chunk

    def _splitter_loop(self):
        while self._running:
            text_gen = self._text_generator()
            try:
                for sentence in generate_sentences(
                    text_gen,
                    minimum_sentence_length=10,
                    minimum_first_fragment_length=10,
                    quick_yield_single_sentence_fragment=True,
                    context_size=3,
                    context_size_look_overhead=3,
                    force_first_fragment_after_words=15,
                ):
                    if self._interrupted or not self._running:
                        break
                    s = _strip_audio_tags(_strip_emojis(sentence))
                    if s:
                        logger.info("TTS sentence ready: %r", s[:80])
                        self._sentence_queue.put(s)
            except Exception as e:
                if not self._interrupted:
                    logger.error("Sentence splitter error: %s", e)

    # -- Async dispatch + feeder (same pattern) ---------------------------

    async def _dispatch_task(self):
        while self._running:
            try:
                sentence = await asyncio.to_thread(
                    self._sentence_queue.get, True, 0.1
                )
            except queue.Empty:
                continue
            except Exception:
                if not self._running:
                    return
                continue
            if self._interrupted:
                continue
            logger.info("TTS dispatch: %r", sentence[:80])
            sub_q: asyncio.Queue[bytes | None] = asyncio.Queue()
            await self._ready_queue.put(sub_q)
            task = asyncio.create_task(self._synthesize_async(sentence, sub_q))
            self._synth_tasks.add(task)
            task.add_done_callback(self._synth_tasks.discard)

    async def _feeder_task(self):
        while self._running:
            try:
                sub_q = await asyncio.wait_for(self._ready_queue.get(), timeout=0.1)
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                continue
            while True:
                try:
                    pcm = await asyncio.wait_for(sub_q.get(), timeout=0.5)
                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    if self._interrupted or not self._running:
                        break
                    continue
                if pcm is None:
                    break
                if not self._interrupted:
                    await self._audio_queue.put(pcm)

    # -- Synthesis via Hoppou API -----------------------------------------

    async def _synthesize_async(self, text: str, sub_q: asyncio.Queue):
        if not self._client:
            sub_q.put_nowait(None)
            return

        async with self._synth_semaphore:
            try:
                async with self._client.stream(
                    "POST",
                    f"{self._api_url}/v1/audio/speech/stream",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "input": text,
                        "voice": self._voice,
                        "response_format": "pcm",
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        if self._interrupted or not self._running:
                            return
                        if chunk:
                            pcm = self._int16_to_playback(chunk)
                            if pcm:
                                sub_q.put_nowait(pcm)
            except httpx.ConnectError:
                logger.error("Cannot connect to Hoppou TTS at %s", self._api_url)
            except httpx.TimeoutException:
                logger.error("Hoppou TTS request timed out")
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._interrupted:
                    logger.error("Hoppou TTS stream error: %s", e)
            finally:
                sub_q.put_nowait(None)

    def _int16_to_playback(self, data: bytes) -> bytes | None:
        """Convert int16 PCM from the API to int16 PCM at target sample rate."""
        try:
            samples = np.frombuffer(data, dtype=np.int16)
            if self._SAMPLE_RATE != self._target_sr:
                float_samples = samples.astype(np.float32) / 32767.0
                float_samples = QwenTTSProvider._resample(float_samples, self._SAMPLE_RATE, self._target_sr)
                samples = (float_samples * 32767).clip(-32767, 32767).astype(np.int16)
            return samples.tobytes()
        except Exception as e:
            logger.error("Audio conversion error: %s", e)
            return None
