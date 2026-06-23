import asyncio
import base64
import io
import logging
import queue
import threading
import time

import httpx
import numpy as np
import soundfile as sf
from stream2sentence import generate_sentences

from ._helpers import _resample, _strip_audio_tags, _strip_emojis

logger = logging.getLogger(__name__)


class TikTokTTSProvider:
    """Streams output transcription text through the Weilbyte TikTok TTS API.

    Uses the free community proxy at tiktok-tts.weilnet.workers.dev which
    wraps TikTok's internal TTS service. No authentication required.
    Returns base64 MP3 audio which gets decoded to PCM for playback.

    Text has a 300 byte (UTF-8) limit per request so sentences are chunked.
    """

    _API_URL = "https://ottsy.weilbyte.dev/api/generation"
    _CHAR_LIMIT = 300

    def __init__(self, config, voice_override=None):
        vo = voice_override or {}
        self._voice = vo.get("voice", config.get("tts", "tiktok", "voice", default="en_us_001"))
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
        self._running = True
        self._interrupted = False
        import nltk
        nltk.download('punkt_tab', quiet=True)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            follow_redirects=True,
        )
        self._splitter_thread = threading.Thread(target=self._splitter_loop, daemon=True)
        self._splitter_thread.start()
        logger.info("TikTok TTS started (voice=%s)", self._voice)

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

    # -- Splitter (same pattern as other providers) -----------------------

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

    # -- Async dispatch + feeder ------------------------------------------

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

    # -- TikTok API synthesis ---------------------------------------------

    @staticmethod
    def _prepare_text(text: str) -> str:
        """Clean text for the TikTok TTS API."""
        t = text.replace("*", "")
        t = t.replace("+", "plus")
        t = t.replace("&", "and")
        t = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        return t

    async def _synthesize_async(self, text: str, sub_q: asyncio.Queue):
        """Call Weilbyte TikTok TTS API and decode the base64 MP3 response to PCM."""
        if not self._client:
            sub_q.put_nowait(None)
            return

        async with self._synth_semaphore:
            try:
                cleaned = self._prepare_text(text)
                chunks = self._chunk_text(cleaned, self._CHAR_LIMIT)

                for chunk in chunks:
                    if self._interrupted or not self._running:
                        return

                    resp = await self._client.post(
                        self._API_URL,
                        json={"text": chunk, "voice": self._voice},
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if data.get("data") is None:
                        error_msg = data.get("error", "Unknown error")
                        logger.error("TikTok TTS error: %s", error_msg)
                        return

                    pcm = self._decode_mp3_b64(data["data"])
                    if pcm and not self._interrupted:
                        sub_q.put_nowait(pcm)

            except httpx.ConnectError:
                logger.error("Cannot connect to TikTok TTS API")
            except httpx.TimeoutException:
                logger.error("TikTok TTS request timed out")
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._interrupted:
                    logger.error("TikTok TTS error: %s", e)
            finally:
                sub_q.put_nowait(None)

    @staticmethod
    def _chunk_text(text: str, limit: int) -> list[str]:
        """Split text into chunks that fit within the API char limit."""
        if len(text) <= limit:
            return [text]
        import textwrap
        return textwrap.wrap(text, width=limit, break_long_words=True, break_on_hyphens=False)

    def _decode_mp3_b64(self, b64_str: str) -> bytes | None:
        """Decode base64 MP3 from TikTok API into PCM int16 at target sample rate."""
        try:
            mp3_data = base64.b64decode(b64_str)
            audio_np, src_sr = sf.read(io.BytesIO(mp3_data), dtype="float32")

            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            if src_sr != self._target_sr:
                audio_np = _resample(audio_np, src_sr, self._target_sr)

            pcm = (audio_np * 32767).clip(-32767, 32767).astype(np.int16)
            return pcm.tobytes()
        except Exception as e:
            logger.error("TikTok TTS audio decode error: %s", e)
            return None
