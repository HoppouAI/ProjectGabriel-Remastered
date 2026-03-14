import asyncio
import base64
import io
import json
import logging
import queue
import re
import threading
import time

import httpx
import numpy as np
import soundfile as sf
from stream2sentence import generate_sentences

logger = logging.getLogger(__name__)

# Strip emoji / invisible symbols the TTS model cannot pronounce
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U000020E3"
    "\U00002600-\U000026FF"
    "\U00002300-\U000023FF"
    "\U0000200B-\U0000200F"
    "\U0000205F-\U00002060"
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    cleaned = _EMOJI_RE.sub(" ", text)
    return re.sub(r"  +", " ", cleaned).strip()


class QwenTTSProvider:
    """Streams output transcription text to a Qwen3 TTS server and produces PCM audio.

    Architecture:
    - Gemini Live stays in AUDIO mode (required by Live API).
    - When active, Gemini audio is discarded by the caller.
    - Output transcription text is fed in via `feed_text()`.
    - Text is buffered and split on sentence boundaries (dedicated thread).
    - Sentences are dispatched to concurrent async synthesis tasks (httpx SSE).
    - Returned audio chunks are resampled and queued for playback.
    - Pre-synthesis overlap: sentence N+1 synthesizes while N streams audio.
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
        self._max_concurrent = config.get("tts", "qwen3", "max_concurrent", default=2)

        # Thread-safe queue: receive_loop -> splitter thread
        self._text_queue = queue.Queue()
        # Thread-safe queue: splitter thread -> async dispatch task
        self._sentence_queue = queue.Queue()
        # Async queues: dispatch -> feeder -> audio output
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

    # ── Public API ───────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._interrupted = False
        import nltk
        nltk.download('punkt_tab', quiet=True)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
            follow_redirects=True,
        )
        self._splitter_thread = threading.Thread(target=self._splitter_loop, daemon=True)
        self._splitter_thread.start()
        logger.info("Qwen3 TTS started (mode=%s, url=%s)", self._mode, self._base_url)

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
            # Schedule close on the event loop if available
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self._client.aclose())
                )
            self._client = None

    def feed_text(self, text: str):
        """Feed a chunk of output transcription text. Called from receive_loop."""
        if not text:
            return
        self._interrupted = False
        logger.debug("TTS feed_text: %r", text)
        self._text_queue.put(text)

    def turn_complete(self):
        """Called when Gemini signals turn_complete. Flushes remaining text."""
        self._text_queue.put(None)

    def interrupt(self):
        """Called on interruption. Clears all queues, cancels in-flight synthesis.

        Sets _interrupted=True which stays set until feed_text() is called again
        with new speech. This prevents stale data from being processed."""
        self._interrupted = True
        # Clear text queue and unblock splitter
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break
        self._text_queue.put(None)
        # Clear sentence queue
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except queue.Empty:
                break
        # Cancel all in-flight synthesis tasks
        for task in self._synth_tasks:
            task.cancel()
        self._synth_tasks.clear()
        # Drain ready queue
        while not self._ready_queue.empty():
            try:
                self._ready_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Drain audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_audio(self) -> bytes | None:
        """Get next audio chunk (PCM 16-bit mono at target_sr). Async-safe."""
        self._ensure_async_tasks()
        try:
            return await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    # ── Async task management ────────────────────────────────────────────

    def _ensure_async_tasks(self):
        """Lazily create async dispatch/feeder tasks on first get_audio() call."""
        if self._async_tasks:
            return
        self._loop = asyncio.get_running_loop()
        self._synth_semaphore = asyncio.Semaphore(self._max_concurrent)
        self._async_tasks = [
            asyncio.create_task(self._dispatch_task()),
            asyncio.create_task(self._feeder_task()),
        ]

    # ── Splitter thread (sync - stream2sentence is blocking) ─────────────

    def _text_generator(self):
        """Yields text chunks from _text_queue until None sentinel.
        
        Also returns (flushing stream2sentence buffer) if no text arrives
        for 3 seconds -- prevents the last sentence from being held
        indefinitely while stream2sentence waits for more context chars.
        """
        last_text_time = time.monotonic()
        while True:
            try:
                chunk = self._text_queue.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() - last_text_time > 3.0:
                    return  # Timeout: flush stream2sentence buffer
                if self._interrupted or not self._running:
                    return
                continue
            if chunk is None:
                return
            last_text_time = time.monotonic()
            yield chunk

    def _splitter_loop(self):
        """Runs generate_sentences on incoming text, queues complete sentences."""
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
                    s = _strip_emojis(sentence)
                    if s:
                        logger.info("TTS sentence ready: %r", s[:80])
                        self._sentence_queue.put(s)
            except Exception as e:
                if not self._interrupted:
                    logger.error("Sentence splitter error: %s", e)

    # ── Async dispatch: sentences -> concurrent synthesis tasks ───────────

    async def _dispatch_task(self):
        """Pull sentences and launch synthesis tasks with ordered sub-queues.

        Each sentence gets its own asyncio.Queue. Sub-queues are pushed into
        _ready_queue in sentence order. Synthesis tasks fill sub-queues
        concurrently (bounded by semaphore), so sentence N+1 can start
        while sentence N is still streaming.
        """
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
        """Drain sub-queues in order into the main audio queue.

        While draining sentence N's audio, sentence N+1 is already
        synthesizing, so audio is ready immediately when N finishes.
        """
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

    # ── Async synthesis via httpx SSE ────────────────────────────────────

    def _build_form(self, text: str) -> dict[str, str]:
        """Build multipart form fields for the /generate/stream endpoint."""
        form: dict[str, str] = {
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

        return form

    async def _synthesize_async(self, text: str, sub_q: asyncio.Queue):
        """Stream audio from TTS server via SSE and fill the sub-queue."""
        if not self._client:
            sub_q.put_nowait(None)
            return

        async with self._synth_semaphore:
            form = self._build_form(text)
            files = None
            if self._mode == "voice_clone" and self._ref_audio:
                try:
                    files = {"ref_audio": open(self._ref_audio, "rb")}
                except FileNotFoundError:
                    logger.warning("Ref audio not found: %s", self._ref_audio)

            try:
                async with self._client.stream(
                    "POST",
                    f"{self._base_url}/generate/stream",
                    data=form,
                    files=files,
                ) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    async for raw in resp.aiter_text():
                        if self._interrupted or not self._running:
                            return
                        buffer += raw
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            try:
                                payload = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue

                            ptype = payload.get("type")
                            if ptype == "chunk":
                                pcm = self._decode_wav_chunk(payload["audio_b64"])
                                if pcm and not self._interrupted:
                                    sub_q.put_nowait(pcm)
                            elif ptype == "error":
                                logger.error("Qwen3 TTS error: %s", payload.get("message"))
                                return
                            elif ptype == "done":
                                return
            except httpx.ConnectError:
                logger.error("Cannot connect to Qwen3 TTS server at %s", self._base_url)
            except httpx.TimeoutException:
                logger.error("Qwen3 TTS request timed out")
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._interrupted:
                    logger.error("Qwen3 TTS stream error: %s", e)
            finally:
                sub_q.put_nowait(None)
                if files:
                    for f in files.values():
                        f.close()

    # ── Audio decoding ───────────────────────────────────────────────────

    def _decode_wav_chunk(self, audio_b64: str) -> bytes | None:
        """Decode base64 WAV chunk from TTS server, resample to target rate."""
        try:
            raw = base64.b64decode(audio_b64)
            audio_np, src_sr = sf.read(io.BytesIO(raw), dtype="float32")

            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            if src_sr != self._target_sr:
                audio_np = self._resample(audio_np, src_sr, self._target_sr)

            pcm = (audio_np * 32767).clip(-32767, 32767).astype(np.int16)
            return pcm.tobytes()
        except Exception as e:
            logger.error("Audio decode/resample error: %s", e)
            return None

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio
        ratio = dst_sr / src_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class HoppouTTSProvider:
    """Streams output transcription text to the Hoppou AI cloud TTS API.

    OpenAI-compatible endpoint returning raw int16 PCM at 24kHz.
    Same sentence splitting and pre-synthesis pipeline as QwenTTSProvider.
    """

    _SAMPLE_RATE = 24000

    def __init__(self, config):
        self._api_url = config.get("tts", "hoppou", "api_url", default="https://api.hoppou.ai/tts").rstrip("/")
        self._api_key = config.get("tts", "hoppou", "api_key", default="")
        self._voice = config.get("tts", "hoppou", "voice", default="alba")
        self._model = config.get("tts", "hoppou", "model", default="tts-1")
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

    # ── Public API ───────────────────────────────────────────────────────

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
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self._client.aclose())
                )
            self._client = None

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

    # ── Async task management ────────────────────────────────────────────

    def _ensure_async_tasks(self):
        if self._async_tasks:
            return
        self._loop = asyncio.get_running_loop()
        self._synth_semaphore = asyncio.Semaphore(2)
        self._async_tasks = [
            asyncio.create_task(self._dispatch_task()),
            asyncio.create_task(self._feeder_task()),
        ]

    # ── Splitter (same as QwenTTSProvider) ───────────────────────────────

    def _text_generator(self):
        last_text_time = time.monotonic()
        while True:
            try:
                chunk = self._text_queue.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() - last_text_time > 3.0:
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
                    s = _strip_emojis(sentence)
                    if s:
                        logger.info("TTS sentence ready: %r", s[:80])
                        self._sentence_queue.put(s)
            except Exception as e:
                if not self._interrupted:
                    logger.error("Sentence splitter error: %s", e)

    # ── Async dispatch + feeder (same pattern) ───────────────────────────

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

    # ── Synthesis via Hoppou API ─────────────────────────────────────────

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
