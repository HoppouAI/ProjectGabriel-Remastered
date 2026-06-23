import asyncio
import logging
import queue
import threading
import time

import numpy as np
from stream2sentence import generate_sentences

from ._helpers import _resample, _strip_audio_tags, _strip_emojis

logger = logging.getLogger(__name__)


class Chirp3HDTTSProvider:
    """Streams output transcription text to Google Cloud Chirp 3: HD TTS.

    Uses the streaming_synthesize gRPC API for low-latency synthesis.
    Same sentence-splitting and pre-synthesis pipeline as the other providers.
    Audio is returned as LINEAR16 PCM at 24kHz by default.
    """

    _SAMPLE_RATE = 24000

    def __init__(self, config, voice_override=None):
        primary = config.get("tts", "chirp3_hd", "api_key", default="")
        backup = config.get("tts", "chirp3_hd", "backup_keys", default=[]) or []
        self._keys = [primary] if primary else []
        if backup:
            self._keys.extend(backup)
        self._key_index = 0
        vo = voice_override or {}
        self._voice_name = vo.get("voice", config.get("tts", "chirp3_hd", "voice", default="Kore")).strip()
        self._language_code = vo.get("language_code", config.get("tts", "chirp3_hd", "language_code", default="en-US")).strip()
        self._speaking_rate = vo.get("speaking_rate", config.get("tts", "chirp3_hd", "speaking_rate", default=1.0))
        self._target_sr = config.get("audio", "receive_sample_rate", default=24000)

        self._text_queue = queue.Queue()
        self._sentence_queue = queue.Queue()
        self._ready_queue: asyncio.Queue[asyncio.Queue[bytes | None]] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        self._running = False
        self._interrupted = False
        self._splitter_thread: threading.Thread | None = None
        self._client = None
        self._async_tasks: list[asyncio.Task] = []
        self._synth_tasks: set[asyncio.Task] = set()
        self._synth_semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _full_voice_name(self) -> str:
        return f"{self._language_code}-Chirp3-HD-{self._voice_name}"

    def _current_key(self) -> str:
        if not self._keys:
            return ""
        return self._keys[self._key_index]

    def _create_client(self):
        from google.cloud import texttospeech
        key = self._current_key()
        if key:
            from google.api_core.client_options import ClientOptions
            self._client = texttospeech.TextToSpeechClient(
                client_options=ClientOptions(api_key=key)
            )
        else:
            self._client = texttospeech.TextToSpeechClient()

    def _rotate_key(self) -> bool:
        if len(self._keys) <= 1:
            logger.warning("Chirp 3 HD: no backup keys available")
            return False
        old_idx = self._key_index
        self._key_index = (self._key_index + 1) % len(self._keys)
        if self._key_index == old_idx:
            return False
        logger.info("Chirp 3 HD: rotated to API key index %d", self._key_index)
        self._create_client()
        return True

    # -- Public API -------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._interrupted = False
        import nltk
        nltk.download('punkt_tab', quiet=True)

        self._create_client()

        self._splitter_thread = threading.Thread(target=self._splitter_loop, daemon=True)
        self._splitter_thread.start()
        logger.info(
            "Chirp 3 HD TTS started (voice=%s, lang=%s, rate=%.1f, keys=%d)",
            self._full_voice_name(), self._language_code, self._speaking_rate, max(len(self._keys), 1),
        )

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

    # -- Streaming synthesis via gRPC -------------------------------------

    def _synthesize_streaming(self, text: str, sub_q: asyncio.Queue):
        """Run streaming_synthesize, pushing PCM chunks into sub_q as they arrive."""
        from google.cloud import texttospeech

        voice_params = texttospeech.VoiceSelectionParams(
            name=self._full_voice_name(),
            language_code=self._language_code,
        )
        config_kwargs = {"voice": voice_params}
        if self._speaking_rate != 1.0:
            config_kwargs["streaming_audio_config"] = texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.PCM,
                speaking_rate=self._speaking_rate,
            )
        streaming_config = texttospeech.StreamingSynthesizeConfig(**config_kwargs)

        config_request = texttospeech.StreamingSynthesizeRequest(
            streaming_config=streaming_config,
        )

        def request_generator():
            yield config_request
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text),
            )

        from google.api_core.exceptions import ResourceExhausted
        retried = False
        while True:
            try:
                responses = self._client.streaming_synthesize(request_generator())
                for response in responses:
                    if self._interrupted:
                        break
                    if response.audio_content:
                        pcm = self._process_audio(response.audio_content)
                        if pcm and not self._interrupted:
                            self._loop.call_soon_threadsafe(sub_q.put_nowait, pcm)
                break
            except ResourceExhausted:
                if retried or not self._rotate_key():
                    logger.error("Chirp 3 HD: rate limited on all keys")
                    break
                retried = True
                logger.warning("Chirp 3 HD: rate limited, retrying with next key")
                continue
            except Exception as e:
                if not self._interrupted:
                    logger.error("Chirp 3 HD TTS error: %s", e)
                break
        self._loop.call_soon_threadsafe(sub_q.put_nowait, None)

    async def _synthesize_async(self, text: str, sub_q: asyncio.Queue):
        async with self._synth_semaphore:
            await asyncio.to_thread(self._synthesize_streaming, text, sub_q)

    # -- Audio processing -------------------------------------------------

    def _process_audio(self, data: bytes) -> bytes | None:
        try:
            samples = np.frombuffer(data, dtype=np.int16)
            if self._SAMPLE_RATE != self._target_sr:
                float_samples = samples.astype(np.float32) / 32767.0
                float_samples = _resample(float_samples, self._SAMPLE_RATE, self._target_sr)
                samples = (float_samples * 32767).clip(-32767, 32767).astype(np.int16)
            return samples.tobytes()
        except Exception as e:
            logger.error("Audio conversion error: %s", e)
            return None
