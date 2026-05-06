"""Suno bridge client + streaming MP3 player.

Talks to a small private bridge server on 127.0.0.1. We POST lyrics, get
back stream URLs, and play one through ffmpeg -> PyAudio.

Rate limited to one create request per N seconds (default 30).

Audio output goes through a separate PyAudio stream so it doesn't fight
the gemini playback path. While a song is playing the AudioManager's
voice fade kicks in (same path as local pygame music) so the mic stays
hot but the synthesised voice ducks out.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
import pyaudio

logger = logging.getLogger(__name__)


# Suno's audiopipe streams come out as stereo MP3, 44.1kHz is the safe target
SUNO_SR = 44100
SUNO_CH = 2
SUNO_BYTES_PER_SAMPLE = 2  # int16


def _sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename on Windows + posix."""
    import re
    name = (name or "").strip() or "suno_song"
    name = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:80] or "suno_song"


def _resolve_ffmpeg() -> Optional[str]:
    """Find an ffmpeg executable. Prefer imageio_ffmpeg's bundled binary."""
    try:
        import imageio_ffmpeg  # type: ignore
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path
    return None


@dataclass
class SunoClip:
    id: str
    title: str
    status: str
    stream_url: str
    audio_url: str = ""
    image_url: str = ""


@dataclass
class LibraryEntry:
    """A song in the operator's Suno library/playlist."""
    id: str
    title: str
    image_url: str = ""
    styles: str = ""
    lyrics: str = ""


@dataclass
class _PlayerState:
    clip: SunoClip
    play_start: Optional[float] = None
    output_bytes: int = 0           # PCM bytes ffmpeg has produced so far
    written_bytes: int = 0          # PCM bytes PyAudio has actually written
    finished: bool = False
    error: Optional[str] = None
    title: str = ""                 # latest known title
    audio_url: str = ""             # set when status flips to complete
    status: str = "submitted"
    _lock: threading.Lock = field(default_factory=threading.Lock)


class SunoBridgeClient:
    """Thin HTTP wrapper around the local bridge server."""

    def __init__(self, base_url: str, request_timeout: float = 140.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = request_timeout

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self.base_url}/api/v1/health")
            r.raise_for_status()
            return r.json()

    async def create_song(self, lyrics: str, style: str | None = None,
                          timeout_ms: int = 120000) -> list[SunoClip]:
        url = f"{self.base_url}/api/v1/songs?timeout_ms={timeout_ms}"
        body: dict = {"lyrics": lyrics}
        if style and style.strip():
            body["style"] = style.strip()
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(url, json=body)
            if r.status_code >= 400:
                try:
                    data = r.json()
                except Exception:
                    data = {"error": "unknown", "message": r.text}
                raise SunoError(r.status_code, data.get("error", "error"),
                                data.get("message", ""))
            data = r.json()
        return [SunoClip(
            id=s.get("id", ""),
            title=s.get("title", ""),
            status=s.get("status", "submitted"),
            stream_url=s.get("stream_url", ""),
            audio_url=s.get("audio_url", "") or "",
            image_url=s.get("image_url", "") or "",
        ) for s in data.get("songs", [])]

    async def get_clip(self, clip_id: str) -> SunoClip:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(f"{self.base_url}/api/v1/songs/{clip_id}")
            r.raise_for_status()
            d = r.json()
        return SunoClip(
            id=d.get("id", clip_id),
            title=d.get("title", ""),
            status=d.get("status", "submitted"),
            stream_url=d.get("stream_url", ""),
            audio_url=d.get("audio_url", "") or "",
            image_url=d.get("image_url", "") or "",
        )

    async def recent(self, since_ms: int | None = None) -> list[SunoClip]:
        """Fetch clips the bridge has sniffed recently.

        Used to recover when the original create call timed out at the
        bridge driver layer but the songs actually generated. Bridge
        endpoint: GET /api/v1/recent?since_ms=...
        """
        url = f"{self.base_url}/api/v1/recent"
        params = {}
        if since_ms is not None:
            params["since_ms"] = since_ms
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(url, params=params)
            if r.status_code == 404:
                return []  # bridge doesn't support /recent yet
            r.raise_for_status()
            data = r.json()
        return [SunoClip(
            id=s.get("id", ""),
            title=s.get("title", ""),
            status=s.get("status", "submitted"),
            stream_url=s.get("stream_url", ""),
            audio_url=s.get("audio_url", "") or "",
            image_url=s.get("image_url", "") or "",
        ) for s in data.get("songs", [])]

    async def library(self, q: str | None = None,
                      playlist_id: str | None = None,
                      refresh: bool = False,
                      timeout_ms: int = 90000) -> list[LibraryEntry]:
        """List songs in the operator's Suno playlist.

        The bridge handles caching and scrolling the virtualized list.
        """
        params: dict = {"timeout_ms": timeout_ms}
        if q:
            params["q"] = q
        if playlist_id:
            params["id"] = playlist_id
        if refresh:
            params["refresh"] = "1"
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/api/v1/library", params=params)
            if r.status_code >= 400:
                try:
                    data = r.json()
                except Exception:
                    data = {"error": "unknown", "message": r.text}
                raise SunoError(r.status_code, data.get("error", "error"),
                                data.get("message", ""))
            data = r.json()
        return [LibraryEntry(
            id=s.get("id", ""),
            title=s.get("title", ""),
            image_url=s.get("image_url", "") or "",
            styles=s.get("styles", "") or "",
            lyrics=s.get("lyrics", "") or "",
        ) for s in data.get("songs", [])]

    async def get_lyrics(self, clip_id: str) -> str:
        """Fetch lyrics for a single clip from the bridge cache."""
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(f"{self.base_url}/api/v1/songs/{clip_id}/lyrics")
            if r.status_code == 404:
                raise SunoError(404, "lyrics_not_found", "Clip has no lyrics (instrumental?) or hasn't been seen.")
            if r.status_code >= 400:
                try:
                    data = r.json()
                except Exception:
                    data = {"error": "unknown", "message": r.text}
                raise SunoError(r.status_code, data.get("error", "error"),
                                data.get("message", ""))
            data = r.json()
        return data.get("lyrics", "") or ""

    async def cover_song(self, source_id: str,
                         lyrics: str | None = None,
                         style: str | None = None,
                         timeout_ms: int = 120000) -> list[SunoClip]:
        """Cover an existing Suno song, optionally with new lyrics/style."""
        url = f"{self.base_url}/api/v1/cover?timeout_ms={timeout_ms}"
        body: dict = {"id": source_id}
        if lyrics and lyrics.strip():
            body["lyrics"] = lyrics
        if style and style.strip():
            body["style"] = style.strip()
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(url, json=body)
            if r.status_code >= 400:
                try:
                    data = r.json()
                except Exception:
                    data = {"error": "unknown", "message": r.text}
                raise SunoError(r.status_code, data.get("error", "error"),
                                data.get("message", ""))
            data = r.json()
        return [SunoClip(
            id=s.get("id", ""),
            title=s.get("title", ""),
            status=s.get("status", "submitted"),
            stream_url=s.get("stream_url", ""),
            audio_url=s.get("audio_url", "") or "",
            image_url=s.get("image_url", "") or "",
        ) for s in data.get("songs", [])]


class SunoError(Exception):
    def __init__(self, status: int, code: str, message: str = ""):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class SunoPlayer:
    """Plays a single Suno clip stream via ffmpeg -> PyAudio.

    Decoder runs in a thread (subprocess + blocking reads). Position/duration
    are derived from byte counters so the chatbox UI can show a moving bar
    as more audio buffers in.
    """

    def __init__(self, clip: SunoClip, audio_mgr, volume: int = 90):
        self.state = _PlayerState(clip=clip, title=clip.title or "Suno Song",
                                  status=clip.status or "streaming")
        self._audio = audio_mgr
        self._volume = max(0, min(200, volume)) / 100.0
        self._pya = audio_mgr.pya  # reuse PyAudio instance
        self._stream = None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._fade_volume = 1.0

    def start(self) -> bool:
        ffmpeg = _resolve_ffmpeg()
        if not ffmpeg:
            self.state.error = "ffmpeg not found (install imageio-ffmpeg or system ffmpeg)"
            logger.error(self.state.error)
            return False

        out_dev = getattr(self._audio, "output_device", None)
        try:
            self._stream = self._pya.open(
                format=pyaudio.paInt16,
                channels=SUNO_CH,
                rate=SUNO_SR,
                output=True,
                output_device_index=out_dev,
            )
        except Exception as e:
            self.state.error = f"pyaudio open failed: {e}"
            logger.error(self.state.error)
            return False

        cmd = [
            ffmpeg, "-loglevel", "error", "-nostdin",
            # Start decoding ASAP -- default probesize is 5MB which can add
            # several seconds of dead air before audio begins.
            "-probesize", "32", "-analyzeduration", "0",
            "-fflags", "+nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-i", self.state.clip.stream_url,
            "-f", "s16le", "-ar", str(SUNO_SR), "-ac", str(SUNO_CH),
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as e:
            self.state.error = f"ffmpeg spawn failed: {e}"
            logger.error(self.state.error)
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            return False

        self._thread = threading.Thread(target=self._pump_loop, daemon=True)
        self._thread.start()
        self.state.play_start = time.time()
        logger.info(f"Suno playback started: {self.state.clip.id}")
        return True

    def _pump_loop(self):
        chunk_size = SUNO_SR * SUNO_CH * SUNO_BYTES_PER_SAMPLE // 10  # ~100ms
        try:
            while not self._stop_flag.is_set():
                data = self._proc.stdout.read(chunk_size)
                if not data:
                    break
                self.state.output_bytes += len(data)
                samples = np.frombuffer(data, dtype=np.int16)
                vol = self._volume * self._fade_volume
                if vol != 1.0:
                    scaled = (samples.astype(np.float32) * vol).clip(-32768, 32767).astype(np.int16)
                    data = scaled.tobytes()
                try:
                    self._stream.write(data)
                except Exception:
                    break
                self.state.written_bytes += len(data)
        except Exception as e:
            logger.error(f"Suno pump loop error: {e}")
            self.state.error = str(e)
        finally:
            self.state.finished = True
            try:
                if self._stream:
                    self._stream.stop_stream()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
            except Exception:
                pass
            logger.info(f"Suno playback ended: {self.state.clip.id}")

    def stop(self):
        if self._stop_flag.is_set():
            return
        self._stop_flag.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def is_playing(self) -> bool:
        return not self.state.finished and self.state.play_start is not None

    def get_progress(self) -> dict:
        s = self.state
        position = 0.0
        if s.play_start is not None:
            position = max(0.0, time.time() - s.play_start)
        # Total decoded seconds gives us a moving "duration" until the song
        # finishes generating. ffmpeg reads ahead of playback so this is a
        # safe upper bound for the bar.
        decoded = s.output_bytes / float(SUNO_SR * SUNO_CH * SUNO_BYTES_PER_SAMPLE)
        if decoded < 0.1:
            decoded = max(decoded, position)  # avoid div by zero on cold start
        position = min(position, decoded if decoded > 0 else position)
        progress = (position / decoded) if decoded > 0 else 0.0
        return {
            "song_name": s.title or "Suno Song",
            "position": position,
            "duration": decoded,
            "progress": min(1.0, max(0.0, progress)),
            "streaming": s.status != "complete",
            "status": s.status,
        }


class SunoManager:
    """Lifecycle + rate limiting for Suno song generation and playback."""

    def __init__(self, config, audio_mgr):
        self._config = config
        self._audio = audio_mgr
        self._client = SunoBridgeClient(
            base_url=config.get("suno", "bridge_url", default="http://127.0.0.1:8787")
        )
        self._min_interval = float(config.get("suno", "min_request_interval_seconds", default=30.0))
        self._volume = int(config.get("suno", "volume", default=90))
        self._max_lyrics = int(config.get("suno", "max_lyrics_chars", default=6000))
        self._max_style = int(config.get("suno", "max_style_chars", default=1000))
        self._save_dir = config.get("suno", "save_dir", default="sfx/music/suno")
        self._save_enabled = bool(config.get("suno", "save_finished_songs", default=True))
        self._last_request_at: float = 0.0
        self._player: Optional[SunoPlayer] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._gen_task: Optional[asyncio.Task] = None
        self._generating: bool = False
        self._generating_started_at: float = 0.0
        self._generating_error: Optional[str] = None
        self._last_chosen: Optional[SunoClip] = None
        self._last_other: Optional[SunoClip] = None
        self._lock = asyncio.Lock()

    @property
    def is_playing(self) -> bool:
        return self._player is not None and self._player.is_playing

    @property
    def is_generating(self) -> bool:
        return self._generating

    @property
    def is_active(self) -> bool:
        return self.is_playing or self.is_generating

    def get_progress(self) -> Optional[dict]:
        if self.is_playing:
            return self._player.get_progress()
        if self._generating:
            elapsed = max(0.0, time.time() - self._generating_started_at)
            return {
                "song_name": "Generating song...",
                "position": elapsed,
                "duration": 0.0,
                "progress": 0.0,
                "streaming": True,
                "status": "generating",
            }
        return None

    def cooldown_remaining(self) -> float:
        if self._last_request_at == 0:
            return 0.0
        elapsed = time.time() - self._last_request_at
        return max(0.0, self._min_interval - elapsed)

    async def generate(self, lyrics: str, style: str | None = None) -> dict:
        # Fast-path validation -- the actual bridge call happens in a
        # background task so the function response goes back to gemini
        # immediately. Otherwise the model sits silent for 5-15 seconds
        # while suno warms up, which trips its session watchdogs.
        if self._generating:
            return {"result": "error", "code": "already_generating",
                    "message": "A song is already being generated, wait for it."}
        cd = self.cooldown_remaining()
        if cd > 0:
            return {"result": "error", "code": "rate_limited",
                    "message": f"Please wait {cd:.0f}s before generating another song."}
        if not lyrics or not lyrics.strip():
            return {"result": "error", "code": "lyrics_required",
                    "message": "Lyrics are required."}
        if len(lyrics) > self._max_lyrics:
            return {"result": "error", "code": "lyrics_too_long",
                    "message": f"Lyrics exceed max length ({self._max_lyrics} chars)."}
        if style and len(style) > self._max_style:
            return {"result": "error", "code": "style_too_long",
                    "message": f"Style exceeds max length ({self._max_style} chars)."}

        # Stop any existing playback before starting a new one
        if self._player is not None:
            await asyncio.to_thread(self._player.stop)
            self._player = None
            if self._poll_task and not self._poll_task.done():
                self._poll_task.cancel()
                self._poll_task = None
            self._audio.set_external_music_active(False)

        self._last_request_at = time.time()
        self._generating = True
        self._generating_started_at = self._last_request_at
        self._generating_error = None
        # Mark external music active right away so the chatbox UI takes over
        # and the AI's voice ducks while the song spins up.
        self._audio.set_external_music_active(True)
        self._gen_task = asyncio.create_task(self._generate_and_play(lyrics, style))

        return {"result": "ok", "code": "submitted",
                "message": "Song generation started. Audio will begin streaming in a few seconds. "
                           "Stay quiet until the music kicks in."}

    async def _generate_and_play(self, lyrics: str, style: str | None = None):
        """Background: call the bridge to create a new song, then play it."""
        async def create():
            return await self._client.create_song(lyrics, style=style)
        await self._run_and_play(create, label="generate")

    async def _cover_and_play(self, source_id: str,
                              lyrics: str | None = None,
                              style: str | None = None):
        """Background: call the bridge to cover a song, then play it."""
        async def cover():
            return await self._client.cover_song(source_id, lyrics=lyrics, style=style)
        await self._run_and_play(cover, label="cover")

    async def _run_and_play(self, create_fn, label: str = "generate"):
        request_started_ms = int(self._generating_started_at * 1000) - 2000
        async with self._lock:
            try:
                clips: list[SunoClip] = []
                try:
                    clips = await create_fn()
                except SunoError as e:
                    err_blob = (e.code + " " + e.message).lower()
                    is_timeout = "timeout" in err_blob
                    if is_timeout:
                        # Bridge driver gave up but suno may have generated the
                        # songs anyway. Poll /recent to recover the IDs.
                        logger.warning(f"Suno {label} timed out, polling /recent for sniffed clips...")
                        clips = await self._wait_for_recent(request_started_ms,
                                                            poll_seconds=60.0)
                        if not clips:
                            self._generating_error = (
                                "bridge_timeout: bridge gave up and no clips appeared "
                                "in /recent within 60s. The operator should refresh "
                                "the suno tab."
                            )
                            logger.error(self._generating_error)
                            return
                        logger.info(f"Recovered {len(clips)} clip(s) from /recent")
                    else:
                        self._generating_error = f"{e.code}: {e.message}"
                        logger.error(f"Suno {label} failed: {self._generating_error}")
                        return
                except httpx.HTTPError as e:
                    self._generating_error = f"bridge_unreachable: {e}"
                    logger.error(f"Suno {label} failed: {self._generating_error}")
                    return

                valid = [c for c in clips if c.stream_url]
                if not valid and clips:
                    # Bridge gave us clip IDs but no stream URLs yet (happens
                    # on cover sometimes). Poll get_clip() per id until they
                    # populate.
                    logger.info(f"Suno returned {len(clips)} clip(s) without stream URLs, polling...")
                    valid = await self._wait_for_streams([c.id for c in clips], poll_seconds=60.0)
                if not valid:
                    # Last resort: maybe /recent has them
                    logger.warning("No stream URLs after polling clip IDs, trying /recent...")
                    request_started_ms = int(self._generating_started_at * 1000) - 2000
                    valid = await self._wait_for_recent(request_started_ms, poll_seconds=30.0)
                if not valid:
                    self._generating_error = "no_stream"
                    logger.error("Suno returned no playable stream URL")
                    return
                chosen = valid[0]
                other = valid[1] if len(valid) > 1 else None

                player = SunoPlayer(chosen, self._audio, volume=self._volume)
                ok = await asyncio.to_thread(player.start)
                if not ok:
                    self._generating_error = player.state.error or "playback_failed"
                    logger.error(f"Suno playback failed: {self._generating_error}")
                    return

                self._player = player
                self._last_chosen = chosen
                self._last_other = other
                self._poll_task = asyncio.create_task(self._poll_loop(chosen.id))
            finally:
                self._generating = False
                if self._generating_error and self._player is None:
                    # Failure path -- release the audio fade so the AI can talk again
                    self._audio.set_external_music_active(False)

    async def _wait_for_recent(self, since_ms: int, poll_seconds: float = 60.0,
                               interval: float = 3.0) -> list[SunoClip]:
        """Poll /recent until we get clips with stream_urls or run out of time."""
        deadline = time.monotonic() + poll_seconds
        while time.monotonic() < deadline:
            try:
                clips = await self._client.recent(since_ms=since_ms)
                ready = [c for c in clips if c.stream_url]
                if ready:
                    return ready
            except Exception as e:
                logger.debug(f"Suno /recent poll error: {e}")
            await asyncio.sleep(interval)
        return []

    async def _wait_for_streams(self, clip_ids: list[str], poll_seconds: float = 60.0,
                                interval: float = 2.0) -> list[SunoClip]:
        """Poll get_clip() per id until any of them have a stream URL.

        Used when the bridge returns clip ids but no stream_url (cover sometimes
        comes back early before the sniff lands).
        """
        deadline = time.monotonic() + poll_seconds
        while time.monotonic() < deadline:
            ready: list[SunoClip] = []
            for cid in clip_ids:
                try:
                    c = await self._client.get_clip(cid)
                    if c.stream_url:
                        ready.append(c)
                except Exception as e:
                    logger.debug(f"Suno get_clip poll error for {cid}: {e}")
            if ready:
                return ready
            await asyncio.sleep(interval)
        return []

    async def stop(self) -> dict:
        # Cancel an in-flight generation first so it doesn't start playing
        # right after we say "stop".
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except (asyncio.CancelledError, Exception):
                pass
            self._gen_task = None
        async with self._lock:
            self._generating = False
            if self._player is None:
                self._audio.set_external_music_active(False)
                return {"result": "ok", "message": "Nothing playing"}
            await asyncio.to_thread(self._player.stop)
            self._player = None
            if self._poll_task and not self._poll_task.done():
                self._poll_task.cancel()
            self._poll_task = None
            self._audio.set_external_music_active(False)
            return {"result": "ok", "message": "Stopped"}

    async def replay(self, which: str = "last") -> dict:
        """Replay one of the last generated clips without spending a credit.

        which: "last"  -> the clip we picked and played
               "other" -> the alternate clip suno generated alongside it
        """
        clip = self._last_chosen if which != "other" else self._last_other
        if clip is None:
            if which == "other":
                return {"result": "error", "code": "no_other",
                        "message": "Suno only returned one clip last time, no alternate to play."}
            return {"result": "error", "code": "no_last",
                    "message": "No previously generated song to replay."}
        if not clip.stream_url:
            return {"result": "error", "code": "no_stream",
                    "message": "That clip has no stream URL anymore (probably expired). Generate a new one."}
        # Stop anything currently playing first
        if self._player is not None or self._generating:
            await self.stop()
        async with self._lock:
            player = SunoPlayer(clip, self._audio, volume=self._volume)
            ok = await asyncio.to_thread(player.start)
            if not ok:
                err = player.state.error or "playback_failed"
                return {"result": "error", "code": "playback_failed", "message": err}
            self._player = player
            # Light poll loop so the chatbox stays updated and a late
            # save still happens for the alternate clip if needed.
            self._poll_task = asyncio.create_task(self._poll_loop(clip.id))
        return {"result": "ok", "code": "replaying",
                "message": f"Replaying {'alternate' if which == 'other' else 'last'} song: {clip.title or clip.id}"}

    async def search_library(self, query: str | None = None,
                             limit: int = 25) -> dict:
        """Search the operator's Suno playlist for songs to cover."""
        try:
            entries = await self._client.library(q=query)
        except SunoError as e:
            return {"result": "error", "code": e.code, "message": e.message or str(e)}
        except httpx.HTTPError as e:
            return {"result": "error", "code": "bridge_unreachable", "message": str(e)}
        # Don't dump huge lyrics blobs into every search result -- the model
        # can ask for them via getSongLyrics on the specific id it picks.
        trimmed = []
        for e in entries[:limit]:
            trimmed.append({
                "id": e.id,
                "title": e.title,
                "styles": e.styles[:200],
                "has_lyrics": bool(e.lyrics),
            })
        return {"result": "ok", "count": len(trimmed),
                "total": len(entries), "songs": trimmed}

    async def get_lyrics(self, clip_id: str) -> dict:
        """Fetch the lyrics for a specific clip from the bridge."""
        if not clip_id or not clip_id.strip():
            return {"result": "error", "code": "bad_id", "message": "Clip id required."}
        try:
            lyrics = await self._client.get_lyrics(clip_id.strip())
        except SunoError as e:
            return {"result": "error", "code": e.code, "message": e.message or str(e)}
        except httpx.HTTPError as e:
            return {"result": "error", "code": "bridge_unreachable", "message": str(e)}
        return {"result": "ok", "id": clip_id, "lyrics": lyrics}

    async def cover(self, source_id: str,
                    lyrics: str | None = None,
                    style: str | None = None) -> dict:
        """Cover an existing Suno song. lyrics/style are optional overrides."""
        if self._generating:
            return {"result": "error", "code": "already_generating",
                    "message": "A song is already being generated, wait for it."}
        cd = self.cooldown_remaining()
        if cd > 0:
            return {"result": "error", "code": "rate_limited",
                    "message": f"Please wait {cd:.0f}s before covering another song."}
        if not source_id or not source_id.strip():
            return {"result": "error", "code": "bad_id", "message": "Source song id required."}
        if lyrics and len(lyrics) > self._max_lyrics:
            return {"result": "error", "code": "lyrics_too_long",
                    "message": f"Lyrics exceed max length ({self._max_lyrics} chars)."}
        if style and len(style) > self._max_style:
            return {"result": "error", "code": "style_too_long",
                    "message": f"Style exceeds max length ({self._max_style} chars)."}

        # Stop any existing playback before kicking off the cover
        if self._player is not None:
            await asyncio.to_thread(self._player.stop)
            self._player = None
            if self._poll_task and not self._poll_task.done():
                self._poll_task.cancel()
                self._poll_task = None
            self._audio.set_external_music_active(False)

        self._last_request_at = time.time()
        self._generating = True
        self._generating_started_at = self._last_request_at
        self._generating_error = None
        self._audio.set_external_music_active(True)
        self._gen_task = asyncio.create_task(
            self._cover_and_play(source_id.strip(), lyrics=lyrics, style=style)
        )

        return {"result": "ok", "code": "submitted",
                "message": "Cover started. Audio will begin streaming in a few seconds. "
                           "Stay quiet until the music kicks in."}

    async def _poll_loop(self, clip_id: str):
        """Poll the bridge for title/status updates while we play."""
        saved = False
        try:
            # While playing
            while self._player is not None and self._player.is_playing:
                saved = await self._poll_once(clip_id, saved)
                await asyncio.sleep(5.0)
            # Mark external music inactive once playback ended
            self._audio.set_external_music_active(False)
            # Keep polling briefly so we still catch a late audio_url
            if not saved and self._save_enabled:
                deadline = time.monotonic() + 120.0
                while time.monotonic() < deadline and not saved:
                    saved = await self._poll_once(clip_id, saved, allow_save=True)
                    if saved:
                        break
                    await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            pass

    async def _poll_once(self, clip_id: str, already_saved: bool,
                         allow_save: bool = True) -> bool:
        try:
            clip = await self._client.get_clip(clip_id)
            if self._player is not None:
                s = self._player.state
                if clip.title:
                    s.title = clip.title
                if clip.status:
                    s.status = clip.status
                if clip.audio_url:
                    s.audio_url = clip.audio_url
            if clip.audio_url and self._save_enabled and allow_save and not already_saved:
                asyncio.create_task(self._download_song(clip))
                return True
        except Exception as e:
            logger.debug(f"Suno poll error: {e}")
        return already_saved

    async def _download_song(self, clip: SunoClip):
        """Download the finished mp3 to the local music dir for replay."""
        try:
            os.makedirs(self._save_dir, exist_ok=True)
            safe_title = _sanitize_filename(clip.title or "suno_song")
            filename = f"{safe_title}_{clip.id[:8]}.mp3"
            target = os.path.join(self._save_dir, filename)
            if os.path.exists(target):
                logger.info(f"Suno song already saved: {filename}")
                return
            tmp = target + ".part"
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as c:
                async with c.stream("GET", clip.audio_url) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        async for chunk in r.aiter_bytes(64 * 1024):
                            f.write(chunk)
            os.replace(tmp, target)
            logger.info(f"Saved Suno song to {target}")
        except Exception as e:
            logger.warning(f"Failed to save Suno song {clip.id}: {e}")
