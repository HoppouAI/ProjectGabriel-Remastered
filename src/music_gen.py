import asyncio
import logging
import threading
import time
import pyaudio
import numpy as np
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Lyria outputs 48kHz stereo 16-bit PCM
LYRIA_SAMPLE_RATE = 48000
LYRIA_CHANNELS = 2


class MusicGenerator:
    """Lyria RealTime music generation via WebSocket streaming.

    Manages a persistent connection to Lyria RealTime, receives stereo 48kHz
    PCM audio, and plays it through a dedicated PyAudio output stream.
    Completely optional module - does not interfere with existing audio.

    The session runs entirely inside an async with block held by a background
    task (_session_task). Commands (play/pause/steer/stop) are communicated
    via an asyncio.Queue so the context manager stays alive.
    """

    MODEL = "models/lyria-realtime-exp"

    def __init__(self, config, audio_mgr):
        self._config = config
        self._audio = audio_mgr
        self._playing = False
        self._paused = False
        self._prompts: list[types.WeightedPrompt] = []
        self._gen_config = self._default_gen_config()
        self._volume = config.get("music_gen", "volume", default=80)
        self._fade_volume = 1.0  # Fade multiplier (1.0 = full, 0.0 = silent)
        self._fading = False
        self._lock = asyncio.Lock()
        self._pya = pyaudio.PyAudio()
        self._out_stream = None
        self._stream_lock = threading.Lock()
        # Background task holding the session context
        self._session_task = None
        # Command queue for the session task
        self._cmd_queue: asyncio.Queue = asyncio.Queue()
        # Result futures for commands
        self._pending_result: asyncio.Future | None = None
        # Session reference (set by session task, used for direct calls)
        self._session = None
        self._stop_event = asyncio.Event()
        self._start_time: float | None = None
        self._paused_elapsed: float = 0.0

    def _default_gen_config(self) -> types.LiveMusicGenerationConfig:
        cfg = self._config
        return types.LiveMusicGenerationConfig(
            bpm=cfg.get("music_gen", "default_bpm", default=120),
            temperature=cfg.get("music_gen", "temperature", default=1.1),
            guidance=cfg.get("music_gen", "guidance", default=4.0),
            density=cfg.get("music_gen", "density", default=None),
            brightness=cfg.get("music_gen", "brightness", default=None),
            mute_bass=cfg.get("music_gen", "mute_bass", default=True),
            mute_drums=cfg.get("music_gen", "mute_drums", default=True),
            music_generation_mode=types.MusicGenerationMode.QUALITY,
        )

    def _open_output_stream(self):
        if self._out_stream is not None:
            return
        out_dev = self._audio.output_device if hasattr(self._audio, "output_device") else None
        self._out_stream = self._pya.open(
            format=pyaudio.paInt16,
            channels=LYRIA_CHANNELS,
            rate=LYRIA_SAMPLE_RATE,
            output=True,
            output_device_index=out_dev,
        )

    def _close_output_stream(self):
        with self._stream_lock:
            if self._out_stream is not None:
                try:
                    self._out_stream.stop_stream()
                    self._out_stream.close()
                except Exception:
                    pass
                self._out_stream = None

    def _write_audio(self, pcm: bytes):
        """Thread-safe write to PyAudio output stream."""
        with self._stream_lock:
            if self._out_stream is not None:
                self._out_stream.write(pcm)

    async def _session_loop(self, prompts, gen_config):
        """Background task that holds the Lyria session open via async with."""
        client = genai.Client(
            api_key=self._config.api_key,
            http_options={"api_version": "v1alpha"},
        )
        try:
            async with client.aio.live.music.connect(model=self.MODEL) as session:
                self._session = session
                logger.info("Lyria RealTime session connected")

                # Initial setup
                await session.set_weighted_prompts(prompts=prompts)
                await session.set_music_generation_config(config=gen_config)
                await session.play()
                self._playing = True
                self._start_time = time.time()

                # Signal that start is complete
                if self._pending_result and not self._pending_result.done():
                    self._pending_result.set_result({"result": "ok", "message": "Now playing"})

                self._open_output_stream()

                # Run receive and command processing concurrently
                receive_task = asyncio.create_task(self._receive_audio(session))
                cmd_task = asyncio.create_task(self._process_commands(session))
                try:
                    done, pending = await asyncio.wait(
                        [receive_task, cmd_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
                except asyncio.CancelledError:
                    receive_task.cancel()
                    cmd_task.cancel()
                    try:
                        await receive_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    try:
                        await cmd_task
                    except (asyncio.CancelledError, Exception):
                        pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Lyria session error: {e}")
            if self._pending_result and not self._pending_result.done():
                self._pending_result.set_result({"result": "error", "message": str(e)})
        finally:
            self._session = None
            self._playing = False
            self._paused = False
            self._start_time = None
            self._paused_elapsed = 0.0
            await asyncio.to_thread(self._close_output_stream)
            self._prompts.clear()
            logger.info("Lyria RealTime session ended")

    async def _receive_audio(self, session):
        """Receive audio chunks from Lyria and write to PyAudio."""
        try:
            async for message in session.receive():
                if self._stop_event.is_set():
                    return
                if self._paused:
                    continue
                try:
                    audio_data = message.server_content.audio_chunks[0].data
                except (AttributeError, IndexError):
                    continue
                pcm = self._apply_volume(audio_data)
                if pcm and len(pcm) > 0:
                    try:
                        await asyncio.to_thread(self._write_audio, pcm)
                    except OSError:
                        return
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Lyria receive error: {e}")

    async def _process_commands(self, session):
        """Process steer/pause/resume/stop commands from the queue."""
        try:
            while not self._stop_event.is_set():
                try:
                    cmd = await asyncio.wait_for(self._cmd_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                action = cmd["action"]
                future = cmd.get("future")
                try:
                    if action == "stop":
                        await session.stop()
                        self._stop_event.set()
                        if future and not future.done():
                            future.set_result({"result": "ok", "message": "Stopped playing"})
                        return
                    elif action == "pause":
                        await session.pause()
                        self._paused_elapsed += time.time() - self._start_time
                        self._paused = True
                        if future and not future.done():
                            future.set_result({"result": "ok", "message": "Paused"})
                    elif action == "resume":
                        await session.play()
                        self._start_time = time.time()
                        self._paused = False
                        if future and not future.done():
                            future.set_result({"result": "ok", "message": "Resumed"})
                    elif action == "set_prompts":
                        await session.set_weighted_prompts(prompts=cmd["prompts"])
                        if future and not future.done():
                            future.set_result({"result": "ok"})
                    elif action == "set_config":
                        await session.set_music_generation_config(config=cmd["config"])
                        if cmd.get("reset_context"):
                            await session.reset_context()
                        if future and not future.done():
                            future.set_result({"result": "ok"})
                except Exception as e:
                    if future and not future.done():
                        future.set_result({"result": "error", "message": str(e)})
        except asyncio.CancelledError:
            pass

    def _apply_volume(self, raw_bytes: bytes) -> bytes:
        """Apply volume scaling and fade to raw 16-bit PCM audio."""
        try:
            vol_scale = (self._volume / 100.0) * self._fade_volume
            if vol_scale == 1.0:
                return raw_bytes
            if vol_scale == 0.0:
                return b'\x00' * len(raw_bytes)
            samples = np.frombuffer(raw_bytes, dtype=np.int16)
            scaled = (samples.astype(np.float32) * vol_scale).clip(-32768, 32767).astype(np.int16)
            return scaled.tobytes()
        except Exception as e:
            logger.error(f"Volume processing error: {e}")
            return raw_bytes

    async def _send_cmd(self, cmd: dict) -> dict:
        """Send a command to the session task and wait for result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        cmd["future"] = future
        await self._cmd_queue.put(cmd)
        return await future

    # ── Public API ──

    @property
    def is_playing(self) -> bool:
        return self._playing and not self._paused

    @property
    def is_paused(self) -> bool:
        return self._playing and self._paused

    @property
    def is_active(self) -> bool:
        return self._playing

    @property
    def current_prompts(self) -> list[dict]:
        return [{"text": p.text, "weight": p.weight} for p in self._prompts]

    @property
    def elapsed(self) -> float:
        """Seconds since playback started (excludes paused time)."""
        if self._start_time is None:
            return 0.0
        if self._paused:
            return self._paused_elapsed
        return self._paused_elapsed + (time.time() - self._start_time)

    async def start(self, prompts: list[dict], bpm: int | None = None,
                    scale: str | None = None) -> dict:
        """Start generating music with the given prompts."""
        async with self._lock:
            if self._playing:
                await self._stop_internal()

            # Build prompts
            self._prompts = [
                types.WeightedPrompt(text=p["text"], weight=p.get("weight", 1.0))
                for p in prompts
            ]

            # Build config
            gen_config = self._default_gen_config()
            if bpm is not None:
                gen_config.bpm = max(60, min(200, bpm))
            if scale is not None:
                try:
                    gen_config.scale = types.Scale[scale]
                except (KeyError, ValueError):
                    pass
            self._gen_config = gen_config

            # Setup result future and start session task
            loop = asyncio.get_running_loop()
            self._pending_result = loop.create_future()
            self._stop_event.clear()
            self._session_task = asyncio.create_task(
                self._session_loop(list(self._prompts), gen_config)
            )

            # Wait for session to start (or fail)
            try:
                result = await asyncio.wait_for(self._pending_result, timeout=15.0)
            except asyncio.TimeoutError:
                result = {"result": "error", "message": "Connection timed out"}
                if self._session_task:
                    self._session_task.cancel()
            self._pending_result = None

            logger.info(f"Lyria RealTime started: {[p['text'] for p in prompts]}")
            return result

    async def _fade_out(self, duration: float = 1.5):
        """Gradually fade volume to zero over the given duration."""
        steps = 15
        interval = duration / steps
        for i in range(steps):
            self._fade_volume = 1.0 - ((i + 1) / steps)
            await asyncio.sleep(interval)
        self._fade_volume = 0.0

    async def stop(self) -> dict:
        async with self._lock:
            return await self._stop_internal()

    async def _stop_internal(self) -> dict:
        if not self._playing and not self._session_task:
            return {"result": "ok", "message": "Nothing playing"}

        # Fade out before stopping (unless already fading or not playing audio)
        if self._playing and not self._fading:
            self._fading = True
            await self._fade_out()

        # Signal stop so loops exit cleanly
        self._stop_event.set()

        # Send stop command through queue if session is active
        if self._playing and self._session:
            try:
                result = await asyncio.wait_for(
                    self._send_cmd({"action": "stop"}), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception):
                pass

        # Wait briefly for receive loop to finish writing audio
        await asyncio.sleep(0.1)

        # Cancel the session task
        if self._session_task:
            self._session_task.cancel()
            try:
                await self._session_task
            except (asyncio.CancelledError, Exception):
                pass
            self._session_task = None

        self._playing = False
        self._paused = False
        self._fading = False
        self._fade_volume = 1.0
        self._start_time = None
        self._paused_elapsed = 0.0
        self._prompts.clear()
        # Output stream is closed in _session_loop finally block
        logger.info("Lyria RealTime stopped")
        return {"result": "ok", "message": "Stopped playing"}

    async def pause(self) -> dict:
        if not self._playing:
            return {"result": "error", "message": "Nothing playing"}
        if self._paused:
            return {"result": "ok", "message": "Already paused"}
        return await self._send_cmd({"action": "pause"})

    async def resume(self) -> dict:
        if not self._playing:
            return {"result": "error", "message": "Nothing playing"}
        if not self._paused:
            return {"result": "ok", "message": "Already playing"}
        return await self._send_cmd({"action": "resume"})

    async def steer(self, prompts: list[dict] | None = None,
                    bpm: int | None = None,
                    scale: str | None = None,
                    density: float | None = None,
                    brightness: float | None = None,
                    guidance: float | None = None,
                    mute_bass: bool | None = None,
                    mute_drums: bool | None = None,
                    mode: str | None = None) -> dict:
        """Steer the active music generation in real-time."""
        if not self._playing or not self._session:
            return {"result": "error", "message": "Not currently playing anything"}

        needs_reset = False
        changes = []

        # Update prompts if provided
        if prompts is not None:
            self._prompts = [
                types.WeightedPrompt(text=p["text"], weight=p.get("weight", 1.0))
                for p in prompts
            ]
            await self._send_cmd({"action": "set_prompts", "prompts": list(self._prompts)})
            changes.append(f"prompts={[p['text'] for p in prompts]}")

        # Update config if any parameter changed
        config_changed = any(x is not None for x in [bpm, scale, density, brightness, guidance, mute_bass, mute_drums, mode])
        if config_changed:
            cfg = self._gen_config
            if bpm is not None:
                cfg.bpm = max(60, min(200, bpm))
                needs_reset = True
                changes.append(f"bpm={bpm}")
            if scale is not None:
                try:
                    cfg.scale = types.Scale[scale]
                    needs_reset = True
                    changes.append(f"scale={scale}")
                except (KeyError, ValueError):
                    pass
            if density is not None:
                cfg.density = max(0.0, min(1.0, density))
                changes.append(f"density={density}")
            if brightness is not None:
                cfg.brightness = max(0.0, min(1.0, brightness))
                changes.append(f"brightness={brightness}")
            if guidance is not None:
                cfg.guidance = max(0.0, min(6.0, guidance))
                changes.append(f"guidance={guidance}")
            if mute_bass is not None:
                cfg.mute_bass = mute_bass
                changes.append(f"mute_bass={mute_bass}")
            if mute_drums is not None:
                cfg.mute_drums = mute_drums
                changes.append(f"mute_drums={mute_drums}")
            if mode is not None:
                mode_map = {
                    "quality": types.MusicGenerationMode.QUALITY,
                    "diversity": types.MusicGenerationMode.DIVERSITY,
                    "vocalization": types.MusicGenerationMode.VOCALIZATION,
                }
                if mode.lower() in mode_map:
                    cfg.music_generation_mode = mode_map[mode.lower()]
                    changes.append(f"mode={mode}")
            self._gen_config = cfg
            await self._send_cmd({
                "action": "set_config",
                "config": cfg,
                "reset_context": needs_reset,
            })

        logger.info(f"Lyria steered: {', '.join(changes)}")
        return {"result": "ok", "message": f"Updated: {', '.join(changes)}", "reset_context": needs_reset}

    async def set_volume(self, volume: int) -> dict:
        self._volume = max(0, min(200, volume))
        return {"result": "ok", "volume": self._volume}
