import pyaudio
import numpy as np
import logging
import os
import re
import time
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

logger = logging.getLogger(__name__)

MUSIC_FADEOUT_START = 2.0  # Start fading AI voice after this many seconds
MUSIC_FADEOUT_END = 10.0   # AI voice completely muted after this many seconds


class _PitchShifter:
    """WSOLA-based streaming pitch shifter. Resamples to shift pitch, then
    uses Waveform Similarity Overlap-Add to correct the speed back to
    original, preserving waveform shape for natural-sounding voice."""

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.win_size = 1024
        self.hop = self.win_size // 2
        self.search = 128
        self.window = np.hanning(self.win_size).astype(np.float64)
        self.in_buf = np.zeros(0, dtype=np.float64)
        self.out_buf = np.zeros(0, dtype=np.float64)
        self._prev_tail = np.zeros(self.hop, dtype=np.float64)

    def reset(self):
        self.in_buf = np.zeros(0, dtype=np.float64)
        self.out_buf = np.zeros(0, dtype=np.float64)
        self._prev_tail = np.zeros(self.hop, dtype=np.float64)

    def process(self, chunk: np.ndarray, semitones: float) -> np.ndarray:
        from scipy.signal import resample
        ratio = 2.0 ** (semitones / 12.0)
        n = len(chunk)
        self.in_buf = np.append(self.in_buf, chunk.astype(np.float64))
        process_size = max(self.win_size * 4, n * 2)
        if len(self.in_buf) < process_size:
            if len(self.out_buf) >= n:
                result = self.out_buf[:n].copy()
                self.out_buf = self.out_buf[n:]
                return result.astype(np.float32)
            return chunk

        block = self.in_buf[:process_size]
        self.in_buf = self.in_buf[process_size:]

        resampled_len = max(self.win_size, int(round(len(block) / ratio)))
        resampled = resample(block, resampled_len)
        stretched = self._wsola(resampled, len(block))
        self.out_buf = np.append(self.out_buf, stretched)

        if len(self.out_buf) >= n:
            result = self.out_buf[:n].copy()
            self.out_buf = self.out_buf[n:]
            return result.astype(np.float32)
        return chunk

    def _wsola(self, audio, target_len):
        n = len(audio)
        if n <= self.win_size:
            return np.resize(audio, target_len) if target_len > 0 else audio
        stretch = target_len / n
        hop_in = self.hop
        hop_out = max(1, int(round(hop_in * stretch)))

        out_len = target_len + self.win_size
        output = np.zeros(out_len, dtype=np.float64)
        norm = np.zeros(out_len, dtype=np.float64)

        pos_in = 0
        pos_out = 0

        while pos_out + self.win_size <= out_len:
            best = self._find_best(audio, pos_in, output, pos_out)
            if best + self.win_size > n:
                break
            grain = audio[best:best + self.win_size] * self.window
            output[pos_out:pos_out + self.win_size] += grain
            norm[pos_out:pos_out + self.win_size] += self.window
            pos_in = best + hop_in
            pos_out += hop_out

        mask = norm > 1e-8
        output[mask] /= norm[mask]
        self._prev_tail = output[max(0, min(target_len, len(output)) - self.hop):min(target_len, len(output))].copy()
        if len(self._prev_tail) < self.hop:
            self._prev_tail = np.pad(self._prev_tail, (0, self.hop - len(self._prev_tail)))
        return output[:target_len]

    def _find_best(self, audio, expected, output, out_pos):
        n = len(audio)
        lo = max(0, expected - self.search)
        hi = min(n - self.win_size, expected + self.search)
        if lo > hi:
            return max(0, min(expected, n - self.win_size))

        tpl_len = min(self.hop, out_pos)
        if tpl_len < 16 or out_pos == 0:
            return max(lo, min(expected, hi))

        template = output[out_pos - tpl_len:out_pos]
        seg = audio[lo:hi + tpl_len]
        if len(seg) < tpl_len:
            return max(lo, min(expected, hi))

        corr = np.correlate(seg, template, mode='valid')
        return lo + int(np.argmax(corr))


class AudioManager:
    def __init__(self, config):
        self.config = config
        self.pya = pyaudio.PyAudio()
        self.boost_level = 0
        self._music_start_time = None
        self._music_paused_at = None  # Track when paused for resume
        self._current_song_name = None
        self._current_song_duration = None
        self._current_volume = 50  # Track current volume (0-300)
        self._using_boosted_sound = False  # True if using boosted Sound instead of music
        self._boosted_sound_channel = None  # Channel for boosted playback
        self._lyrics = []  # Parsed SRT entries: [(start_sec, end_sec, text), ...]
        self._pygame_ready = False
        self._thinking_sound = None  # Loaded thinking sound
        self._thinking_channel = None  # Channel playing the thinking sound
        self._pitch_semitones = 0.0
        self._pitch_shifter = None
        self._setup_devices()

    def _setup_devices(self):
        if self.config.input_device is not None:
            self.input_device = self.config.input_device
        else:
            self.input_device = self.pya.get_default_input_device_info()["index"]
        if self.config.output_device is not None:
            self.output_device = self.config.output_device
        else:
            self.output_device = self.pya.get_default_output_device_info()["index"]

    def _setup_pygame(self):
        if self._pygame_ready:
            return
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        self._pygame_ready = True

    def open_input_stream(self):
        return self.pya.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.send_sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.config.chunk_size,
        )

    def open_output_stream(self):
        return self.pya.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.receive_sample_rate,
            output=True,
            output_device_index=self.output_device,
        )

    def is_music_playing(self) -> bool:
        """Check if music is currently playing."""
        if not self._pygame_ready:
            return False
        return pygame.mixer.music.get_busy() or pygame.mixer.get_busy()

    def get_voice_volume_multiplier(self) -> float:
        """Get the volume multiplier for AI voice based on music playing state.
        
        Returns:
            1.0 if no music playing
            1.0 → 0.0 fading during first 10s of music
            0.0 after 10s of music
        """
        if not self.is_music_playing():
            if self._music_paused_at is None:
                self._music_start_time = None
            return 1.0
        
        if self._music_start_time is None:
            return 1.0  # Music just started, wait for update
        
        elapsed = time.time() - self._music_start_time
        
        if elapsed < MUSIC_FADEOUT_START:
            return 1.0  # Full volume for first 2 seconds
        elif elapsed >= MUSIC_FADEOUT_END:
            return 0.0  # Muted after 10 seconds
        else:
            # Linear fade from 1.0 to 0.0 between 2s and 10s
            fade_progress = (elapsed - MUSIC_FADEOUT_START) / (MUSIC_FADEOUT_END - MUSIC_FADEOUT_START)
            return 1.0 - fade_progress

    def process_output_audio(self, data: bytes) -> bytes:
        """Process AI voice output with boost/distortion and music fade."""
        # Get volume multiplier based on music state
        voice_mult = self.get_voice_volume_multiplier()
        
        # If completely muted, return silence
        if voice_mult == 0.0:
            return b'\x00' * len(data)
        
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        
        # Apply boost if set
        if self.boost_level > 0:
            gain = 1.0 + (self.boost_level * 0.8)
            samples *= gain
            if self.boost_level >= 3:
                alpha = 0.15 + (self.boost_level * 0.03)
                boosted = np.zeros_like(samples)
                boosted[0] = samples[0]
                for i in range(1, len(samples)):
                    boosted[i] = alpha * samples[i] + (1 - alpha) * boosted[i - 1]
                samples = samples + boosted * (self.boost_level * 0.3)
            if self.boost_level >= 2:
                max_val = 32767.0
                samples = np.tanh(samples / max_val * (1 + self.boost_level * 0.2)) * max_val
        
        # Apply music fade multiplier
        if voice_mult < 1.0:
            samples *= voice_mult
        
        if self._pitch_semitones != 0.0 and self.config.get("audio", "pitch_shift", "enabled", default=False):
            if self._pitch_shifter is None:
                self._pitch_shifter = _PitchShifter()
            samples = self._pitch_shifter.process(samples, self._pitch_semitones)

        samples = np.clip(samples, -32767, 32767).astype(np.int16)
        return samples.tobytes()

    def set_pitch(self, semitones: float):
        max_st = self.config.get("audio", "pitch_shift", "max_semitones", default=12)
        self._pitch_semitones = max(-max_st, min(max_st, semitones))
        if self._pitch_shifter is not None:
            self._pitch_shifter.reset()
        logger.info(f"Voice pitch set to {self._pitch_semitones:+.1f} semitones")

    def get_pitch(self) -> float:
        return self._pitch_semitones

    def set_boost(self, level: int):
        self.boost_level = max(0, min(10, level))
        logger.info(f"Voice boost set to {self.boost_level}")

    def list_music(self) -> list[str]:
        from pathlib import Path
        music_dir = Path(self.config.music_dir)
        if not music_dir.exists():
            music_dir.mkdir(parents=True, exist_ok=True)
            return []
        exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
        files = []
        for filepath in music_dir.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in exts:
                rel_path = filepath.relative_to(music_dir)
                files.append(str(rel_path))
        return sorted(files)

    def play_music(self, filename: str, volume: int = 50) -> bool:
        self._setup_pygame()
        filepath = os.path.join(self.config.music_dir, filename)
        if not os.path.exists(filepath):
            return False
        try:
            self._current_volume = min(max(0, volume), 300)
            vol_float = self._current_volume / 100.0
            self._music_start_time = time.time()
            self._music_paused_at = None
            
            # Store song name without extension
            self._current_song_name = os.path.splitext(os.path.basename(filename))[0]
            
            # Try to get duration using mutagen
            self._current_song_duration = self._get_audio_duration(filepath)
            
            # Load matching SRT lyrics
            self._lyrics = self._load_srt(filename)
            
            if vol_float <= 1.0:
                self._using_boosted_sound = False
                self._boosted_sound_channel = None
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.set_volume(vol_float)
                pygame.mixer.music.play()
            else:
                self._using_boosted_sound = True
                sound = pygame.mixer.Sound(filepath)
                arr = pygame.sndarray.array(sound)
                arr = (arr.astype(np.float32) * vol_float).clip(-32767, 32767).astype(np.int16)
                boosted = pygame.sndarray.make_sound(arr)
                self._boosted_sound_channel = boosted.play()
            logger.info(f"Playing music: {filename} (duration: {self._current_song_duration:.1f}s)")
            return True
        except Exception as e:
            self._music_start_time = None
            self._music_paused_at = None
            self._current_song_name = None
            self._current_song_duration = None
            self._lyrics = []
            logger.error(f"Music playback failed: {e}")
            return False

    def _get_audio_duration(self, filepath: str) -> float:
        """Get audio file duration in seconds."""
        try:
            from mutagen import File
            audio = File(filepath)
            if audio is not None and audio.info is not None:
                return audio.info.length
        except Exception:
            pass
        
        # Fallback: try pygame.mixer.Sound for shorter files
        try:
            self._setup_pygame()
            sound = pygame.mixer.Sound(filepath)
            return sound.get_length()
        except Exception:
            pass
        
        return 0.0

    def _parse_srt_time(self, ts: str) -> float:
        """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
        if not m:
            return 0.0
        h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600 + mi * 60 + s + ms / 1000.0

    def _load_srt(self, music_filename: str) -> list:
        """Load matching .srt from sfx/music/srt/ for the given music file."""
        from pathlib import Path
        stem = Path(music_filename).stem
        srt_path = Path(self.config.music_dir) / "srt" / f"{stem}.srt"
        if not srt_path.exists():
            return []
        try:
            text = srt_path.read_text(encoding="utf-8")
            entries = []
            blocks = re.split(r"\n\s*\n", text.strip())
            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) < 3:
                    continue
                time_line = lines[1]
                arrow = re.split(r"\s*-->\s*", time_line)
                if len(arrow) != 2:
                    continue
                start = self._parse_srt_time(arrow[0])
                end = self._parse_srt_time(arrow[1])
                lyric = " ".join(lines[2:]).strip()
                if lyric:
                    entries.append((start, end, lyric))
            if entries:
                logger.info(f"Loaded {len(entries)} lyric entries from {srt_path.name}")
            return entries
        except Exception as e:
            logger.warning(f"Failed to load SRT {srt_path}: {e}")
            return []

    def get_current_lyric(self) -> str | None:
        """Get the lyric line for the current playback position, or None."""
        if not self._lyrics or self._music_start_time is None:
            return None
        if self._music_paused_at is not None:
            pos = self._music_paused_at - self._music_start_time
        else:
            pos = time.time() - self._music_start_time
        for start, end, text in self._lyrics:
            if start <= pos < end:
                return text
        return None

    def get_music_progress(self) -> dict | None:
        """Get current music playback status.
        
        Returns:
            dict with keys: name, position, duration, progress (0.0-1.0)
            or None if no music playing
        """
        if self._music_start_time is None:
            return None
        
        # Handle paused state
        if self._music_paused_at is not None:
            position = self._music_paused_at - self._music_start_time
        elif not self.is_music_playing():
            return None
        else:
            position = time.time() - self._music_start_time
        duration = self._current_song_duration or 0.0
        progress = (position / duration) if duration > 0 else 0.0
        
        return {
            "song_name": self._current_song_name or "Unknown",
            "position": position,
            "duration": duration,
            "progress": min(1.0, progress),
        }

    def stop_music(self):
        if not self._pygame_ready:
            return
        pygame.mixer.music.stop()
        pygame.mixer.stop()
        self._music_start_time = None
        self._music_paused_at = None
        self._current_song_name = None
        self._current_song_duration = None
        self._using_boosted_sound = False
        self._boosted_sound_channel = None
        self._lyrics = []

    def pause_music(self) -> bool:
        """Pause currently playing music. Returns False if nothing is playing."""
        if not self.is_music_playing():
            return False
        
        if self._using_boosted_sound:
            if self._boosted_sound_channel:
                self._boosted_sound_channel.pause()
        else:
            pygame.mixer.music.pause()
        
        # Track pause time for accurate progress
        if self._music_start_time:
            self._music_paused_at = time.time()
        
        logger.info("Music paused")
        return True

    def resume_music(self) -> bool:
        """Resume paused music. Returns False if nothing is paused."""
        if self._music_paused_at is None:
            return False
        
        if self._using_boosted_sound:
            if self._boosted_sound_channel:
                self._boosted_sound_channel.unpause()
        else:
            pygame.mixer.music.unpause()
        
        # Adjust start time to account for pause duration
        if self._music_start_time and self._music_paused_at:
            pause_duration = time.time() - self._music_paused_at
            self._music_start_time += pause_duration
        
        self._music_paused_at = None
        logger.info("Music resumed")
        return True

    def set_music_volume(self, volume: int) -> bool:
        """Set music volume while playing (0-300). Returns False if nothing playing."""
        if not self.is_music_playing() and self._music_paused_at is None:
            return False
        
        self._current_volume = min(max(0, volume), 300)
        vol_float = self._current_volume / 100.0
        
        # Note: For boosted mode (>100%), volume can only be changed at start
        # For normal mode, we can adjust in real-time
        if not self._using_boosted_sound:
            pygame.mixer.music.set_volume(min(vol_float, 1.0))
            logger.info(f"Music volume set to {self._current_volume}%")
            return True
        else:
            # For boosted mode, volume changes aren't fully supported mid-playback
            logger.info(f"Volume change requested to {self._current_volume}% (boosted mode - limited support)")
            return True

    def play_sfx_file(self, filepath: str, boost: int = 0) -> bool:
        self._setup_pygame()
        try:
            sound = pygame.mixer.Sound(filepath)
            if boost > 0:
                import numpy as np
                raw = pygame.sndarray.array(sound)
                samples = raw.astype(np.float32)
                gain = 1.0 + (boost * 0.8)
                samples *= gain
                if boost >= 3:
                    alpha = 0.15 + (boost * 0.03)
                    bass = np.zeros_like(samples)
                    bass[0] = samples[0]
                    for i in range(1, len(samples)):
                        bass[i] = alpha * samples[i] + (1 - alpha) * bass[i - 1]
                    samples = samples + bass * (boost * 0.3)
                if boost >= 2:
                    max_val = 32767.0
                    samples = np.tanh(samples / max_val * (1 + boost * 0.2)) * max_val
                samples = np.clip(samples, -32767, 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(samples)
                logger.info(f"SFX boost applied: level {boost}")
            sound.play()
            return True
        except Exception as e:
            logger.error(f"SFX playback failed: {e}")
            return False

    def stop_sfx(self):
        """Stop all currently playing sound effects."""
        if not self._pygame_ready:
            return
        pygame.mixer.stop()
        self._thinking_channel = None
        logger.info("All SFX playback stopped")

    def start_thinking_sound(self, source="thinking"):
        """Start playing the thinking sound effect with fade-in (looping).
        
        Args:
            source: "thinking" or "recall" -- checked against on_thinking/on_recall config.
        """
        if not self.config.thinking_sound_enabled:
            return
        if source == "thinking" and not self.config.thinking_sound_on_thinking:
            return
        if source == "recall" and not self.config.thinking_sound_on_recall:
            return
        self._setup_pygame()
        filepath = self.config.thinking_sound_file
        if not os.path.exists(filepath):
            logger.warning(f"Thinking sound file not found: {filepath}")
            return
        if self._thinking_channel and self._thinking_channel.get_busy():
            return  # Already playing
        try:
            if self._thinking_sound is None:
                self._thinking_sound = pygame.mixer.Sound(filepath)
            vol = min(max(0, self.config.thinking_sound_volume), 100) / 100.0
            self._thinking_sound.set_volume(vol)
            fade_in = self.config.thinking_sound_fade_in_ms
            self._thinking_channel = self._thinking_sound.play(loops=-1, fade_ms=fade_in)
            logger.debug("Thinking sound started")
        except Exception as e:
            logger.error(f"Failed to start thinking sound: {e}")

    def stop_thinking_sound(self):
        """Stop the thinking sound effect with fade-out."""
        if self._thinking_channel and self._thinking_channel.get_busy():
            fade_out = self.config.thinking_sound_fade_out_ms if self.config.thinking_sound_enabled else 800
            self._thinking_channel.fadeout(fade_out)
            self._thinking_channel = None
            logger.debug("Thinking sound stopping (fade out)")

    def cleanup(self):
        if self._pygame_ready:
            pygame.mixer.quit()
        self.pya.terminate()
