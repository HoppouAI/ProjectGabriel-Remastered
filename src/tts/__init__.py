"""TTS providers for ProjectGabriel.

Each provider takes streaming text via feed_text() and produces 16-bit PCM
audio chunks via get_audio(). They share the same sentence-splitting +
pre-synthesis-overlap pipeline; the only thing that differs is the
network call to the actual TTS backend.

Providers:
    QwenTTSProvider     -- local Qwen3 TTS server (SSE streaming)
    HoppouTTSProvider   -- Hoppou AI cloud TTS (OpenAI-compatible)
    Chirp3HDTTSProvider -- Google Cloud Chirp 3: HD (gRPC streaming)
    TikTokTTSProvider   -- Weilbyte TikTok TTS proxy (free, no auth)
"""

from ._helpers import _strip_audio_tags, _strip_emojis
from .chirp3hd import Chirp3HDTTSProvider
from .hoppou import HoppouTTSProvider
from .qwen import QwenTTSProvider
from .tiktok import TikTokTTSProvider

__all__ = [
    "QwenTTSProvider",
    "HoppouTTSProvider",
    "Chirp3HDTTSProvider",
    "TikTokTTSProvider",
    "_strip_emojis",
    "_strip_audio_tags",
]
