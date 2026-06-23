"""Local backend for ProjectGabriel.

Drop-in replacement for the Gemini Live session, glued together from:
  - parakeet.cpp (NVIDIA NeMo Parakeet on ggml) for streaming / batch STT,
    auto downloaded as a prebuilt CPU or GPU (vulkan) library
  - Silero VAD (already used by gemini_live for client side speech detection)
  - LM Studio (any OpenAI-compatible /v1/chat/completions endpoint) for the LLM
  - one of the existing TTS providers (hoppou / chirp3_hd / tiktok /
    plugin) for voice. local mode REQUIRES one of those.

Pick the backend via top-level `backend:` in config.yml.
"""

from .session import LocalLiveSession

__all__ = ["LocalLiveSession"]
