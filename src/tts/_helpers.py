"""Shared TTS helpers: regex-based emoji/audio-tag stripping + resampling."""

from __future__ import annotations

import re

import numpy as np

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


# Inline expressive tags like [curious] / [whispers]. Some upstream models
# emit these as performance hints; most TTS engines just speak them. Strip
# them at the per-sentence layer so the audio doesn't say "curious" out loud.
_AUDIO_TAG_RE = re.compile(r"\[(?:[A-Za-z][A-Za-z\s,'-]{0,40})\]")

# Tool-call shaped square brackets the model occasionally narrates, like
# [setMood:emotion=angry,level=6,reason=...]. The plain audio-tag regex above
# misses these (the :, = and digits break its char class), so nuke them here
# too, otherwise an external TTS provider reads the leaked call out loud. Mirror
# of _BRACKET_CALL in gemini_live/leak_filter.py (kept local to avoid a circular
# import back through the gemini_live package).
_TOOL_TAG_RE = re.compile(
    r"\[\s*(?=\w*(?:[A-Z]|_))[a-zA-Z_]\w{2,}[\s:=(]+[^\[\]]*?[a-zA-Z_]\w*\s*=\s*[^\[\]]{1,300}?\]",
    re.DOTALL,
)


def _strip_audio_tags(text: str) -> str:
    cleaned = _TOOL_TAG_RE.sub(" ", text)
    cleaned = _AUDIO_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resample of a float32 mono signal."""
    if src_sr == dst_sr:
        return audio
    ratio = dst_sr / src_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
