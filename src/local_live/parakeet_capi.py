"""ctypes binding for parakeet.cpp's flat C-API (parakeet_capi.h).

parakeet.cpp ships a single statically linked `parakeet.dll` exposing an
extern "C" surface meant for dlopen (the same one LocalAI drives). We load it
once, bind the handful of entry points we use, and wrap the opaque context /
stream handles in small Python objects.

Two transcription paths:
  - offline: parakeet_capi_transcribe_pcm_lang on a finished audio segment.
  - streaming: parakeet_capi_stream_begin_lang then repeated stream_feed, which
    returns newly finalized text plus an event bitmask. EOU = the speaker
    yielded the turn (respond), EOB = a backchannel like "uh huh" (keep
    listening).

All returned strings are malloc'd UTF-8 owned by us and freed with
parakeet_capi_free_string, so we take the raw pointer (c_void_p) instead of
letting ctypes coerce to bytes, which would leak the allocation.
"""

from __future__ import annotations

import ctypes
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# event bits returned in stream_feed's *eou_out (ABI v5). in v4 the flag was a
# plain any-event 0/1, which still lines up with EOU being bit 0.
PARAKEET_EVENT_EOU = 1
PARAKEET_EVENT_EOB = 2


class ParakeetError(RuntimeError):
    pass


def load_library(dll_path: str) -> ctypes.CDLL:
    """Load parakeet.dll and bind its C-API. Raises OSError if the dll or one
    of its dependencies (eg system vulkan-1.dll for the vulkan build) can't be
    resolved, which the caller uses to fall back to the cpu bundle."""
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    # let the loader find sibling dlls and the dll's own import deps
    if hasattr(os, "add_dll_directory") and os.path.isdir(dll_dir):
        try:
            os.add_dll_directory(dll_dir)
        except OSError as e:
            logger.debug(f"add_dll_directory({dll_dir}) failed: {e}")
    lib = ctypes.CDLL(dll_path)
    _bind(lib)
    return lib


def _bind(lib: ctypes.CDLL) -> None:
    c_ptr = ctypes.c_void_p
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int)

    lib.parakeet_capi_abi_version.restype = ctypes.c_int
    lib.parakeet_capi_abi_version.argtypes = []

    lib.parakeet_capi_load.restype = c_ptr
    lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]

    lib.parakeet_capi_free.restype = None
    lib.parakeet_capi_free.argtypes = [c_ptr]

    lib.parakeet_capi_transcribe_pcm_lang.restype = c_ptr
    lib.parakeet_capi_transcribe_pcm_lang.argtypes = [
        c_ptr, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
    ]

    lib.parakeet_capi_stream_begin_lang.restype = c_ptr
    lib.parakeet_capi_stream_begin_lang.argtypes = [c_ptr, ctypes.c_char_p]

    lib.parakeet_capi_stream_feed.restype = c_ptr
    lib.parakeet_capi_stream_feed.argtypes = [c_ptr, f32p, ctypes.c_int, i32p]

    lib.parakeet_capi_stream_finalize.restype = c_ptr
    lib.parakeet_capi_stream_finalize.argtypes = [c_ptr]

    lib.parakeet_capi_stream_free.restype = None
    lib.parakeet_capi_stream_free.argtypes = [c_ptr]

    lib.parakeet_capi_free_string.restype = None
    lib.parakeet_capi_free_string.argtypes = [c_ptr]

    lib.parakeet_capi_last_error.restype = ctypes.c_char_p
    lib.parakeet_capi_last_error.argtypes = [c_ptr]


def _take_string(lib: ctypes.CDLL, ptr) -> str:
    """Copy a malloc'd C string into a Python str and free the original. A NULL
    pointer becomes "" (callers check for NULL separately when it means error)."""
    if not ptr:
        return ""
    try:
        raw = ctypes.cast(ptr, ctypes.c_char_p).value or b""
        return raw.decode("utf-8", "replace")
    finally:
        lib.parakeet_capi_free_string(ptr)


def _as_f32_ptr(samples: np.ndarray):
    """Return a contiguous float32 view of samples plus a c_float pointer to it.
    The array is returned too so the caller keeps it alive for the call."""
    arr = np.ascontiguousarray(samples, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class ParakeetStream:
    """A cache-aware streaming session over a loaded model."""

    def __init__(self, lib: ctypes.CDLL, handle):
        self._lib = lib
        self._s = handle

    def feed(self, samples: np.ndarray) -> tuple[str, int]:
        """Feed a block of 16 kHz mono float PCM. Returns (newly finalized text,
        event bitmask). Empty text means nothing finalized this call."""
        if not self._s:
            raise ParakeetError("stream already freed")
        arr, ptr = _as_f32_ptr(samples)
        ev = ctypes.c_int(0)
        out = self._lib.parakeet_capi_stream_feed(self._s, ptr, arr.size, ctypes.byref(ev))
        if not out:
            raise ParakeetError("parakeet_capi_stream_feed returned NULL")
        return _take_string(self._lib, out), ev.value

    def finalize(self) -> str:
        """Flush the end of stream tail and return any last finalized text."""
        if not self._s:
            return ""
        out = self._lib.parakeet_capi_stream_finalize(self._s)
        return _take_string(self._lib, out)

    def free(self) -> None:
        if self._s:
            self._lib.parakeet_capi_stream_free(self._s)
            self._s = None


class ParakeetContext:
    """A loaded GGUF model. Reused across transcribe / stream calls. Not safe to
    call concurrently from multiple threads, drive it from one worker."""

    def __init__(self, lib: ctypes.CDLL, gguf_path: str):
        self._lib = lib
        self._ctx = lib.parakeet_capi_load(gguf_path.encode("utf-8"))
        if not self._ctx:
            raise ParakeetError(f"parakeet_capi_load failed for {gguf_path}")

    @property
    def abi_version(self) -> int:
        try:
            return int(self._lib.parakeet_capi_abi_version())
        except Exception:
            return 0

    def last_error(self) -> str:
        if not self._ctx:
            return ""
        raw = self._lib.parakeet_capi_last_error(self._ctx)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", "replace")
        return raw or ""

    def transcribe(self, samples: np.ndarray, decoder: int = 0, lang: str = "auto") -> str:
        """Transcribe a finished 16 kHz mono float PCM segment. decoder:
        0=default, 1=ctc, 2=tdt/rnnt."""
        if not self._ctx:
            raise ParakeetError("context already freed")
        arr, ptr = _as_f32_ptr(samples)
        out = self._lib.parakeet_capi_transcribe_pcm_lang(
            self._ctx, ptr, arr.size, 16000, int(decoder),
            (lang or "auto").encode("utf-8"),
        )
        if not out:
            raise ParakeetError(self.last_error() or "transcribe_pcm_lang returned NULL")
        return _take_string(self._lib, out)

    def begin_stream(self, lang: str = "auto") -> ParakeetStream:
        """Start a streaming session. Raises if the model is not a cache-aware
        streaming model."""
        if not self._ctx:
            raise ParakeetError("context already freed")
        handle = self._lib.parakeet_capi_stream_begin_lang(
            self._ctx, (lang or "auto").encode("utf-8"),
        )
        if not handle:
            raise ParakeetError(
                self.last_error() or "stream_begin failed (model is not streaming?)"
            )
        return ParakeetStream(self._lib, handle)

    def free(self) -> None:
        if self._ctx:
            self._lib.parakeet_capi_free(self._ctx)
            self._ctx = None
