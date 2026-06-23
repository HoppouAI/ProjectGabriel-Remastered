"""Download + cache the parakeet.cpp runtime and GGUF models.

The prebuilt parakeet.dll bundles (a vulkan build that runs on any GPU, and a
tiny cpu only build) are mirrored in the plugin resources repo so users never
have to build the library for their hardware. Models come straight from the
upstream HuggingFace GGUF repo.

Everything lands under data/parakeet/ which is gitignored:
  data/parakeet/runtime/<version>/<variant>/parakeet.dll
  data/parakeet/models/<name>-<quant>.gguf
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# bump when the hosted bundle changes so clients re-download cleanly.
RUNTIME_VERSION = "v0.3.2"

_RESOURCE_BASE = (
    "https://github.com/HoppouAI/ProjectGabriel-Plugin-Resources/raw/main/parakeet_cpp_asr"
)
_BUNDLES = {
    "vulkan": "parakeet-cpp-vulkan-win64.zip",
    "cpu": "parakeet-cpp-cpu-win64.zip",
}

_HF_MODEL_BASE = "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main"

_ROOT = Path(__file__).resolve().parents[2]
_CACHE = _ROOT / "data" / "parakeet"
_RUNTIME_DIR = _CACHE / "runtime" / RUNTIME_VERSION
_MODEL_DIR = _CACHE / "models"


def _download(url: str, dest: Path, desc: str) -> None:
    """Stream a download to dest, writing to a .part file then renaming so a
    half finished download is never mistaken for a complete one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info(f"downloading {desc} from {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        next_log = 0.10
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    frac = done / total
                    if frac >= next_log:
                        logger.info(f"  {desc}: {frac * 100:.0f}% ({done >> 20}/{total >> 20} MB)")
                        next_log += 0.10
    tmp.replace(dest)
    logger.info(f"  {desc}: done ({dest.stat().st_size >> 20} MB)")


def _has_vulkan() -> bool:
    """Cheap check for a usable Vulkan loader so we don't download the vulkan
    bundle on a machine that can't load it. parakeet's vulkan build links
    against the system vulkan-1.dll shipped with GPU drivers."""
    if os.name != "nt":
        # the bundles we host are win64 only; other platforms build their own.
        return False
    candidates = []
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    candidates.append(Path(sysroot) / "System32" / "vulkan-1.dll")
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d:
            candidates.append(Path(d) / "vulkan-1.dll")
    return any(p.is_file() for p in candidates)


def resolve_variants(compute: str) -> list[str]:
    """Turn a configured compute preference into an ordered list of bundle
    variants to try. 'auto' prefers vulkan when the loader is present, always
    keeping cpu as the fallback."""
    compute = (compute or "auto").lower()
    if compute == "cpu":
        return ["cpu"]
    if compute == "vulkan":
        return ["vulkan", "cpu"]
    # auto
    if _has_vulkan():
        return ["vulkan", "cpu"]
    logger.info("no vulkan loader found, using the cpu parakeet build")
    return ["cpu"]


def ensure_runtime(variant: str) -> Path:
    """Make sure the parakeet.dll for `variant` is downloaded and extracted.
    Returns the path to parakeet.dll."""
    if variant not in _BUNDLES:
        raise ValueError(f"unknown parakeet runtime variant '{variant}'")
    target = _RUNTIME_DIR / variant
    dll = target / "parakeet.dll"
    if dll.is_file():
        return dll

    target.mkdir(parents=True, exist_ok=True)
    url = f"{_RESOURCE_BASE}/{_BUNDLES[variant]}"
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / _BUNDLES[variant]
        _download(url, zip_path, f"parakeet runtime ({variant})")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)

    # the bundle is flat, but tolerate a nested folder just in case.
    if not dll.is_file():
        found = next(target.rglob("parakeet.dll"), None)
        if found is None:
            raise FileNotFoundError(f"parakeet.dll not found in {_BUNDLES[variant]}")
        if found != dll:
            shutil.copy2(found, dll)
    return dll


def ensure_model(model: str, quant: str) -> Path:
    """Make sure the GGUF for `model` at `quant` is downloaded. `model` may be
    a bare model name (downloaded from HuggingFace) or a path to a local
    .gguf, which is used as is. Returns the gguf path."""
    # explicit local file wins
    p = Path(model)
    if model.lower().endswith(".gguf") and p.is_file():
        return p

    filename = f"{model}-{quant}.gguf"
    dest = _MODEL_DIR / filename
    if dest.is_file():
        return dest

    url = f"{_HF_MODEL_BASE}/{filename}?download=true"
    _download(url, dest, f"parakeet model {filename}")
    return dest
