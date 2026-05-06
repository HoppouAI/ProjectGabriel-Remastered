"""Quick suno-bridge latency probe.

Measures:
  1. Time from POST /api/v1/songs to bridge response (with stream_url).
  2. Time from stream_url known to ffmpeg producing the first PCM byte
     (with the same flags our SunoPlayer uses).

Run with `python scripts/suno_latency_probe.py`.

Costs ONE generation credit per run. Don't loop this.
"""

import json
import subprocess
import time
import urllib.request

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG = get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"

BRIDGE = "http://127.0.0.1:8787"

LYRICS = (
    "[Verse]\n"
    "timing probe just rolling through\n"
    "checking how long suno needs to brew\n"
    "[Chorus]\n"
    "tick tock tick tock\n"
    "watch the latency clock\n"
)
STYLE = "lo-fi acoustic, gentle finger picked guitar, soft male vocals"


def post_song():
    body = json.dumps({"lyrics": LYRICS, "style": STYLE}).encode()
    req = urllib.request.Request(
        f"{BRIDGE}/api/v1/songs?timeout_ms=120000",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=140) as r:
        data = json.loads(r.read().decode())
    t1 = time.perf_counter()
    return data, t1 - t0


def time_first_pcm_byte(stream_url):
    cmd = [
        FFMPEG, "-loglevel", "error", "-nostdin",
        "-probesize", "32", "-analyzeduration", "0",
        "-fflags", "+nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", stream_url,
        "-f", "s16le", "-ar", "44100", "-ac", "2",
        "pipe:1",
    ]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    first = p.stdout.read(1)
    t1 = time.perf_counter()
    try:
        p.terminate()
    except Exception:
        pass
    return t1 - t0, len(first) > 0


def time_first_pcm_byte_default(stream_url):
    """Same but WITHOUT our custom low-latency flags (apples to apples vs default ffmpeg)."""
    cmd = [
        FFMPEG, "-loglevel", "error", "-nostdin",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", stream_url,
        "-f", "s16le", "-ar", "44100", "-ac", "2",
        "pipe:1",
    ]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    first = p.stdout.read(1)
    t1 = time.perf_counter()
    try:
        p.terminate()
    except Exception:
        pass
    return t1 - t0, len(first) > 0


def main():
    print(f"ffmpeg: {FFMPEG}")
    print("posting to /api/v1/songs ...")
    data, post_secs = post_song()
    songs = data.get("songs", [])
    print(f"bridge POST returned in {post_secs:.2f}s with {len(songs)} clip(s)")
    if not songs:
        print("no clips returned, aborting")
        return
    s = songs[0]
    print(f"  first clip id     : {s.get('id')}")
    print(f"  status            : {s.get('status')}")
    print(f"  stream_url        : {s.get('stream_url')[:80]}...")
    stream_url = s.get("stream_url")

    # Measure default-flags ffmpeg first-byte time
    print("\nmeasuring DEFAULT ffmpeg time-to-first-PCM-byte ...")
    secs_default, ok_default = time_first_pcm_byte_default(stream_url)
    print(f"  default ffmpeg    : {secs_default:.2f}s (got byte: {ok_default})")

    # Then measure our optimized flags (use the OTHER clip if present so we
    # don't double-pull the same stream which suno may have rate limits on)
    if len(songs) > 1:
        stream_url = songs[1].get("stream_url")
        print(f"\nusing alternate clip for low-latency test: {songs[1].get('id')}")
    print("\nmeasuring OPTIMIZED ffmpeg time-to-first-PCM-byte ...")
    secs_opt, ok_opt = time_first_pcm_byte(stream_url)
    print(f"  optimized ffmpeg  : {secs_opt:.2f}s (got byte: {ok_opt})")

    print("\n--- summary ---")
    print(f"bridge POST           : {post_secs:.2f}s")
    print(f"ffmpeg default        : {secs_default:.2f}s -> first audio at ~{post_secs + secs_default:.2f}s")
    print(f"ffmpeg optimized      : {secs_opt:.2f}s -> first audio at ~{post_secs + secs_opt:.2f}s")
    saved = secs_default - secs_opt
    print(f"flags saved           : {saved:+.2f}s")


if __name__ == "__main__":
    main()
