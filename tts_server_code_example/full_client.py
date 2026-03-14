#!/usr/bin/env python3
"""
Full-featured Python client for the Faster Qwen3-TTS demo API server.

Wraps every endpoint in a clean TTSClient class, usable as a library or CLI.

Usage as library:
    from full_client import TTSClient

    client = TTSClient("http://localhost:7860")
    client.load_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    # Voice clone (non-streaming)
    audio, sr, metrics = client.generate_voice_clone("Hello!", ref_audio="ref.wav")

    # Voice clone (streaming — real-time playback)
    for chunk_audio, sr, meta in client.stream_voice_clone("Hello!", ref_audio="ref.wav"):
        play(chunk_audio, sr)

    # Custom voice
    audio, sr, _ = client.generate_custom_voice("Hello!", speaker="Chelsie")

    # Voice design
    audio, sr, _ = client.generate_voice_design("Hello!", instruct="A deep male voice")

    # Transcribe reference audio
    text = client.transcribe("ref.wav")

Usage as CLI:
    # Show server status
    python demo/clients/full_client.py status

    # Load a model
    python demo/clients/full_client.py load --model Qwen/Qwen3-TTS-12Hz-1.7B-Base

    # Voice clone (streaming)
    python demo/clients/full_client.py clone --text "Hello!" --ref-audio ref.wav --stream

    # Custom voice
    python demo/clients/full_client.py custom --text "Hello!" --speaker Chelsie

    # Voice design
    python demo/clients/full_client.py design --text "Hello!" --instruct "A calm narrator"

    # Transcribe
    python demo/clients/full_client.py transcribe --audio ref.wav
"""

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import dataclass

import numpy as np
import requests
import soundfile as sf


@dataclass
class ServerStatus:
    loaded: bool
    model: str | None
    loading: bool
    available_models: list[str]
    model_type: str | None
    speakers: list[str]
    transcription_available: bool
    preset_refs: list[dict]
    queue_depth: int
    cached_models: list[str]


class TTSClient:
    """Client for the Faster Qwen3-TTS demo HTTP API."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    # ── Server management ────────────────────────────────────────────────

    def status(self) -> ServerStatus:
        """Get server status including loaded model, speakers, and queue depth."""
        resp = requests.get(f"{self.base_url}/status")
        resp.raise_for_status()
        return ServerStatus(**resp.json())

    def load_model(self, model_id: str) -> dict:
        """Load a model on the server. Blocks until ready."""
        resp = requests.post(f"{self.base_url}/load", data={"model_id": model_id})
        resp.raise_for_status()
        return resp.json()

    def ensure_model(self, model_id: str) -> None:
        """Load model only if not already active."""
        st = self.status()
        if st.model != model_id:
            print(f"Loading model: {model_id}...")
            self.load_model(model_id)

    # ── Transcription ────────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a WAV file using the server's nano-parakeet model."""
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/transcribe",
                files={"audio": (audio_path, f, "audio/wav")},
            )
        resp.raise_for_status()
        return resp.json()["text"]

    # ── Preset references ────────────────────────────────────────────────

    def get_preset_ref(self, preset_id: str) -> dict:
        """Fetch a preset reference voice (audio + transcript)."""
        resp = requests.get(f"{self.base_url}/preset_ref/{preset_id}")
        resp.raise_for_status()
        return resp.json()

    # ── Non-streaming generation ─────────────────────────────────────────

    def _generate(self, **data) -> tuple[np.ndarray, int, dict]:
        files = {}
        ref_audio_path = data.pop("ref_audio_file", None)
        if ref_audio_path:
            files["ref_audio"] = open(ref_audio_path, "rb")

        # Stringify bools and numbers for multipart form
        form = {k: str(v) if isinstance(v, (bool, int, float)) else v
                for k, v in data.items() if v is not None and v != ""}

        try:
            resp = requests.post(f"{self.base_url}/generate", data=form,
                                 files=files if files else None)
            resp.raise_for_status()
        finally:
            for f in files.values():
                f.close()

        result = resp.json()
        audio_bytes = base64.b64decode(result["audio_b64"])
        audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return audio_np, sr, result.get("metrics", {})

    def generate_voice_clone(
        self,
        text: str,
        ref_audio: str | None = None,
        ref_text: str = "",
        ref_preset: str = "",
        language: str = "English",
        xvec_only: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> tuple[np.ndarray, int, dict]:
        return self._generate(
            text=text, language=language, mode="voice_clone",
            ref_text=ref_text, ref_preset=ref_preset,
            xvec_only=xvec_only, temperature=temperature,
            top_k=top_k, repetition_penalty=repetition_penalty,
            ref_audio_file=ref_audio,
        )

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = "English",
        instruct: str = "",
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> tuple[np.ndarray, int, dict]:
        return self._generate(
            text=text, language=language, mode="custom",
            speaker=speaker, instruct=instruct,
            temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "English",
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> tuple[np.ndarray, int, dict]:
        return self._generate(
            text=text, language=language, mode="voice_design",
            instruct=instruct, temperature=temperature,
            top_k=top_k, repetition_penalty=repetition_penalty,
        )

    # ── Streaming generation ─────────────────────────────────────────────

    def _stream(self, **data):
        files = {}
        ref_audio_path = data.pop("ref_audio_file", None)
        if ref_audio_path:
            files["ref_audio"] = open(ref_audio_path, "rb")

        form = {k: str(v) if isinstance(v, (bool, int, float)) else v
                for k, v in data.items() if v is not None and v != ""}

        try:
            with requests.post(f"{self.base_url}/generate/stream", data=form,
                               files=files if files else None, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])

                    if payload["type"] == "queued":
                        continue
                    if payload["type"] == "error":
                        raise RuntimeError(payload["message"])
                    if payload["type"] == "done":
                        yield None, None, payload
                        return
                    if payload["type"] == "chunk":
                        audio_bytes = base64.b64decode(payload["audio_b64"])
                        audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                        yield audio_np, sr, payload
        finally:
            for f in files.values():
                f.close()

    def stream_voice_clone(
        self,
        text: str,
        ref_audio: str | None = None,
        ref_text: str = "",
        ref_preset: str = "",
        language: str = "English",
        xvec_only: bool = True,
        chunk_size: int = 8,
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ):
        yield from self._stream(
            text=text, language=language, mode="voice_clone",
            ref_text=ref_text, ref_preset=ref_preset,
            xvec_only=xvec_only, chunk_size=chunk_size,
            temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
            ref_audio_file=ref_audio,
        )

    def stream_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = "English",
        instruct: str = "",
        chunk_size: int = 8,
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ):
        yield from self._stream(
            text=text, language=language, mode="custom",
            speaker=speaker, instruct=instruct,
            chunk_size=chunk_size, temperature=temperature,
            top_k=top_k, repetition_penalty=repetition_penalty,
        )

    def stream_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "English",
        chunk_size: int = 8,
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ):
        yield from self._stream(
            text=text, language=language, mode="voice_design",
            instruct=instruct, chunk_size=chunk_size,
            temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
        )


# ── Helpers ──────────────────────────────────────────────────────────────

def _play_or_save(audio: np.ndarray, sr: int, output: str | None):
    if output:
        sf.write(output, audio, sr)
        print(f"Saved to {output}")
    else:
        try:
            import sounddevice as sd
            print("Playing audio...")
            sd.play(audio, sr)
            sd.wait()
        except ImportError:
            sf.write("output.wav", audio, sr)
            print("sounddevice not installed — saved to output.wav")


def _collect_stream(gen, output: str | None):
    all_audio, sample_rate = [], None
    for audio_np, sr, meta in gen:
        if audio_np is None:
            print(f"\n  Done! TTFA={meta['ttfa_ms']}ms | RTF={meta['rtf']}x | "
                  f"Audio={meta['total_audio_s']:.2f}s | Total={meta['total_ms']}ms")
            break
        all_audio.append(audio_np)
        sample_rate = sr
        dur_ms = len(audio_np) / sr * 1000
        print(f"  Chunk: {dur_ms:.0f}ms audio | RTF={meta.get('rtf', 0)}x", end="\r")
    if all_audio:
        _play_or_save(np.concatenate(all_audio), sample_rate, output)
    else:
        print("No audio received.", file=sys.stderr)


# ── CLI ──────────────────────────────────────────────────────────────────

def _add_common_args(p):
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--output", "-o", help="Save to WAV file")


def main():
    parser = argparse.ArgumentParser(description="Faster Qwen3-TTS API client")
    parser.add_argument("--base-url", default="http://localhost:7860")
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show server status")

    # load
    p_load = sub.add_parser("load", help="Load a model")
    p_load.add_argument("--model", required=True)

    # transcribe
    p_trans = sub.add_parser("transcribe", help="Transcribe audio")
    p_trans.add_argument("--audio", required=True, help="Path to WAV file")

    # clone
    p_clone = sub.add_parser("clone", help="Voice clone TTS")
    p_clone.add_argument("--text", required=True)
    p_clone.add_argument("--ref-audio", help="Reference WAV path")
    p_clone.add_argument("--ref-preset", default="", help="Preset ID")
    p_clone.add_argument("--ref-text", default="", help="Reference transcript")
    p_clone.add_argument("--auto-transcribe", action="store_true")
    p_clone.add_argument("--language", default="English")
    p_clone.add_argument("--full-clone", action="store_true",
                         help="Full clone instead of x-vector only")
    p_clone.add_argument("--stream", action="store_true")
    p_clone.add_argument("--chunk-size", type=int, default=8)
    p_clone.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    _add_common_args(p_clone)

    # custom
    p_custom = sub.add_parser("custom", help="Custom voice TTS")
    p_custom.add_argument("--text", required=True)
    p_custom.add_argument("--speaker", required=True)
    p_custom.add_argument("--instruct", default="")
    p_custom.add_argument("--language", default="English")
    p_custom.add_argument("--stream", action="store_true")
    p_custom.add_argument("--chunk-size", type=int, default=8)
    p_custom.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    _add_common_args(p_custom)

    # design
    p_design = sub.add_parser("design", help="Voice design TTS")
    p_design.add_argument("--text", required=True)
    p_design.add_argument("--instruct", required=True)
    p_design.add_argument("--language", default="English")
    p_design.add_argument("--stream", action="store_true")
    p_design.add_argument("--chunk-size", type=int, default=8)
    p_design.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    _add_common_args(p_design)

    args = parser.parse_args()
    client = TTSClient(args.base_url)

    if args.command == "status":
        st = client.status()
        print(f"Model:       {st.model or '(none)'} {'[loading]' if st.loading else ''}")
        print(f"Loaded:      {st.loaded}")
        print(f"Type:        {st.model_type or 'N/A'}")
        print(f"Cached:      {', '.join(st.cached_models) or '(none)'}")
        print(f"Speakers:    {', '.join(st.speakers) or 'N/A'}")
        print(f"Transcribe:  {'yes' if st.transcription_available else 'no'}")
        print(f"Queue:       {st.queue_depth}")
        print(f"Available models:")
        for m in st.available_models:
            tag = " <-- active" if m == st.model else ""
            print(f"  - {m}{tag}")
        if st.preset_refs:
            print(f"Preset refs:")
            for p in st.preset_refs:
                print(f"  - {p['id']}: {p['label']} — \"{p['ref_text'][:60]}...\"")

    elif args.command == "load":
        result = client.load_model(args.model)
        print(f"Result: {result['status']} — {args.model}")

    elif args.command == "transcribe":
        text = client.transcribe(args.audio)
        print(f"Transcript: {text}")

    elif args.command == "clone":
        client.ensure_model(args.model)

        ref_text = args.ref_text
        if args.auto_transcribe and args.ref_audio and not ref_text:
            print("Transcribing reference audio...")
            ref_text = client.transcribe(args.ref_audio)
            print(f"  Transcript: {ref_text}")

        xvec_only = not args.full_clone
        kw = dict(
            text=args.text, ref_audio=args.ref_audio, ref_text=ref_text,
            ref_preset=args.ref_preset, language=args.language,
            xvec_only=xvec_only, temperature=args.temperature,
            top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        )
        if args.stream:
            _collect_stream(client.stream_voice_clone(**kw, chunk_size=args.chunk_size),
                            args.output)
        else:
            t0 = time.perf_counter()
            audio, sr, metrics = client.generate_voice_clone(**kw)
            print(f"  {(time.perf_counter()-t0)*1000:.0f}ms | "
                  f"Audio={metrics.get('audio_duration_s',0):.2f}s | "
                  f"RTF={metrics.get('rtf',0):.1f}x")
            _play_or_save(audio, sr, args.output)

    elif args.command == "custom":
        client.ensure_model(args.model)
        kw = dict(
            text=args.text, speaker=args.speaker, language=args.language,
            instruct=args.instruct, temperature=args.temperature,
            top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        )
        if args.stream:
            _collect_stream(client.stream_custom_voice(**kw, chunk_size=args.chunk_size),
                            args.output)
        else:
            t0 = time.perf_counter()
            audio, sr, metrics = client.generate_custom_voice(**kw)
            print(f"  {(time.perf_counter()-t0)*1000:.0f}ms | "
                  f"Audio={metrics.get('audio_duration_s',0):.2f}s | "
                  f"RTF={metrics.get('rtf',0):.1f}x")
            _play_or_save(audio, sr, args.output)

    elif args.command == "design":
        client.ensure_model(args.model)
        kw = dict(
            text=args.text, instruct=args.instruct, language=args.language,
            temperature=args.temperature, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        if args.stream:
            _collect_stream(client.stream_voice_design(**kw, chunk_size=args.chunk_size),
                            args.output)
        else:
            t0 = time.perf_counter()
            audio, sr, metrics = client.generate_voice_design(**kw)
            print(f"  {(time.perf_counter()-t0)*1000:.0f}ms | "
                  f"Audio={metrics.get('audio_duration_s',0):.2f}s | "
                  f"RTF={metrics.get('rtf',0):.1f}x")
            _play_or_save(audio, sr, args.output)


if __name__ == "__main__":
    main()
