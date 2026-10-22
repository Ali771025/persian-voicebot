from __future__ import annotations

import subprocess
import wave
from io import BytesIO
from typing import Iterator

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # PCM16LE

def run_ffmpeg_to_pcm16le_mono_16k(input_bytes: bytes) -> bytes:
    """Decode almost any audio (e.g., webm/opus) to raw PCM16LE mono 16k."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]
    p = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[:2000])
    return p.stdout

def pcm16le_to_wav_bytes(pcm16: bytes, sr: int = SAMPLE_RATE) -> bytes:
    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sr)
        wf.writeframes(pcm16)
    return bio.getvalue()

def wav_bytes_to_webm_opus(wav_bytes: bytes, bitrate_k: int = 24) -> bytes:
    """Convert WAV bytes to WebM/Opus for browser-friendly streaming."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-b:a", f"{bitrate_k}k",
        "-vbr", "on",
        "-application", "voip",
        "-f", "webm",
        "pipe:1",
    ]
    p = subprocess.run(cmd, input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[:2000])
    return p.stdout

def iter_bytes(data: bytes, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

def pcm16le_rms(pcm16: bytes) -> float:
    """Simple energy estimator for optional VAD."""
    if not pcm16:
        return 0.0
    # int16 little endian
    import numpy as np
    a = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if a.size == 0:
        return 0.0
    return float((a*a).mean() ** 0.5)
