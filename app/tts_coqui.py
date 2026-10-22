from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Tuple, List

import numpy as np
import soundfile as sf
from io import BytesIO

from TTS.utils.synthesizer import Synthesizer
from .config import settings
from .audio_utils import wav_bytes_to_webm_opus

Voice = Literal["female", "male"]

@dataclass
class TTSResult:
    pcm: np.ndarray
    sr: int
    wav_bytes: bytes

def _clean(text: str) -> str:
    t = (text or "").strip()
    t = " ".join(t.split())
    return t

@lru_cache(maxsize=4)
def _load_synth(voice: Voice) -> Synthesizer:
    if voice == "female":
        ckpt = settings.TTS_FEMALE_MODEL
        cfg = settings.TTS_FEMALE_CONFIG
    else:
        ckpt = settings.TTS_MALE_MODEL
        cfg = settings.TTS_MALE_CONFIG

    return Synthesizer(
        tts_checkpoint=ckpt,
        tts_config_path=cfg,
        use_cuda=settings.TTS_USE_CUDA,
    )

def tts_fa(text: str, voice: Voice = "female") -> TTSResult:
    synth = _load_synth(voice)
    t = _clean(text)
    if not t:
        raise ValueError("Empty text")

    wav_list = synth.tts(t)
    pcm = np.asarray(wav_list, dtype=np.float32)
    sr = int(synth.output_sample_rate)

    bio = BytesIO()
    sf.write(bio, pcm, sr, format="WAV", subtype="PCM_16")
    wav_bytes = bio.getvalue()

    return TTSResult(pcm=pcm, sr=sr, wav_bytes=wav_bytes)

def tts_to_opus_webm(text: str, voice: Voice = "female", bitrate_k: int = 24) -> bytes:
    r = tts_fa(text, voice)
    return wav_bytes_to_webm_opus(r.wav_bytes, bitrate_k=bitrate_k)

def merge_pcm(chunks: List[Tuple[np.ndarray, int]]) -> bytes:
    """Merge PCM chunks (must share sr) into one WAV bytes."""
    if not chunks:
        return b""
    sr = chunks[0][1]
    for _, s in chunks:
        if s != sr:
            raise ValueError("Sample rate mismatch among TTS chunks")
    pcm = np.concatenate([c for c, _ in chunks], axis=0) if len(chunks) > 1 else chunks[0][0]
    bio = BytesIO()
    sf.write(bio, pcm, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()
