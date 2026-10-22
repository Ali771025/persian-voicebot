from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

from .config import settings
from .audio_utils import pcm16le_to_wav_bytes, SAMPLE_RATE

_TS_LINE = re.compile(r"^\[[0-9:\.\- >]+\]\s+(.*)$")

def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout[-4000:])
    return p.stdout

class WhisperCpp:
    """Thin wrapper around whisper-cli."""
    def __init__(self):
        if not os.path.exists(settings.WHISPER_CLI):
            raise FileNotFoundError(f"whisper-cli not found: {settings.WHISPER_CLI}")
        if not os.path.exists(settings.WHISPER_MODEL):
            raise FileNotFoundError(f"whisper model not found: {settings.WHISPER_MODEL}")
        self._help = _run([settings.WHISPER_CLI, "-h"])

    def build_cmd(self, wav_path: str, language: str | None) -> list[str]:
        cmd = [
            settings.WHISPER_CLI,
            "-m", settings.WHISPER_MODEL,
            "-f", wav_path,
            "-t", str(settings.WHISPER_THREADS),
        ]
        if language:
            if "--language" in self._help:
                cmd += ["--language", language]
            elif "\n-l " in self._help or " -l " in self._help:
                cmd += ["-l", language]
        if "--no-timestamps" in self._help:
            cmd += ["--no-timestamps"]
        return cmd

    def transcribe_wav_file(self, wav_path: str, language: str | None = None) -> str:
        out = _run(self.build_cmd(wav_path, language))
        lines = []
        for line in out.splitlines():
            line = line.strip()
            m = _TS_LINE.match(line)
            if m:
                lines.append(m.group(1).strip())
        text = " ".join(lines).strip()
        if not text:
            tail = [ln.strip() for ln in out.splitlines() if ln.strip()]
            text = " ".join(tail[-5:]).strip()
        return text

    def transcribe_pcm16le(self, pcm16: bytes, language: str | None = None) -> str:
        wav_bytes = pcm16le_to_wav_bytes(pcm16, sr=SAMPLE_RATE)
        with tempfile.TemporaryDirectory(prefix="stt_") as d:
            wav_path = os.path.join(d, "audio.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            return self.transcribe_wav_file(wav_path, language)

@dataclass
class RealtimeSTTConfig:
    window_sec: float = 8.0

class RealtimeSTTSession:
    """Sliding window buffer for realtime partial transcriptions."""
    def __init__(self, whisper: WhisperCpp, cfg: RealtimeSTTConfig):
        self.whisper = whisper
        self.cfg = cfg
        self.buf = bytearray()
        self.max_bytes = int(cfg.window_sec * SAMPLE_RATE * 2)
        self.last_text = ""

    def reset(self):
        self.buf = bytearray()
        self.last_text = ""

    def append_pcm16(self, pcm16: bytes):
        self.buf.extend(pcm16)
        if len(self.buf) > self.max_bytes:
            extra = len(self.buf) - self.max_bytes
            del self.buf[:extra]

    def transcribe_window(self) -> str:
        lang = settings.WHISPER_LANG.strip() or None
        return self.whisper.transcribe_pcm16le(bytes(self.buf), lang)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def delta(self, new_text: str) -> str:
        l = self._lcp(self.last_text, new_text)
        d = new_text[l:].strip()
        self.last_text = new_text
        return d
