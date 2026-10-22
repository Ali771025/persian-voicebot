from __future__ import annotations

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional


@dataclass
class WebmChunkerConfig:
    bitrate_k: int = 24
    audio_chunk_ms: int = 250
    application: str = "voip"  # voip / audio / lowdelay


class WebmMSEChunker:
    """
    Generates:
      - init segment (header) once
      - subsequent media segments (chunks) compatible with MSE SourceBuffer (audio/webm; codecs="opus")

    Uses FFmpeg `webm_chunk` muxer which writes header + chunks as separate files.
    """

    def __init__(self, cfg: WebmChunkerConfig = WebmChunkerConfig()):
        self.cfg = cfg
        self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="mse_webm_")
        self.dir = Path(self.tmpdir_obj.name)
        self.header_path = self.dir / "init.hdr"
        self.chunk_pattern = str(self.dir / "chunk_%d.chk")

        self.proc: Optional[subprocess.Popen] = None
        self._next_chunk_index = 0
        self._closed = False

    def start(self, sample_rate: int):
        if self.proc:
            return

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-i", "pipe:0",
            "-c:a", "libopus",
            "-b:a", f"{self.cfg.bitrate_k}k",
            "-vbr", "on",
            "-application", self.cfg.application,
            "-f", "webm_chunk",
            "-header", str(self.header_path),
            "-chunk_start_index", "0",
            "-audio_chunk_duration", str(self.cfg.audio_chunk_ms),
            self.chunk_pattern,
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_pcm16(self, pcm16: bytes):
        if self._closed:
            return
        if (not self.proc) or (not self.proc.stdin):
            raise RuntimeError("chunker not started")
        self.proc.stdin.write(pcm16)
        self.proc.stdin.flush()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        if self.proc:
            try:
                await asyncio.to_thread(self.proc.wait, 5)
            except Exception:
                pass

    async def iter_segments(self) -> AsyncGenerator[tuple[str, bytes], None]:
        # Wait for init header
        while not self.header_path.exists():
            if self.proc and self.proc.poll() is not None:
                err = b""
                try:
                    err = self.proc.stderr.read() if self.proc.stderr else b""
                except Exception:
                    pass
                raise RuntimeError(f"ffmpeg exited before header. stderr={err[:2000]!r}")
            await asyncio.sleep(0.01)

        yield ("init", self.header_path.read_bytes())

        # Stream chunks
        while True:
            chunk_path = self.dir / f"chunk_{self._next_chunk_index}.chk"
            if chunk_path.exists():
                data = chunk_path.read_bytes()
                yield ("chunk", data)
                try:
                    chunk_path.unlink()
                except Exception:
                    pass
                self._next_chunk_index += 1
                continue

            if self._closed:
                await asyncio.sleep(0.05)
                # try to drain last chunk
                chunk_path = self.dir / f"chunk_{self._next_chunk_index}.chk"
                if chunk_path.exists():
                    continue
                break

            if self.proc and self.proc.poll() is not None:
                err = b""
                try:
                    err = self.proc.stderr.read() if self.proc.stderr else b""
                except Exception:
                    pass
                raise RuntimeError(f"ffmpeg exited. stderr={err[:2000]!r}")

            await asyncio.sleep(0.01)

    def cleanup(self):
        try:
            self.tmpdir_obj.cleanup()
        except Exception:
            pass
