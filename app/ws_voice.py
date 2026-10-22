from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
from fastapi import WebSocket

from .config import settings
from .audio_utils import run_ffmpeg_to_pcm16le_mono_16k
from .stt_whispercpp import WhisperCpp, RealtimeSTTSession, RealtimeSTTConfig
from .llm_ollama import ollama_chat_stream, ollama_chat_once
from .tts_coqui import tts_fa, merge_pcm
from .text_segmenter import SentenceSegmenter, SegmenterConfig
from .vad_webrtc import WebRTCVAD, VADConfig
from .webm_mse_chunker import WebmMSEChunker, WebmChunkerConfig

AudioMode = Literal["mse", "legacy"]


def j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


@dataclass
class VoiceWSConfig:
    stt_window_sec: float = settings.STT_WINDOW_SEC
    stt_step_sec: float = settings.STT_STEP_SEC

    opus_bitrate_k: int = settings.OPUS_BITRATE_K
    mse_chunk_ms: int = int(float(getattr(settings, "MSE_CHUNK_MS", 250)))

    min_flush_chars: int = settings.MIN_FLUSH_CHARS
    max_flush_chars: int = settings.MAX_FLUSH_CHARS


@dataclass
class ChatMessage:
    role: str
    content: str


class VoiceWebSocket:
    """
    WS دوطرفه پیشرفته:
      - STT ریل تایم whisper.cpp
      - LLM استریم (Ollama)
      - TTS استریم با MediaSource/SourceBuffer (WebM init + chunk ها)
      - VAD حرفه‌ای با webrtcvad (auto-stop اختیاری)
      - Duplex/Barge-in: اگر کاربر شروع به صحبت کند، پاسخ دستیار قطع می‌شود.
    """

    def __init__(self, cfg: VoiceWSConfig):
        self.cfg = cfg

        self.whisper = WhisperCpp()
        self.rt = RealtimeSTTSession(self.whisper, RealtimeSTTConfig(window_sec=cfg.stt_window_sec))
        self.segmenter = SentenceSegmenter(SegmenterConfig(cfg.min_flush_chars, cfg.max_flush_chars))

        # recording state
        self.recording = False
        self.full_pcm = bytearray()

        # realtime STT state
        self._last_stt_at = 0.0
        self._stt_busy = False

        # assistant run state (interruptable)
        self._assistant_task: Optional[asyncio.Task] = None
        self._audio_sender_task: Optional[asyncio.Task] = None
        self._chunker: Optional[WebmMSEChunker] = None
        self._assistant_cancelled = False

        # session memory
        self.sessions: Dict[str, List[ChatMessage]] = {}

        # VAD
        self.auto_stop = False
        self.vad = WebRTCVAD(VADConfig(sample_rate=16000, frame_ms=20, aggressiveness=2))
        self._silence_started_at: Optional[float] = None
        self._auto_stop_after = 1.2  # seconds

        # audio mode
        self.audio_mode: AudioMode = "mse"

    # -------- session memory --------
    def _get_session(self, session_id: str) -> List[ChatMessage]:
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                ChatMessage("system", "تو یک دستیار فارسی هستی. کوتاه، محاوره‌ای و واضح جواب بده. ایموجی نگذار. عددها را تا حد ممکن با حروف بنویس."),
            ]
        return self.sessions[session_id]

    def _trim_session(self, session_id: str, keep: int = 20):
        msgs = self.sessions.get(session_id, [])
        if len(msgs) > keep:
            sys = msgs[0:1]
            tail = msgs[-(keep - 1):]
            self.sessions[session_id] = sys + tail

    # -------- interrupt / cancel --------
    async def _cancel_assistant(self, ws: WebSocket, reason: str = "interrupt"):
        self._assistant_cancelled = True

        if self._assistant_task and not self._assistant_task.done():
            self._assistant_task.cancel()

        if self._audio_sender_task and not self._audio_sender_task.done():
            self._audio_sender_task.cancel()

        if self._chunker:
            try:
                await self._chunker.close()
            except Exception:
                pass
            try:
                self._chunker.cleanup()
            except Exception:
                pass
            self._chunker = None

        await ws.send_text(j({"type": "assistant.interrupted", "reason": reason}))

    # -------- STT partial --------
    async def _partial_stt_task(self, ws: WebSocket):
        try:
            text = await asyncio.to_thread(self.rt.transcribe_window)
            delta = self.rt.delta(text)
            if delta and len(delta) >= 2:
                await ws.send_text(j({"type": "stt.partial", "text": text, "delta": delta}))
        finally:
            self._stt_busy = False

    # -------- optional rewrite --------
    async def _rewrite_if_enabled(self, text: str) -> str:
        if not settings.ENABLE_REWRITE:
            return text
        prompt = [
            {"role": "system", "content": "این متن را کوتاه‌تر، محاوره‌ای‌تر و مناسب خوانش صوتی کن. ایموجی نگذار. عددها را با حروف بنویس."},
            {"role": "user", "content": text},
        ]
        try:
            out = (await ollama_chat_once(prompt, model=settings.OLLAMA_MODEL_REWRITE)).strip()
            return out or text
        except Exception:
            return text

    # -------- MSE audio sender --------
    async def _audio_sender(self, ws: WebSocket, chunker: WebmMSEChunker):
        seq = 0
        async for kind, data in chunker.iter_segments():
            if self._assistant_cancelled:
                break
            if kind == "init":
                await ws.send_text(j({"type": "audio.init", "mime": 'audio/webm; codecs="opus"'}))
                await ws.send_bytes(data)
            else:
                seq += 1
                await ws.send_text(j({"type": "audio.mse", "seq": seq}))
                await ws.send_bytes(data)

    # -------- assistant pipeline --------
    async def _assistant_stream_and_tts(self, ws: WebSocket, session_id: str, user_text: str, voice: str):
        self._assistant_cancelled = False
        await ws.send_text(j({"type": "assistant.start"}))

        pcm_chunks: List[Tuple[np.ndarray, int]] = []

        if self.audio_mode == "mse":
            self._chunker = WebmMSEChunker(WebmChunkerConfig(
                bitrate_k=self.cfg.opus_bitrate_k,
                audio_chunk_ms=self.cfg.mse_chunk_ms,
                application="voip",
            ))
            self._audio_sender_task = None

        msgs = self._get_session(session_id)
        msgs.append(ChatMessage("user", user_text))
        self._trim_session(session_id)

        messages = [{"role": m.role, "content": m.content} for m in msgs]

        full_answer = ""
        buf = ""

        try:
            async for delta in ollama_chat_stream(messages):
                if self._assistant_cancelled:
                    break

                full_answer += delta
                await ws.send_text(j({"type": "llm.delta", "delta": delta}))
                buf += delta

                if self.segmenter.should_flush(buf):
                    flush, rest = self.segmenter.split_flush(buf)
                    if not flush:
                        continue
                    buf = rest

                    flush2 = await self._rewrite_if_enabled(flush)
                    tts_res = await asyncio.to_thread(tts_fa, flush2, voice)
                    pcm_chunks.append((tts_res.pcm, tts_res.sr))

                    if self.audio_mode == "mse":
                        if not self._chunker:
                            raise RuntimeError("chunker missing")
                        if self._chunker.proc is None:
                            self._chunker.start(tts_res.sr)
                            self._audio_sender_task = asyncio.create_task(self._audio_sender(ws, self._chunker))

                        pcm16 = np.clip(tts_res.pcm, -1.0, 1.0)
                        pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
                        await asyncio.to_thread(self._chunker.write_pcm16, pcm16)

                    else:
                        # fallback: send WAV per chunk (client can play directly)
                        await ws.send_text(j({"type": "audio.chunk", "codec": "wav"}))
                        await ws.send_bytes(tts_res.wav_bytes)

            tail = buf.strip()
            if (not self._assistant_cancelled) and tail:
                tail2 = await self._rewrite_if_enabled(tail)
                tts_res = await asyncio.to_thread(tts_fa, tail2, voice)
                pcm_chunks.append((tts_res.pcm, tts_res.sr))

                if self.audio_mode == "mse":
                    if not self._chunker:
                        raise RuntimeError("chunker missing")
                    if self._chunker.proc is None:
                        self._chunker.start(tts_res.sr)
                        self._audio_sender_task = asyncio.create_task(self._audio_sender(ws, self._chunker))
                    pcm16 = np.clip(tts_res.pcm, -1.0, 1.0)
                    pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
                    await asyncio.to_thread(self._chunker.write_pcm16, pcm16)

        except asyncio.CancelledError:
            self._assistant_cancelled = True
        except Exception as e:
            await ws.send_text(j({"type": "error", "message": f"assistant_failed: {e}"}))
        finally:
            if full_answer.strip() and not self._assistant_cancelled:
                msgs.append(ChatMessage("assistant", full_answer.strip()))
                self._trim_session(session_id)

            if self._chunker:
                try:
                    await self._chunker.close()
                except Exception:
                    pass

            if self._audio_sender_task:
                try:
                    await asyncio.wait_for(self._audio_sender_task, timeout=3)
                except Exception:
                    pass

            if self._chunker:
                try:
                    self._chunker.cleanup()
                except Exception:
                    pass
                self._chunker = None

            # Final WAV merge
            if pcm_chunks and not self._assistant_cancelled:
                try:
                    wav_final = await asyncio.to_thread(merge_pcm, pcm_chunks)
                    await ws.send_text(j({"type": "audio.final", "codec": "wav", "container": "wav"}))
                    await ws.send_bytes(wav_final)
                except Exception as e:
                    await ws.send_text(j({"type": "warn", "message": f"final_wav_failed: {e}"}))

            await ws.send_text(j({"type": "assistant.done", "cancelled": self._assistant_cancelled}))

    # -------- handle websocket --------
    async def handle(self, ws: WebSocket):
        await ws.accept()
        await ws.send_text(j({"type": "hello", "msg": "voice ws ready"}))

        session_id = "default"

        try:
            while True:
                msg = await ws.receive()

                # JSON controls
                if "text" in msg and msg["text"] is not None:
                    data = json.loads(msg["text"])
                    t = data.get("type")

                    if t == "start":
                        # duplex: starting to speak interrupts assistant
                        if self._assistant_task and not self._assistant_task.done():
                            await self._cancel_assistant(ws, reason="barge_in_start")

                        session_id = data.get("session_id", session_id) or "default"
                        self.audio_mode = data.get("audio_mode", "mse")
                        if self.audio_mode not in ("mse", "legacy"):
                            self.audio_mode = "mse"

                        self.auto_stop = bool(data.get("auto_stop", False))
                        self._auto_stop_after = float(data.get("auto_stop_after", self._auto_stop_after))
                        self.vad.reset()
                        self._silence_started_at = None

                        self.recording = True
                        self.full_pcm = bytearray()
                        self.rt.reset()

                        await ws.send_text(j({
                            "type": "ack.start",
                            "session_id": session_id,
                            "audio_mode": self.audio_mode,
                            "auto_stop": self.auto_stop,
                        }))

                    elif t == "stop":
                        if not self.recording:
                            await ws.send_text(j({"type": "ack.stop", "note": "not recording"}))
                            continue

                        self.recording = False
                        await ws.send_text(j({"type": "ack.stop"}))

                        lang = settings.WHISPER_LANG.strip() or None
                        final_text = await asyncio.to_thread(self.whisper.transcribe_pcm16le, bytes(self.full_pcm), lang)
                        final_text = (final_text or "").strip()
                        await ws.send_text(j({"type": "stt.final", "text": final_text}))

                        voice = data.get("voice", "female")
                        if self._assistant_task and not self._assistant_task.done():
                            await self._cancel_assistant(ws, reason="new_request")
                        self._assistant_task = asyncio.create_task(self._assistant_stream_and_tts(ws, session_id, final_text, voice))

                    elif t == "text":
                        user_text = (data.get("text") or "").strip()
                        if not user_text:
                            continue

                        # duplex: typing interrupts too
                        if self._assistant_task and not self._assistant_task.done():
                            await self._cancel_assistant(ws, reason="new_text")

                        session_id = data.get("session_id", session_id) or "default"
                        self.audio_mode = data.get("audio_mode", self.audio_mode)
                        if self.audio_mode not in ("mse", "legacy"):
                            self.audio_mode = "mse"
                        voice = data.get("voice", "female")

                        await ws.send_text(j({"type": "user.text", "text": user_text, "session_id": session_id}))
                        self._assistant_task = asyncio.create_task(self._assistant_stream_and_tts(ws, session_id, user_text, voice))

                    elif t == "cancel":
                        if self._assistant_task and not self._assistant_task.done():
                            await self._cancel_assistant(ws, reason="user_cancel")
                        else:
                            await ws.send_text(j({"type": "ack.cancel", "note": "no active assistant"}))

                    else:
                        await ws.send_text(j({"type": "error", "message": f"Unknown type: {t}"}))

                # binary audio
                elif "bytes" in msg and msg["bytes"] is not None:
                    chunk = msg["bytes"]

                    if not self.recording:
                        # duplex: if user sends audio while assistant is talking, treat as barge-in
                        if self._assistant_task and not self._assistant_task.done():
                            await self._cancel_assistant(ws, reason="barge_in_audio")
                        continue

                    pcm16 = await asyncio.to_thread(run_ffmpeg_to_pcm16le_mono_16k, chunk)
                    self.full_pcm.extend(pcm16)
                    self.rt.append_pcm16(pcm16)

                    # VAD auto-stop
                    if self.auto_stop:
                        self.vad.update(pcm16)
                        now = time.time()
                        if self.vad.is_speaking:
                            self._silence_started_at = None
                        else:
                            if self._silence_started_at is None:
                                self._silence_started_at = now
                            elif (now - self._silence_started_at) >= self._auto_stop_after:
                                await ws.send_text(j({"type": "vad.auto_stop"}))
                                self.recording = False

                                lang = settings.WHISPER_LANG.strip() or None
                                final_text = await asyncio.to_thread(self.whisper.transcribe_pcm16le, bytes(self.full_pcm), lang)
                                final_text = (final_text or "").strip()
                                await ws.send_text(j({"type": "stt.final", "text": final_text}))

                                if self._assistant_task and not self._assistant_task.done():
                                    await self._cancel_assistant(ws, reason="new_request")
                                self._assistant_task = asyncio.create_task(self._assistant_stream_and_tts(ws, session_id, final_text, "female"))
                                continue

                    # realtime partial stt
                    now = time.time()
                    if (now - self._last_stt_at) >= self.cfg.stt_step_sec and not self._stt_busy:
                        self._last_stt_at = now
                        self._stt_busy = True
                        asyncio.create_task(self._partial_stt_task(ws))

                else:
                    await ws.send_text(j({"type": "error", "message": "Unsupported WS message"}))

        except Exception as e:
            try:
                await ws.send_text(j({"type": "error", "message": str(e)}))
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
