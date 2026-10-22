from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, UploadFile, File, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio_utils import run_ffmpeg_to_pcm16le_mono_16k
from .stt_whispercpp import WhisperCpp
from .tts_coqui import tts_fa, tts_to_opus_webm
from .ws_voice import VoiceWebSocket, VoiceWSConfig
from .config import settings
from .media_cache import MediaStoreConfig, store_bytes, resolve_path


app = FastAPI(title="Local Voice Assistant (Whisper.cpp + Ollama + Coqui TTS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR), html=True), name="static")


@app.get("/")
async def index():
    idx = WEB_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"ok": True, "msg": "UI not found (web/index.html missing)"})


@app.get("/health")
async def health():
    return {"ok": True}


class TTSReq(BaseModel):
    text: str
    voice: str = "female"


@app.post("/api/tts")
async def api_tts(req: TTSReq, fmt: str = Query("webm_opus")):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "text is empty"}, status_code=400)

    if fmt == "wav":
        res = tts_fa(text, req.voice)
        return StreamingResponse(iter([res.wav_bytes]), media_type="audio/wav")

    webm = tts_to_opus_webm(text, voice=req.voice)
    return StreamingResponse(iter([webm]), media_type='audio/webm; codecs="opus"')


@app.post("/api/stt")
async def api_stt(file: UploadFile = File(...)):
    data = await file.read()
    pcm16 = run_ffmpeg_to_pcm16le_mono_16k(data)
    whisper = WhisperCpp()
    text = whisper.transcribe_pcm16le(pcm16, None)
    return JSONResponse({"text": text})


def _base_url(req: Request) -> str:
    if settings.PUBLIC_BASE_URL:
        return settings.PUBLIC_BASE_URL.rstrip("/")
    return str(req.base_url).rstrip("/")


@app.post("/media/store")
async def media_store(
    request: Request,
    ext: str = Query("webm"),
    file: UploadFile | None = File(default=None),
):
    if file is not None:
        data = await file.read()
    else:
        data = await request.body()

    if not data:
        raise HTTPException(status_code=400, detail="empty body/file")

    cfg = MediaStoreConfig(media_dir=Path(settings.MEDIA_DIR))
    media_id, path = store_bytes(data, cfg, ext=ext)

    audio_url = f"/media/{media_id}.{path.suffix.lstrip('.')}"
    return JSONResponse({
        "id": media_id,
        "audio_url": audio_url,
        "audio_full_url": _base_url(request) + audio_url,
        "bytes": len(data),
    })


@app.get("/media/{name}")
async def media_get(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="bad name")

    parts = name.rsplit(".", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="bad name")
    media_id, ext = parts[0], parts[1]

    cfg = MediaStoreConfig(media_dir=Path(settings.MEDIA_DIR))
    path = resolve_path(media_id, cfg, ext=ext)
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")

    mime = "audio/webm" if ext == "webm" else "audio/wav" if ext == "wav" else "audio/ogg"
    return FileResponse(str(path), media_type=mime)


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    handler = VoiceWebSocket(VoiceWSConfig())
    await handler.handle(ws)
