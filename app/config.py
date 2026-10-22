import os
from dataclasses import dataclass

def _env(key: str, default: str) -> str:
    return os.getenv(key, default)

@dataclass(frozen=True)
class Settings:
    # whisper.cpp
    WHISPER_CLI: str = _env("WHISPER_CLI", os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"))
    WHISPER_MODEL: str = _env("WHISPER_MODEL", os.path.expanduser("~/whisper.cpp/models/ggml-small.bin"))
    WHISPER_THREADS: int = int(_env("WHISPER_THREADS", "6"))
    WHISPER_LANG: str = _env("WHISPER_LANG", "fa")  # "" برای auto

    # Ollama
    OLLAMA_HOST: str = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    OLLAMA_MODEL_REWRITE: str = _env("OLLAMA_MODEL_REWRITE", "mshojaei77/gemma3persian")  # اختیاری
    OLLAMA_TEMPERATURE: float = float(_env("OLLAMA_TEMPERATURE", "0.3"))
    ENABLE_REWRITE: bool = _env("ENABLE_REWRITE", "0") == "1"

    # Coqui TTS
    TTS_FEMALE_CONFIG: str = _env("TTS_FEMALE_CONFIG", os.path.expanduser("~/voicebot/models/tts/fa_female/config.json"))
    TTS_FEMALE_MODEL: str  = _env("TTS_FEMALE_MODEL",  os.path.expanduser("~/voicebot/models/tts/fa_female/model.pth"))
    TTS_MALE_CONFIG: str = _env("TTS_MALE_CONFIG", os.path.expanduser("~/voicebot/models/tts/fa_male/config.json"))
    TTS_MALE_MODEL: str  = _env("TTS_MALE_MODEL",  os.path.expanduser("~/voicebot/models/tts/fa_male/model.pth"))
    TTS_USE_CUDA: bool = _env("TTS_USE_CUDA", "1") == "1"

    # Realtime STT
    STT_WINDOW_SEC: float = float(_env("STT_WINDOW_SEC", "8.0"))
    STT_STEP_SEC: float = float(_env("STT_STEP_SEC", "0.5"))

    # TTS streaming
    OPUS_BITRATE_K: int = int(_env("OPUS_BITRATE_K", "24"))
    MSE_CHUNK_MS: int = int(_env("MSE_CHUNK_MS", "250"))
    MIN_FLUSH_CHARS: int = int(_env("MIN_FLUSH_CHARS", "28"))
    MAX_FLUSH_CHARS: int = int(_env("MAX_FLUSH_CHARS", "140"))

    # Media cache
    MEDIA_DIR: str = _env("MEDIA_DIR", os.path.expanduser("~/voicebot/media"))
    PUBLIC_BASE_URL: str = _env("PUBLIC_BASE_URL", "")  # e.g. http://localhost:8000

settings = Settings()
