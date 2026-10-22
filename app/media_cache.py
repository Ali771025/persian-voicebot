from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class MediaStoreConfig:
    media_dir: Path


def _safe_ext(ext: str) -> str:
    ext = (ext or "").strip().lower().lstrip(".")
    if ext not in ("webm", "wav", "opus"):
        ext = "webm"
    return ext


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def store_bytes(data: bytes, cfg: MediaStoreConfig, ext: str = "webm") -> Tuple[str, Path]:
    ext = _safe_ext(ext)
    ensure_dir(cfg.media_dir)

    media_id = uuid.uuid4().hex
    path = cfg.media_dir / f"{media_id}.{ext}"
    tmp = cfg.media_dir / f".{media_id}.{ext}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return media_id, path


def resolve_path(media_id: str, cfg: MediaStoreConfig, ext: str = "webm") -> Path:
    ext = _safe_ext(ext)
    return cfg.media_dir / f"{media_id}.{ext}"
