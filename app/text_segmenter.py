from __future__ import annotations
from dataclasses import dataclass

PUNCT = {"。", ".", "!", "?", "؟", "…", "\n"}

@dataclass
class SegmenterConfig:
    min_flush_chars: int = 28
    max_flush_chars: int = 140

class SentenceSegmenter:
    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg

    def should_flush(self, buf: str) -> bool:
        t = buf.strip()
        if len(t) >= self.cfg.max_flush_chars:
            return True
        if len(t) >= self.cfg.min_flush_chars and any(t.endswith(p) for p in PUNCT):
            return True
        return False

    def split_flush(self, buf: str) -> tuple[str, str]:
        t = buf.strip()
        if not t:
            return "", ""

        last = -1
        for i, ch in enumerate(t):
            if ch in PUNCT:
                last = i
        if last >= 10:
            return t[:last+1].strip(), t[last+1:].strip()

        if len(t) > self.cfg.max_flush_chars:
            return t[: self.cfg.max_flush_chars].strip(), t[self.cfg.max_flush_chars :].strip()

        return "", t
