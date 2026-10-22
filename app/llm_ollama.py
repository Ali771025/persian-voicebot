from __future__ import annotations

import json
from typing import AsyncGenerator, Dict, List, Optional

import httpx
from .config import settings

async def ollama_chat_stream(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> AsyncGenerator[str, None]:
    """Stream text deltas from Ollama /api/chat (NDJSON)."""
    url = f"{settings.OLLAMA_HOST}/api/chat"
    payload = {
        "model": model or settings.OLLAMA_MODEL,
        "messages": messages,
        "options": {"temperature": temperature if temperature is not None else settings.OLLAMA_TEMPERATURE},
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done") is True:
                    break

async def ollama_chat_once(messages: List[Dict], model: Optional[str] = None) -> str:
    """Non-streaming helper (still consumes streaming output)."""
    out = []
    async for d in ollama_chat_stream(messages, model=model):
        out.append(d)
    return "".join(out).strip()
