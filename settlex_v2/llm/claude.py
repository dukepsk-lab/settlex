"""Claude provider via the official Anthropic SDK (synthesises the briefing)."""
from __future__ import annotations

from typing import Optional

from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, config):
        self._cfg = config

    def is_available(self) -> bool:
        return bool(getattr(self._cfg, "anthropic_api_key", None))

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        import httpx

        headers = {
            "x-api-key": self._cfg.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self._cfg.anthropic_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        with httpx.Client(timeout=120.0) as client:
            for attempt in range(3):
                try:
                    resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text").strip()
                except Exception as e:
                    if attempt == 2:
                        raise e
                    import time
                    time.sleep(2)
        return ""
