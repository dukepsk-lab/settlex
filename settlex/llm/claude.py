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
        import anthropic  # lazy import — only needed when briefing runs

        client = anthropic.Anthropic(api_key=self._cfg.anthropic_api_key)
        kwargs = {
            "model": self._cfg.anthropic_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text").strip()
