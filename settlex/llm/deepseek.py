"""DeepSeek provider via the OpenAI-compatible SDK (verifies yesterday's calls)."""
from __future__ import annotations

from typing import Optional

from .base import LLMProvider


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def __init__(self, config):
        self._cfg = config

    def is_available(self) -> bool:
        return bool(getattr(self._cfg, "deepseek_api_key", None))

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        from openai import OpenAI  # lazy import — DeepSeek speaks the OpenAI API

        client = OpenAI(api_key=self._cfg.deepseek_api_key, base_url=self._cfg.deepseek_base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self._cfg.deepseek_model,
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
