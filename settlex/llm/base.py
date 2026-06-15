"""Provider-agnostic LLM interface.

Each provider wraps one vendor SDK behind ``complete()``. Providers are optional:
``is_available()`` reports whether its API key is configured, and the briefing
orchestrator skips any provider that is unavailable or raises. LLMs are an
*advisory* layer — they never alter the quantitative signal (Top-N / weights).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Minimal text-completion interface shared by all providers."""

    name: str = "llm"

    @abstractmethod
    def is_available(self) -> bool:
        """True if this provider has the credentials it needs to run."""

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        """Return the model's text response for ``prompt`` (raises on failure)."""
