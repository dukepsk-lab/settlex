"""Gemini provider via the `google-genai` SDK, with Google Search grounding.

Search grounding lets Gemini pull current news itself, which is what we want for
the daily market/news summary. If the grounding types differ in your installed
SDK version, see https://ai.google.dev/gemini-api/docs/grounding — the provider
is optional, so a failure here just omits the news section from the briefing.
"""
from __future__ import annotations

from typing import Optional

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config):
        self._cfg = config

    def is_available(self) -> bool:
        return bool(getattr(self._cfg, "gemini_api_key", None))

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        from google import genai  # lazy import
        from google.genai import types

        client = genai.Client(api_key=self._cfg.gemini_api_key)
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        
        models = [self._cfg.gemini_model, "gemini-3.1-flash-lite", "gemini-3-flash-preview"]
        last_err = None
        
        for model in models:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                return (resp.text or "").strip()
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                    print(f"[warn] Gemini model {model} failed (High Demand), falling back...")
                    continue
                else:
                    raise e
                    
        if last_err:
            raise last_err
        return ""
