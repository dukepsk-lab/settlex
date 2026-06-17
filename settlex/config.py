"""Central configuration, loaded from environment / .env file.

All secrets (Settrade credentials, Telegram bot token) and strategy
parameters live here so the rest of the package stays free of literals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # python-dotenv is optional at import time (e.g. in minimal test envs)
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _data_dir() -> Path:
    return Path(_get("SETTLEX_DATA_DIR", str(PROJECT_ROOT / "data")))


def _model_dir() -> Path:
    return Path(_get("SETTLEX_MODEL_DIR", str(PROJECT_ROOT / "models")))


@dataclass
class SettradeConfig:
    """Credentials for the Settrade Open API (`settrade-v2` SDK)."""

    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    broker_id: Optional[str] = None
    app_code: Optional[str] = None
    account_no: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SettradeConfig":
        return cls(
            app_id=_get("SETTRADE_APP_ID"),
            app_secret=_get("SETTRADE_APP_SECRET"),
            broker_id=_get("SETTRADE_BROKER_ID"),
            app_code=_get("SETTRADE_APP_CODE", "ALGO"),
            account_no=_get("SETTRADE_ACCOUNT_NO"),
        )

    @property
    def is_complete(self) -> bool:
        return all([self.app_id, self.app_secret, self.broker_id, self.app_code])


@dataclass
class TelegramConfig:
    """Credentials for delivering signals via the Telegram Bot API."""

    bot_token: Optional[str] = None
    chat_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        return cls(
            bot_token=_get("TELEGRAM_BOT_TOKEN"),
            chat_id=_get("TELEGRAM_CHAT_ID"),
        )

    @property
    def is_complete(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass
class LLMConfig:
    """Optional LLM providers for the daily briefing (each independent/optional)."""

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-opus-4-8"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            anthropic_api_key=_get("ANTHROPIC_API_KEY"),
            anthropic_model=_get("ANTHROPIC_MODEL", "claude-opus-4-8"),
            gemini_api_key=_get("GEMINI_API_KEY"),
            gemini_model=_get("GEMINI_MODEL", "gemini-2.5-flash"),
            deepseek_api_key=_get("DEEPSEEK_API_KEY"),
            deepseek_model=_get("DEEPSEEK_MODEL", "deepseek-chat"),
            deepseek_base_url=_get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            enabled=_get("SETTLEX_LLM_ENABLED", "0") in ("1", "true", "True"),
        )


@dataclass
class InnovestXConfig:
    """Credentials for InnovestX Webhook execution."""

    endpoint: Optional[str] = None
    api_secret: Optional[str] = None

    @classmethod
    def from_env(cls) -> "InnovestXConfig":
        return cls(
            endpoint=_get("INNOVESTX_ENDPOINT"),
            api_secret=_get("INNOVESTX_API_SECRET"),
        )

    @property
    def is_complete(self) -> bool:
        return bool(self.endpoint and self.api_secret)


@dataclass
class Settings:
    """Top-level settlex settings."""

    settrade: SettradeConfig = field(default_factory=SettradeConfig.from_env)
    telegram: TelegramConfig = field(default_factory=TelegramConfig.from_env)
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    innovestx: InnovestXConfig = field(default_factory=InnovestXConfig.from_env)

    # Market-data backend: "yahoo" (free, no creds, EOD) or "settrade".
    data_source: str = field(default_factory=lambda: _get("SETTLEX_DATA_SOURCE", "yahoo").lower())

    # Strategy parameters
    horizon: int = field(default_factory=lambda: int(_get("SETTLEX_HORIZON", "5")))
    lookback: int = field(default_factory=lambda: int(_get("SETTLEX_LOOKBACK", "60")))
    top_n: int = field(default_factory=lambda: int(_get("SETTLEX_TOP_N", "5")))
    max_weight: float = field(default_factory=lambda: float(_get("SETTLEX_MAX_WEIGHT", "0.30")))
    capital: float = field(default_factory=lambda: float(_get("SETTLEX_CAPITAL", "1000000")))
    risk_free_rate: float = field(default_factory=lambda: float(_get("SETTLEX_RISK_FREE", "0.02")))
    history_start: str = field(default_factory=lambda: _get("SETTLEX_HISTORY_START", "2015-01-01"))

    data_dir: Path = field(default_factory=_data_dir)
    model_dir: Path = field(default_factory=_model_dir)

    def ensure_dirs(self) -> None:
        (self.data_dir / "ohlcv").mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ensemble_path(self) -> Path:
        return self.model_dir / "ensemble"


def get_settings() -> Settings:
    """Build a fresh Settings object from the current environment."""
    return Settings()
