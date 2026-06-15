"""Briefing orchestration tests using mock providers (no network/LLM)."""
import json

import pytest

from settlex.briefing import format_briefing, generate_briefing
from settlex.config import get_settings


class FakeProvider:
    def __init__(self, name, available=True, reply="ok", raises=False):
        self.name = name
        self._available = available
        self._reply = reply
        self._raises = raises
        self.calls = 0

    def is_available(self):
        return self._available

    def complete(self, prompt, system=None, max_tokens=2048):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._reply


def _signal(date="2026-06-12"):
    return {
        "date": date,
        "horizon": 5,
        "capital": 1_000_000,
        "model_agreement": 0.5,
        "positions": [
            {"symbol": "AAA", "weight": 0.6, "thb": 600_000, "pred_return": 0.01},
            {"symbol": "BBB", "weight": 0.4, "thb": 400_000, "pred_return": 0.008},
        ],
    }


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTLEX_DATA_DIR", str(tmp_path))
    (tmp_path / "ohlcv").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_last_signal(data_dir, signal):
    with open(data_dir / "last_signal.json", "w", encoding="utf-8") as f:
        json.dump(signal, f)


def test_briefing_assembles_sections_from_providers(data_dir):
    _write_last_signal(data_dir, _signal())
    providers = [
        FakeProvider("claude", reply="ORDER PLAN"),
        FakeProvider("gemini", reply="NEWS"),
        FakeProvider("deepseek", reply="VERIFY"),
    ]
    briefing = generate_briefing(get_settings(), synthetic=True, providers=providers)
    # No previous signal stored -> no evaluation -> deepseek not called
    assert briefing["sections"].get("news") == "NEWS"
    assert briefing["sections"].get("orders") == "ORDER PLAN"
    assert "verification" not in briefing["sections"]
    msg = format_briefing(briefing)
    assert "NEWS" in msg and "ORDER PLAN" in msg
    assert "AAA" in msg  # deterministic target table always present


def test_unavailable_provider_is_skipped(data_dir):
    _write_last_signal(data_dir, _signal())
    gem = FakeProvider("gemini", available=False)
    providers = [FakeProvider("claude", reply="ORDER PLAN"), gem]
    briefing = generate_briefing(get_settings(), synthetic=True, providers=providers)
    assert gem.calls == 0
    assert "news" not in briefing["sections"]
    assert briefing["sections"].get("orders") == "ORDER PLAN"


def test_failing_provider_degrades_gracefully(data_dir):
    _write_last_signal(data_dir, _signal())
    providers = [FakeProvider("claude", raises=True), FakeProvider("gemini", reply="NEWS")]
    briefing = generate_briefing(get_settings(), synthetic=True, providers=providers)
    # Claude raised -> orders omitted, but briefing still builds with news + table
    assert briefing["sections"].get("orders") is None
    assert briefing["sections"].get("news") == "NEWS"
    msg = format_briefing(briefing)
    assert "AAA" in msg


def test_no_providers_still_shows_signal_table(data_dir):
    _write_last_signal(data_dir, _signal())
    briefing = generate_briefing(get_settings(), synthetic=True, providers=[])
    assert briefing["providers_used"] == []
    msg = format_briefing(briefing)
    assert "AAA" in msg and "BBB" in msg


def test_previous_signal_triggers_evaluation(data_dir):
    # store a previous-day signal in history so evaluation runs against synthetic prices
    from settlex.advisory import append_signal_history

    append_signal_history(_signal(date="2026-06-09"), data_dir)
    _write_last_signal(data_dir, _signal(date="2026-06-12"))
    deepseek = FakeProvider("deepseek", reply="VERIFY")
    briefing = generate_briefing(get_settings(), synthetic=True, providers=[deepseek])
    assert briefing["evaluation"] is not None
    assert deepseek.calls == 1
    assert briefing["sections"].get("verification") == "VERIFY"
