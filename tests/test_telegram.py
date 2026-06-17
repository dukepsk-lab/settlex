import pytest

import settlex.signals.telegram as tg
from settlex.config import TelegramConfig
from settlex.signals.telegram import _chunk, format_signal, send_message


def _result():
    return {
        "date": "2026-06-15",
        "horizon": 5,
        "capital": 1_000_000,
        "model_agreement": 0.5,
        "positions": [
            {"symbol": "PTT", "weight": 0.30, "thb": 300000, "pred_return": 0.031},
            {"symbol": "AOT", "weight": 0.20, "thb": 200000, "pred_return": 0.020},
        ],
    }


def test_format_contains_symbols_and_disclaimer():
    msg = format_signal(_result())
    assert "PTT" in msg and "AOT" in msg
    assert "SETTLEX" in msg
    assert "เงินทุนเริ่มต้น" in msg


def test_format_cash_when_no_positions():
    result = _result()
    result["positions"] = []
    assert "ถือเงินสด" in format_signal(result)


def test_chunk_splits_long_text():
    text = "\n".join(f"line {i}" for i in range(2000))
    chunks = _chunk(text, 1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_send_message_posts_to_api(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    out = send_message("hello", TelegramConfig(bot_token="T", chat_id="C"))
    assert captured["json"]["chat_id"] == "C"
    assert "botT" in captured["url"]
    assert out[0]["ok"] is True


def test_send_message_requires_config():
    with pytest.raises(ValueError):
        send_message("x", TelegramConfig())
