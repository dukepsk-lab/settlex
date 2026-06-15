"""Format and deliver signals via the Telegram Bot API (plain `requests`)."""
from __future__ import annotations

from typing import Dict, List

import requests

from ..config import TelegramConfig

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_LEN = 4000  # Telegram hard limit is 4096; leave headroom


def _chunk(text: str, limit: int = MAX_LEN) -> List[str]:
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks or [""]


def send_message(text: str, config: TelegramConfig, parse_mode: str = "Markdown") -> List[dict]:
    """Send ``text`` to the configured chat, splitting if over the length limit."""
    if not config.is_complete:
        raise ValueError(
            "Telegram is not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
        )
    url = API_URL.format(token=config.bot_token)
    responses = []
    for chunk in _chunk(text):
        resp = requests.post(
            url,
            json={"chat_id": config.chat_id, "text": chunk, "parse_mode": parse_mode},
            timeout=30,
        )
        resp.raise_for_status()
        responses.append(resp.json())
    return responses


def format_signal(result: Dict) -> str:
    """Render a signal result dict (see signals.generate) as a Markdown message."""
    date = result.get("date", "?")
    horizon = result.get("horizon", "?")
    capital = result.get("capital", 0)
    lines = [
        "*📈 SETTLEX — SET50 Signal*",
        f"_Date:_ {date}   _Horizon:_ {horizon} trading days",
        f"_Capital:_ ฿{capital:,.0f}",
        "",
    ]

    positions = result.get("positions", [])
    if not positions:
        lines.append("*No qualifying long signals today → hold cash (100%).*")
    else:
        invested = sum(p["weight"] for p in positions)
        lines.append("*Top picks* (Markowitz MV, long-only):")
        lines.append("```")
        lines.append(f"{'#':<2}{'SYM':<8}{'WT':>7}{'THB':>12}{'PRED':>8}")
        for i, p in enumerate(positions, 1):
            lines.append(
                f"{i:<2}{p['symbol']:<8}{p['weight']*100:>6.1f}%"
                f"{p['thb']:>12,.0f}{p['pred_return']*100:>7.1f}%"
            )
        lines.append("```")
        cash = max(0.0, 1.0 - invested)
        lines.append(f"_Cash:_ {cash*100:.1f}%")

    agreement = result.get("model_agreement")
    if agreement is not None:
        lines.append(f"_Model agreement (CNN-BiLSTM↔XGB rank corr):_ {agreement:.2f}")

    lines += [
        "",
        "⚠️ _Decision-support signal only. Not financial advice. "
        "Not auto-executed — review before trading._",
    ]
    return "\n".join(lines)
