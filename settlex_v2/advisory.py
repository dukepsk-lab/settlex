"""Daily advisory: trading-day status, last signal summary, and action checklist."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, Optional

try:
    from zoneinfo import ZoneInfo
    _BKK = ZoneInfo("Asia/Bangkok")
except Exception:  # pragma: no cover — Python < 3.9 or missing tzdata
    _BKK = None

from .config import Settings, get_settings

# SET official holidays 2026 — verify at https://www.set.or.th before live use.
_TH_HOLIDAYS: frozenset = frozenset({
    "2026-01-01",  # New Year's Day
    "2026-02-05",  # Makha Bucha Day
    "2026-04-06",  # Chakri Day
    "2026-04-13",  # Songkran
    "2026-04-14",  # Songkran
    "2026-04-15",  # Songkran
    "2026-05-01",  # Labour Day
    "2026-05-04",  # Visakha Bucha Day
    "2026-06-03",  # HM Queen's Birthday
    "2026-07-02",  # Asanha Bucha Day
    "2026-07-03",  # Buddhist Lent Day
    "2026-07-28",  # HM King's Birthday (Rama X)
    "2026-08-12",  # HM Queen Mother's Birthday
    "2026-10-13",  # HM King Bhumibol Memorial Day
    "2026-10-23",  # Chulalongkorn Day
    "2026-12-05",  # HM King Bhumibol's Birthday / Father's Day
    "2026-12-07",  # Substitution (Dec 5 falls on Saturday)
    "2026-12-10",  # Constitution Day
    "2026-12-31",  # New Year's Eve
})

_TH_MONTHS = [
    "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
]
_TH_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

_LAST_SIGNAL_FILE = "last_signal.json"
_SIGNAL_HISTORY_DIR = "signal_history"
_SIGNAL_READY = dt.time(16, 35)  # 5-min buffer after SET close (16:30)


def _now_bkk() -> dt.datetime:
    if _BKK is not None:
        return dt.datetime.now(tz=_BKK).replace(tzinfo=None)
    return dt.datetime.now()  # fallback: assume system clock is BKK time


def is_trading_day(date: dt.date) -> bool:
    """Return True if the SET is expected to be open on *date*."""
    if date.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return date.isoformat() not in _TH_HOLIDAYS


def _next_trading_day(from_date: dt.date) -> dt.date:
    d = from_date + dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def _thai_date_str(date: dt.date) -> str:
    be_year = date.year + 543
    return f"วัน{_TH_DAYS[date.weekday()]}ที่ {date.day} {_TH_MONTHS[date.month]} {be_year}"


def save_last_signal(signal: Dict, data_dir: Path) -> None:
    """Persist the latest signal dict to disk so advisory can reference it."""
    path = data_dir / _LAST_SIGNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)


def load_last_signal(data_dir: Path) -> Optional[Dict]:
    path = data_dir / _LAST_SIGNAL_FILE
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def append_signal_history(signal: Dict, data_dir: Path) -> None:
    """Persist a dated copy of the signal so the briefing can verify it later."""
    date = signal.get("date")
    if not date:
        return
    hist_dir = data_dir / _SIGNAL_HISTORY_DIR
    hist_dir.mkdir(parents=True, exist_ok=True)
    with open(hist_dir / f"{date}.json", "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)


def load_previous_signal(data_dir: Path, before: str) -> Optional[Dict]:
    """Return the most recent stored signal strictly before date ``before``."""
    hist_dir = data_dir / _SIGNAL_HISTORY_DIR
    if not hist_dir.exists():
        return None
    candidates = sorted(
        p for p in hist_dir.glob("*.json") if p.stem < before
    )
    if not candidates:
        return None
    try:
        with open(candidates[-1], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def generate_advisory(settings: Optional[Settings] = None) -> Dict:
    settings = settings or get_settings()
    now = _now_bkk()
    today = now.date()
    trading = is_trading_day(today)
    after_close = now.time() >= _SIGNAL_READY

    minutes_until_signal: Optional[int] = None
    if trading and not after_close:
        target = dt.datetime.combine(today, _SIGNAL_READY)
        minutes_until_signal = max(0, int((target - now).total_seconds() / 60))

    next_td = _next_trading_day(today) if (not trading or after_close) else None

    return {
        "date": today.isoformat(),
        "thai_date": _thai_date_str(today),
        "is_trading_day": trading,
        "after_close": after_close,
        "minutes_until_signal": minutes_until_signal,
        "next_trading_day": next_td.isoformat() if next_td else None,
        "last_signal": load_last_signal(settings.data_dir),
    }


def format_advisory(advisory: Dict) -> str:
    lines = [
        "*📋 SETTLEX — แนะนำประจำวัน*",
        advisory["thai_date"],
        "",
    ]

    if not advisory["is_trading_day"]:
        nd = advisory.get("next_trading_day", "?")
        lines += [
            "🔴 *วันหยุด — ไม่ต้องรัน signal วันนี้*",
            f"  วันทำการถัดไป: {nd}",
        ]
    elif advisory.get("after_close"):
        lines += [
            "✅ *ตลาดปิดแล้ว — รัน signal ได้เลยตอนนี้!*",
            "",
            "⏩ *รัน:*",
            "  `python -m settlex.cli signal --dry-run`   ← ดูก่อน",
            "  `python -m settlex.cli signal`             ← ส่ง Telegram จริง",
        ]
    else:
        mins = advisory.get("minutes_until_signal")
        if mins is not None:
            h, m = divmod(mins, 60)
            wait_str = f" (อีก ~{h} ชม. {m} นาที)" if h else f" (อีก ~{m} นาที)"
        else:
            wait_str = ""
        lines += [
            "🟡 *ตลาดยังเปิดอยู่ — รอให้ตลาดปิดก่อน*",
            "",
            "🕐 *กำหนดการวันนี้* (เวลาไทย):",
            "  • 09:30 น. — ตลาดเปิด",
            "  • 12:30 น. — พักเที่ยง",
            "  • 14:00 น. — ตลาดเปิดรอบบ่าย",
            f"  • 16:35 น. — ⬅ รัน signal ได้เลย{wait_str}",
        ]

    lines.append("")

    last = advisory.get("last_signal")
    if last:
        last_date = last.get("date", "?")
        capital = last.get("capital", 0)
        positions = last.get("positions", [])
        lines.append(f"📊 *สัญญาณล่าสุด* ({last_date}):")
        for i, p in enumerate(positions[:5], 1):
            sym = p["symbol"]
            wt = p["weight"] * 100
            thb = p["thb"]
            shares = p.get("shares")
            s_text = f" ({shares} หุ้น)" if shares is not None else ""
            pred = p["pred_return"] * 100
            lines.append(
                f"  {i}. {sym:<8} {wt:5.1f}%   ฿{thb:>10,.0f}{s_text}   ({pred:+.1f}%)"
            )
        lines.append(f"  เงินลงทุนรวม: ฿{capital:,.0f}")
    else:
        lines.append("📊 _ยังไม่มีสัญญาณ — รัน_ `settlex signal` _ก่อน_")

    if advisory["is_trading_day"] and advisory.get("after_close"):
        nd = advisory.get("next_trading_day")
        if nd:
            lines += [
                "",
                f"📅 วันทำการถัดไป: {nd}",
                "  → รัน `settlex advisory` อีกครั้งเช้าวันนั้น",
            ]

    lines += [
        "",
        "⚠️ _เพื่อการตัดสินใจเสริมเท่านั้น ไม่ใช่คำแนะนำการลงทุน_",
    ]
    return "\n".join(lines)
