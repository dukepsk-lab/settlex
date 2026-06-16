"""Turnover/rebalance diff between two consecutive signals — pure, no LLM.

Given yesterday's target portfolio (assumed to be what you currently hold) and
today's target, derive the explicit per-name actions:

  * SELL        — held yesterday, dropped today
  * HOLD/ADJUST — in both; weight (and THB) may have moved
  * BUY         — newly added today

This turns the deterministic ML signal into an actionable order delta. It does
NOT alter the signal — it only describes the difference. The baseline is the
*previous signal* (i.e. it assumes you executed it); when a live broker
positions feed is wired in later, that can replace the baseline for exactness.
"""
from __future__ import annotations

from typing import Dict, List, Optional

_EPS = 1e-9


def compute_rebalance(prev_signal: Optional[Dict], today_signal: Optional[Dict], live_portfolio: Optional[Dict] = None) -> Dict:
    """Diff ``prev_signal`` (or ``live_portfolio``) -> ``today_signal`` into sell/hold/buy actions."""
    if live_portfolio is not None and "positions" in live_portfolio:
        total_mv = max(1e-9, sum(p["market_value"] for p in live_portfolio["positions"].values()))
        prev = {
            sym: {
                "symbol": sym,
                "weight": p["market_value"] / total_mv,
                "thb": p["market_value"],
                "shares": p["shares"],
            }
            for sym, p in live_portfolio["positions"].items()
        }
        prev_date = "Live Portfolio"
    else:
        prev = {p["symbol"]: p for p in (prev_signal or {}).get("positions", [])}
        prev_date = (prev_signal or {}).get("date")

    today = {p["symbol"]: p for p in (today_signal or {}).get("positions", [])}

    sells: List[Dict] = []
    holds: List[Dict] = []
    buys: List[Dict] = []

    for sym, p in prev.items():
        if sym not in today:
            sells.append(
                {
                    "symbol": sym,
                    "prev_weight": float(p.get("weight", 0.0)),
                    "prev_thb": float(p.get("thb", 0.0)),
                    "prev_shares": p.get("shares"),
                }
            )

    for sym, t in today.items():
        tw, tthb, ts = float(t.get("weight", 0.0)), float(t.get("thb", 0.0)), t.get("shares")
        if sym in prev:
            p = prev[sym]
            pw, pthb, ps = float(p.get("weight", 0.0)), float(p.get("thb", 0.0)), p.get("shares")
            holds.append(
                {
                    "symbol": sym,
                    "prev_weight": pw,
                    "weight": tw,
                    "prev_thb": pthb,
                    "thb": tthb,
                    "prev_shares": ps,
                    "shares": ts,
                    "weight_delta": tw - pw,
                    "thb_delta": tthb - pthb,
                    "shares_delta": (ts - ps) if (ts is not None and ps is not None) else None,
                }
            )
        else:
            buys.append({"symbol": sym, "weight": tw, "thb": tthb, "shares": ts})

    sells.sort(key=lambda r: r["prev_thb"], reverse=True)
    holds.sort(key=lambda r: r["thb"], reverse=True)
    buys.sort(key=lambda r: r["thb"], reverse=True)

    # One-way turnover (fraction of portfolio traded) = half the gross weight change.
    union = set(prev) | set(today)
    gross = sum(
        abs(float(today.get(s, {}).get("weight", 0.0)) - float(prev.get(s, {}).get("weight", 0.0)))
        for s in union
    )

    return {
        "prev_date": prev_date,
        "date": (today_signal or {}).get("date"),
        "has_prev": bool(prev),
        "sell": sells,
        "hold": holds,
        "buy": buys,
        "turnover": gross / 2.0,
    }


def has_changes(reb: Dict) -> bool:
    """True if any actual order is required (a sell, a buy, or a weight change)."""
    if reb.get("sell") or reb.get("buy"):
        return True
    for h in reb.get("hold", []):
        if h.get("shares_delta") is not None:
            if h["shares_delta"] != 0:
                return True
        elif abs(h["weight_delta"]) > _EPS:
            return True
    return False


def rebalance_summary_text(reb: Dict) -> str:
    """Plain-text action list (fed to the LLM order-checklist prompt)."""
    lines: List[str] = []
    for r in reb.get("sell", []):
        s_text = f" {r['prev_shares']} หุ้น" if r.get("prev_shares") is not None else ""
        lines.append(
            f"ขายปิด {r['symbol']}{s_text} (เคยถือ {r['prev_weight']*100:.1f}%, ได้เงิน ~฿{r['prev_thb']:,.0f})"
        )
    for r in reb.get("hold", []):
        sd = r.get("shares_delta")
        if sd is not None:
            if sd == 0:
                lines.append(f"ถือ {r['symbol']} {r['shares']} หุ้น คงเดิม (น้ำหนัก {r['weight']*100:.1f}%)")
            else:
                verb = "ซื้อเพิ่ม" if sd > 0 else "ขายออก"
                cost = f"ใช้เงิน" if sd > 0 else "ได้เงิน"
                lines.append(
                    f"{verb} {r['symbol']} {abs(sd)} หุ้น (รวมเป็น {r['shares']} หุ้น, {cost} ~฿{abs(r['thb_delta']):,.0f})"
                )
        else:
            if abs(r["weight_delta"]) <= _EPS:
                lines.append(f"ถือ {r['symbol']} คงน้ำหนัก {r['weight']*100:.1f}%")
            else:
                verb = "เพิ่ม" if r["thb_delta"] > 0 else "ลด"
                lines.append(
                    f"ถือ {r['symbol']} {r['prev_weight']*100:.1f}%→{r['weight']*100:.1f}% "
                    f"({verb} ฿{abs(r['thb_delta']):,.0f})"
                )
    for r in reb.get("buy", []):
        s_text = f" {r['shares']} หุ้น" if r.get("shares") is not None else ""
        lines.append(f"ซื้อใหม่ {r['symbol']}{s_text} {r['weight']*100:.1f}% (฿{r['thb']:,.0f})")
    if not lines:
        return "ไม่มีการเปลี่ยนแปลง — ถือพอร์ตเดิมทั้งหมด"
    return "\n".join(lines)
