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
    """Diff ``prev_signal`` (or ``live_portfolio``) -> ``today_signal`` into explicit sell/buy actions."""
    if live_portfolio is not None and "positions" in live_portfolio:
        total_mv = max(1e-9, sum(p["market_value"] for p in live_portfolio["positions"].values()))
        prev = {
            sym: {
                "symbol": sym,
                "weight": p["market_value"] / total_mv,
                "thb": p["market_value"],
                "shares": p["shares"],
                "average_cost": p.get("average_cost", 0.0)
            }
            for sym, p in live_portfolio["positions"].items()
        }
        prev_date = "Live Portfolio"
        base_cash = float(live_portfolio.get("cash", 0.0))
    else:
        prev = {p["symbol"]: p for p in (prev_signal or {}).get("positions", [])}
        prev_date = (prev_signal or {}).get("date")
        base_cash = 0.0

    today = {p["symbol"]: p for p in (today_signal or {}).get("positions", [])}

    sells: List[Dict] = []
    buys: List[Dict] = []
    holds_unchanged: List[Dict] = []
    
    cash_from_sell = 0.0

    # 1. Fully sold
    for sym, p in prev.items():
        if sym not in today:
            shares = p.get("shares") or 0
            if shares > 0:
                price = p.get("thb", 0) / shares
                avg_cost = p.get("average_cost", 0.0)
                result_pct = (price / avg_cost - 1) if avg_cost > 0 else 0.0
                thb = shares * price
                
                sells.append({
                    "symbol": sym,
                    "shares": shares,
                    "price": price,
                    "avg_cost": avg_cost,
                    "result_pct": result_pct,
                    "thb": thb
                })
                cash_from_sell += thb

    # 2. In both (Holds / Adjusts)
    for sym, t in today.items():
        tw, tthb, ts = float(t.get("weight", 0.0)), float(t.get("thb", 0.0)), t.get("shares")
        price = t.get("price", 0.0)
        pred = t.get("pred_return", 0.0)
        
        if sym in prev:
            p = prev[sym]
            pw, pthb, ps = float(p.get("weight", 0.0)), float(p.get("thb", 0.0)), p.get("shares")
            avg_cost = p.get("average_cost", 0.0)
            
            if ts is not None and ps is not None:
                sd = ts - ps
                if sd < 0:
                    sell_shares = abs(sd)
                    result_pct = (price / avg_cost - 1) if avg_cost > 0 else 0.0
                    thb = sell_shares * price
                    sells.append({
                        "symbol": sym,
                        "shares": sell_shares,
                        "price": price,
                        "avg_cost": avg_cost,
                        "result_pct": result_pct,
                        "thb": thb
                    })
                    cash_from_sell += thb
                elif sd > 0:
                    thb = sd * price
                    buys.append({
                        "symbol": sym,
                        "shares": sd,
                        "price": price,
                        "pred_pct": pred,
                        "thb": thb
                    })
                else:
                    holds_unchanged.append({"symbol": sym, "shares": ts, "thb": tthb})
        else:
            # 3. Newly added (Buys)
            if ts is not None and ts > 0:
                buys.append({
                    "symbol": sym,
                    "shares": ts,
                    "price": price,
                    "pred_pct": pred,
                    "thb": tthb
                })

    sells.sort(key=lambda r: r["thb"], reverse=True)
    buys.sort(key=lambda r: r["thb"], reverse=True)

    total_cash_for_buy = base_cash + cash_from_sell

    return {
        "prev_date": prev_date,
        "date": (today_signal or {}).get("date"),
        "has_prev": bool(prev),
        "sell": sells,
        "buy": buys,
        "hold": holds_unchanged,
        "cash_from_sell": cash_from_sell,
        "total_cash_for_buy": total_cash_for_buy,
        "base_cash": base_cash
    }


def has_changes(reb: Dict) -> bool:
    """True if any actual order is required (a sell, a buy, or a weight change)."""
    return bool(reb.get("sell") or reb.get("buy"))


def rebalance_summary_text(reb: Dict) -> str:
    """Plain-text action list (fed to the LLM order-checklist prompt)."""
    lines: List[str] = []
    for r in reb.get("sell", []):
        lines.append(f"ขาย {r['symbol']} {r['shares']} หุ้น (ได้เงิน ~฿{r['thb']:,.0f})")
    for r in reb.get("buy", []):
        lines.append(f"ซื้อ {r['symbol']} {r['shares']} หุ้น (฿{r['thb']:,.0f})")
        
    if not lines:
        return "ไม่มีการเปลี่ยนแปลง — ถือพอร์ตเดิมทั้งหมด"
    return "\n".join(lines)
