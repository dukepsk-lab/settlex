"""Prediction-accuracy evaluation — pure, no network or LLM.

Given a previously-saved signal and current OHLCV, compute how the predicted
Top-N actually performed: realised returns, hit rate, and the recommended
portfolio's return versus an equal-weight universe benchmark. This is what
DeepSeek narrates in the daily briefing ("verify yesterday's calls").
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import pandas as pd


def _close_asof(df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
    s = df.sort_values("date").set_index("date")["close"]
    s = s[s.index <= date]
    return float(s.iloc[-1]) if len(s) else None


def _latest_close(df: pd.DataFrame, as_of: Optional[pd.Timestamp]):
    s = df.sort_values("date").set_index("date")["close"]
    if as_of is not None:
        s = s[s.index <= as_of]
    if not len(s):
        return None, None
    return s.index[-1], float(s.iloc[-1])


def evaluate_predictions(
    prev_signal: Dict,
    ohlcv: Dict[str, pd.DataFrame],
    as_of: Optional[str] = None,
) -> Dict:
    """Compare ``prev_signal``'s picks against realised prices in ``ohlcv``."""
    d0 = pd.Timestamp(prev_signal["date"])
    as_of_ts = pd.Timestamp(as_of) if as_of else None
    positions = prev_signal.get("positions", [])

    rows = []
    last_dates = []
    for p in positions:
        df = ohlcv.get(p["symbol"])
        if df is None or df.empty:
            continue
        p0 = _close_asof(df, d0)
        d1, p1 = _latest_close(df, as_of_ts)
        if p0 is None or p1 is None or p0 <= 0 or d1 is None or d1 <= d0:
            continue
        last_dates.append(d1)
        realized = p1 / p0 - 1
        pred = float(p.get("pred_return", float("nan")))
        rows.append(
            {
                "symbol": p["symbol"],
                "weight": float(p.get("weight", 0.0)),
                "pred_return": pred,
                "realized_return": realized,
                "correct": None if math.isnan(pred) else ((pred > 0) == (realized > 0)),
            }
        )

    portfolio_return = hit_rate = None
    if rows:
        wsum = sum(r["weight"] for r in rows) or 1.0
        portfolio_return = sum(r["weight"] * r["realized_return"] for r in rows) / wsum
        graded = [r for r in rows if r["correct"] is not None]
        if graded:
            hit_rate = sum(1 for r in graded if r["correct"]) / len(graded)

    # Equal-weight universe benchmark over the same window.
    bench = []
    for df in ohlcv.values():
        if df is None or df.empty:
            continue
        p0 = _close_asof(df, d0)
        _, p1 = _latest_close(df, as_of_ts)
        if p0 and p1 and p0 > 0:
            bench.append(p1 / p0 - 1)
    benchmark_return = (sum(bench) / len(bench)) if bench else None

    eval_date = max(last_dates) if last_dates else None
    days_elapsed = int(len(pd.bdate_range(d0, eval_date)) - 1) if eval_date is not None else None
    excess = (
        portfolio_return - benchmark_return
        if portfolio_return is not None and benchmark_return is not None
        else None
    )

    return {
        "signal_date": prev_signal["date"],
        "eval_date": eval_date.date().isoformat() if eval_date is not None else None,
        "days_elapsed": days_elapsed,
        "horizon": prev_signal.get("horizon"),
        "positions": rows,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess,
        "hit_rate": hit_rate,
    }
