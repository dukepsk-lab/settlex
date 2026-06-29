"""Financial performance metrics for a return series.

All functions accept a sequence of periodic returns. ``periods_per_year`` lets
the caller annualise correctly when returns are sampled at the rebalance
cadence (e.g. every ``horizon`` trading days -> ``252 / horizon``).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns: Sequence[float]) -> pd.Series:
    return pd.Series(returns, dtype=float).dropna()


def equity_curve(returns: Sequence[float]) -> pd.Series:
    return (1 + _clean(returns)).cumprod()


def cagr(returns: Sequence[float], periods_per_year: float = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.empty:
        return 0.0
    total = float((1 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1 / years) - 1


def sharpe(returns: Sequence[float], rf: float = 0.0, periods_per_year: float = TRADING_DAYS) -> float:
    r = _clean(returns) - rf / periods_per_year
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std(ddof=1))


def sortino(returns: Sequence[float], rf: float = 0.0, periods_per_year: float = TRADING_DAYS) -> float:
    r = _clean(returns) - rf / periods_per_year
    downside = r[r < 0]
    if len(r) < 2 or downside.std(ddof=1) == 0 or downside.empty:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / downside.std(ddof=1))


def max_drawdown(returns: Sequence[float]) -> float:
    eq = equity_curve(returns)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    return float((eq / peak - 1).min())


def profit_factor(returns: Sequence[float]) -> float:
    r = _clean(returns)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def hit_rate(returns: Sequence[float]) -> float:
    r = _clean(returns)
    return float((r > 0).mean()) if not r.empty else 0.0


def summary(returns: Sequence[float], rf: float = 0.0, periods_per_year: float = TRADING_DAYS) -> Dict[str, float]:
    return {
        "CAGR": cagr(returns, periods_per_year),
        "Sharpe": sharpe(returns, rf, periods_per_year),
        "Sortino": sortino(returns, rf, periods_per_year),
        "MaxDrawdown": max_drawdown(returns),
        "ProfitFactor": profit_factor(returns),
        "HitRate": hit_rate(returns),
        "Periods": int(len(_clean(returns))),
    }
