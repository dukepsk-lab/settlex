"""Markowitz mean-variance optimisation over the selected basket.

Uses PyPortfolioOpt: predicted returns become the expected-return vector and a
Ledoit-Wolf shrunk sample covariance becomes the risk model. Long-only, fully
invested, with a per-name cap. Falls back gracefully to a capped
prediction-proportional (or equal) weighting if the convex solve is infeasible.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _aligned_prices(symbols, price_history: Dict[str, pd.DataFrame], lookback: int) -> pd.DataFrame:
    closes = {}
    for symbol in symbols:
        df = price_history.get(symbol)
        if df is None or df.empty:
            continue
        closes[symbol] = df.sort_values("date").set_index("date")["close"].tail(lookback + 5)
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).dropna()


def optimize_weights(
    selected: pd.Series,
    price_history: Dict[str, pd.DataFrame],
    max_weight: float = 0.30,
    risk_free_rate: float = 0.02,
    horizon: int = 5,
    lookback: int = 252,
) -> pd.Series:
    """Return optimal long-only weights (summing to 1) for ``selected`` names.

    ``selected`` is a Series of predicted ``horizon``-day returns indexed by
    symbol. ``price_history`` maps symbol -> OHLCV frame for the covariance.
    """
    symbols = list(selected.index)
    if not symbols:
        return pd.Series(dtype=float)
    if len(symbols) == 1:
        return pd.Series({symbols[0]: 1.0})

    prices = _aligned_prices(symbols, price_history, lookback)
    if prices.shape[0] < 30 or prices.shape[1] < 2:
        return pd.Series(1.0 / len(symbols), index=symbols)

    # A per-name cap below 1/n makes "fully invested" infeasible; lift it just enough.
    n = prices.shape[1]
    cap = max(max_weight, 1.0 / n + 1e-3)

    # Annualise predicted horizon returns to match the annualised covariance.
    mu = selected.reindex(prices.columns).astype(float) * (TRADING_DAYS / horizon)

    try:
        from pypfopt import EfficientFrontier, risk_models

        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, cap))
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = pd.Series(ef.clean_weights())
        weights = weights.reindex(prices.columns).fillna(0.0)
        if weights.sum() > 0:
            return weights / weights.sum()
    except Exception:
        pass

    # Fallback: capped prediction-proportional weighting.
    w = selected.reindex(prices.columns).clip(lower=0.0)
    if w.sum() == 0:
        w = pd.Series(1.0, index=prices.columns)
    w = (w / w.sum()).clip(upper=cap)
    return w / w.sum()


def allocate_capital(weights: pd.Series, capital: float) -> pd.Series:
    """Convert weights to a THB allocation per name."""
    return (weights * capital).round(0)
