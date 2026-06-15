"""Fractional differencing (Lopez de Prado, fixed-width window method).

Standard integer differencing (daily returns) makes a price series stationary
but destroys long-term memory. Fractional differencing applies a fractional
order ``d`` that makes the series stationary while preserving as much memory as
possible — the research report recommends this for the recurrent deep model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def frac_weights(d: float, size: int) -> np.ndarray:
    """Binomial weights for fractional differencing order ``d`` (length ``size``)."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def _ffd_weights(d: float, thres: float, max_width: int) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_width:
        next_w = -w[-1] * (d - k + 1) / k
        if abs(next_w) < thres:
            break
        w.append(next_w)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    thres: float = 1e-4,
    max_width: int = 100,
) -> pd.Series:
    """Fixed-width fractional differencing of ``series``.

    Returns a series of the same index with NaNs where there is insufficient
    history for the weight window.
    """
    weights = _ffd_weights(d, thres, max_width)
    width = len(weights)
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(width - 1, len(values)):
        window = values[i - width + 1 : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(weights, window))
    return pd.Series(out, index=series.index)


def min_ffd_order(
    series: pd.Series,
    thres: float = 1e-4,
    step: float = 0.1,
    max_d: float = 1.0,
    pvalue: float = 0.05,
    default: float = 0.4,
) -> float:
    """Smallest ``d`` in (0, max_d] making the series ADF-stationary.

    Requires statsmodels (optional dependency); falls back to ``default`` when
    it is unavailable or no order passes the test.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return default

    d = 0.0
    while d <= max_d + 1e-9:
        fd = frac_diff_ffd(series, d, thres).dropna()
        if len(fd) > 20:
            try:
                p = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
                if p < pvalue:
                    return round(d, 3)
            except Exception:
                pass
        d += step
    return default
