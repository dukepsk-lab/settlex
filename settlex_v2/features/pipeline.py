"""Assemble per-symbol features into model-ready datasets.

Produces, with strict no-look-ahead controls:
  * ``X_seq``  : (n_samples, lookback, n_features) sequences for the CNN-BiLSTM
  * ``X_flat`` : (n_samples, n_features) last-step features for XGBoost
  * ``y``      : (n_samples,) forward ``horizon``-day return (regression target)
  * ``symbols``/``dates`` : index arrays mapping each sample back to (symbol, date)

Normalisation uses a trailing rolling z-score (data up to and including the
current bar only), so no future information leaks into a sample's features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .fracdiff import frac_diff_ffd
from .technical import build_technical_features

NORM_WINDOW = 252


@dataclass
class Dataset:
    X_seq: np.ndarray
    X_flat: np.ndarray
    y: np.ndarray
    symbols: np.ndarray
    dates: np.ndarray
    feature_names: List[str]

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(
            X_seq=self.X_seq[mask],
            X_flat=self.X_flat[mask],
            y=self.y[mask],
            symbols=self.symbols[mask],
            dates=self.dates[mask],
            feature_names=self.feature_names,
        )


def build_symbol_features(
    df: pd.DataFrame,
    index_close: Optional[pd.Series] = None,
    frac_d: float = 0.4,
) -> pd.DataFrame:
    """Return a date-indexed feature frame for one symbol (pre-normalisation)."""
    feats = build_technical_features(df).set_index("date")
    close = df.set_index("date")["close"]

    # Fractionally-differenced log price preserves memory while being stationary.
    feats["fracdiff_logp"] = frac_diff_ffd(np.log(close), d=frac_d)

    # Relative strength vs the (equal-weight) index over 20 sessions.
    if index_close is not None and not index_close.empty:
        idx = index_close.reindex(feats.index).ffill()
        rel = (close / close.shift(20)) / (idx / idx.shift(20))
        feats["rel_strength_20"] = rel - 1
        
    # Carry over macroeconomic and NVDR features if present
    for col in ["THB_USD", "US_10Y_YIELD", "nvdr_net_value", "foreign_net_value"]:
        if col in df.columns:
            feats[col] = df.set_index("date")[col]

    return feats.replace([np.inf, -np.inf], np.nan)


def _rolling_zscore(feats: pd.DataFrame, window: int = NORM_WINDOW) -> pd.DataFrame:
    mean = feats.rolling(window, min_periods=window // 2).mean()
    std = feats.rolling(window, min_periods=window // 2).std()
    z = (feats - mean) / std.replace(0, np.nan)
    return z.fillna(0.0).clip(-8, 8)


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Forward ``horizon``-day simple return, date-indexed (NaN at the tail)."""
    close = df.sort_values("date").set_index("date")["close"]
    return close.shift(-horizon) / close - 1


def build_dataset(
    ohlcv: Dict[str, pd.DataFrame],
    settings: Optional[Settings] = None,
    index_close: Optional[pd.Series] = None,
    normalize: bool = True,
    min_history: int = 320,
) -> Dataset:
    """Build a training dataset across the whole universe."""
    settings = settings or get_settings()
    lookback, horizon = settings.lookback, settings.horizon

    feature_names: Optional[List[str]] = None
    prepared: Dict[str, pd.DataFrame] = {}
    labels: Dict[str, pd.Series] = {}

    for symbol, df in ohlcv.items():
        if df is None or len(df) < min_history:
            continue
        feats = build_symbol_features(df, index_close)
        if normalize:
            feats = _rolling_zscore(feats)
        if feature_names is None:
            feature_names = list(feats.columns)
        feats = feats.reindex(columns=feature_names)
        prepared[symbol] = feats
        labels[symbol] = forward_return(df, horizon).reindex(feats.index)

    if feature_names is None:
        raise ValueError("No symbol had enough history to build features.")

    X_seq, X_flat, y, syms, dates = [], [], [], [], []
    for symbol, feats in prepared.items():
        arr = feats.to_numpy(dtype=float)
        lab = labels[symbol].to_numpy(dtype=float)
        idx = feats.index.to_numpy()
        for i in range(lookback - 1, len(feats)):
            window = arr[i - lookback + 1 : i + 1]
            target = lab[i]
            if np.isnan(window).any() or np.isnan(target):
                continue
            X_seq.append(window)
            X_flat.append(arr[i])
            y.append(target)
            syms.append(symbol)
            dates.append(idx[i])

    if not y:
        raise ValueError("No valid samples after windowing — check history length / lookback.")

    return Dataset(
        X_seq=np.asarray(X_seq, dtype=np.float32),
        X_flat=np.asarray(X_flat, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        symbols=np.asarray(syms),
        dates=np.asarray(dates, dtype="datetime64[ns]"),
        feature_names=feature_names,
    )


def build_inference_matrix(
    ohlcv: Dict[str, pd.DataFrame],
    settings: Optional[Settings] = None,
    index_close: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    normalize: bool = True,
):
    """Build the latest-bar feature window per symbol for live prediction.

    Returns ``(X_seq, X_flat, symbols, dates, feature_names)``. Each symbol
    contributes its most recent ``lookback`` window (target is unknown / future).
    """
    settings = settings or get_settings()
    lookback = settings.lookback

    X_seq, X_flat, syms, dates = [], [], [], []
    resolved_names = feature_names

    for symbol, df in ohlcv.items():
        if df is None or len(df) < lookback + NORM_WINDOW:
            continue
        feats = build_symbol_features(df, index_close)
        if normalize:
            feats = _rolling_zscore(feats)
        if resolved_names is None:
            resolved_names = list(feats.columns)
        feats = feats.reindex(columns=resolved_names).dropna()
        if len(feats) < lookback:
            continue
        window = feats.to_numpy(dtype=float)[-lookback:]
        if np.isnan(window).any():
            continue
        X_seq.append(window)
        X_flat.append(window[-1])
        syms.append(symbol)
        dates.append(feats.index[-1])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(X_flat, dtype=np.float32),
        np.asarray(syms),
        np.asarray(dates, dtype="datetime64[ns]"),
        resolved_names,
    )
