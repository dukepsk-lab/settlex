"""Technical indicators implemented natively in pandas/numpy.

Kept dependency-free (no TA-Lib / `ta`) so the feature set is transparent,
testable and reproducible. Each indicator follows the conventional Wilder /
standard definitions used in the SET50 literature cited in the research report
(RSI, MACD, ADX, ATR, Bollinger, OBV, VWAP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = (upper - lower) / ma
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pct_b, width


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    pv = typical * volume
    return pv.rolling(period).sum() / volume.rolling(period).sum()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_ = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed frame of technical features from an OHLCV frame."""
    df = df.sort_values("date").reset_index(drop=True)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    feats = pd.DataFrame(index=df.index)
    # Momentum / returns
    feats["ret_1"] = close.pct_change(1)
    feats["ret_5"] = close.pct_change(5)
    feats["ret_10"] = close.pct_change(10)
    feats["rsi_14"] = rsi(close, 14)
    macd_line, macd_sig, macd_hist = macd(close)
    feats["macd"] = macd_line
    feats["macd_signal"] = macd_sig
    feats["macd_hist"] = macd_hist
    feats["adx_14"] = adx(high, low, close, 14)

    # Volatility
    feats["atr_pct"] = atr(high, low, close, 14) / close
    pct_b, bb_width = bollinger(close)
    feats["bb_pctb"] = pct_b
    feats["bb_width"] = bb_width

    # Volume / price dynamics
    obv_series = obv(close, vol)
    feats["obv_chg_5"] = obv_series.pct_change(5)
    feats["vwap_dev"] = close / vwap(high, low, close, vol) - 1
    feats["vol_z_20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

    # Support / resistance proximity
    feats["dist_high_20"] = close / close.rolling(20).max() - 1
    feats["dist_low_20"] = close / close.rolling(20).min() - 1

    feats["date"] = df["date"].to_numpy()
    return feats
