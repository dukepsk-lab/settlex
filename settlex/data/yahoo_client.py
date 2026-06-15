"""EOD daily OHLCV from Yahoo Finance via `yfinance` (no credentials).

Thai SET tickers carry the ``.BK`` suffix on Yahoo (e.g. ``PTT`` -> ``PTT.BK``).
This is end-of-day data, perfect for after-close planning of the next trading
day; it is *not* intraday/real-time. Prices are split/dividend-adjusted
(``auto_adjust=True``) so they line up with the model's return features.

The SDK is imported lazily so the rest of settlex (and the synthetic data path
used in tests) works even when `yfinance` is not installed.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

from .settrade_client import OHLCV_COLUMNS


class YahooClient:
    """Drop-in replacement for :class:`SettradeClient` with a ``get_ohlcv`` API."""

    def __init__(self, suffix: str = ".BK"):
        self._suffix = suffix

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a tidy OHLCV frame for ``symbol`` between ``start`` and ``end``."""
        import yfinance as yf  # lazy import

        # Yahoo's `end` is exclusive; bump by a day so today's bar is included.
        end_ts = pd.Timestamp(end) if end else pd.Timestamp(dt.date.today())
        end_excl = (end_ts + pd.Timedelta(days=1)).date().isoformat()

        ticker = f"{symbol}{self._suffix}"
        raw = yf.download(
            ticker,
            start=start,
            end=end_excl,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return normalize_yahoo(raw)


def normalize_yahoo(raw) -> pd.DataFrame:
    """Normalise a `yfinance` download into ``OHLCV_COLUMNS``.

    `yfinance` returns a date-indexed frame whose columns are either flat
    (``Open``/``High``/.../``Volume``) or a ``MultiIndex`` of ``(field, ticker)``
    for single-ticker downloads. Both shapes are handled.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    cols = {str(c).lower(): c for c in df.columns}

    time_col = cols.get("date") or cols.get("datetime") or cols.get("index")
    if time_col is None:
        raise ValueError(f"Unexpected yfinance payload columns: {list(df.columns)}")

    ts = pd.to_datetime(df[time_col], errors="coerce")
    ts = ts.dt.tz_localize(None) if getattr(ts.dt, "tz", None) is not None else ts
    ts = ts.dt.normalize()

    def col(*names: str) -> pd.Series:
        for name in names:
            src = cols.get(name)
            if src is not None:
                return pd.to_numeric(df[src], errors="coerce")
        raise ValueError(f"yfinance payload missing {names}: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "date": ts,
            "open": col("open"),
            "high": col("high"),
            "low": col("low"),
            "close": col("close", "adj close"),
            "volume": col("volume"),
        }
    )
    out = (
        out.dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out[OHLCV_COLUMNS]
