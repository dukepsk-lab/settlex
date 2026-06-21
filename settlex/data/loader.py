"""Load SET50 OHLCV with on-disk parquet caching and a synthetic fallback.

The synthetic generator produces deterministic price series (seeded per
symbol) so the entire pipeline, backtest and tests can run without Settrade
credentials. Real runs hit the Settrade Open API via :class:`SettradeClient`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..universe import load_universe
from .settrade_client import SettradeClient, OHLCV_COLUMNS


def _cache_path(settings: Settings, symbol: str) -> Path:
    # Append an underscore to avoid Windows reserved filename issues (e.g. COM7.parquet -> COM7_.parquet)
    # Windows treats COM1-9 as reserved device names even with an extension.
    return settings.data_dir / "ohlcv" / f"{symbol}_.parquet"


def make_data_client(settings: Settings):
    """Return a data client (exposing ``get_ohlcv``) for the configured source.

    ``SETTLEX_DATA_SOURCE`` selects the backend: ``yahoo`` (default, free, no
    credentials — EOD daily data, ideal for after-close planning) or
    ``settrade`` (the broker Open API).
    """
    source = (settings.data_source or "yahoo").lower()
    if source == "settrade":
        return SettradeClient(settings.settrade)
    from .yahoo_client import YahooClient

    return YahooClient()


def _stable_seed(symbol: str) -> int:
    digest = hashlib.sha256(symbol.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def synthetic_ohlcv(symbol: str, start: str, periods: Optional[int] = None) -> pd.DataFrame:
    """Deterministic GBM-style OHLCV with mild momentum (learnable signal)."""
    dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(dt.date.today()))
    if periods is not None:
        dates = dates[:periods]
    n = len(dates)
    rng = np.random.default_rng(_stable_seed(symbol))
    mu, sigma = 0.0004, 0.018
    base = 10.0 + (_stable_seed(symbol) % 90)
    shocks = rng.normal(mu, sigma, n)
    # add autocorrelation so momentum features carry information
    momentum = pd.Series(shocks).ewm(span=5).mean().to_numpy()
    rets = momentum + rng.normal(0.0, sigma / 2, n)
    close = base * np.exp(np.cumsum(rets))
    span = np.abs(rng.normal(0.0, 0.01, n))
    high = close * (1 + span)
    low = close * (1 - span)
    open_ = close * (1 + rng.normal(0.0, 0.005, n))
    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "date": dates.normalize(),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )[OHLCV_COLUMNS]


def load_symbol(
    symbol: str,
    settings: Optional[Settings] = None,
    refresh: bool = False,
    synthetic: bool = False,
    client: Optional[SettradeClient] = None,
) -> pd.DataFrame:
    """Load one symbol's OHLCV, using the parquet cache when available."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    path = _cache_path(settings, symbol)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    if synthetic:
        df = synthetic_ohlcv(symbol, settings.history_start)
    else:
        client = client or make_data_client(settings)
        df = client.get_ohlcv(symbol, start=settings.history_start)

    df.to_parquet(path, index=False)
    return df


def load_universe_ohlcv(
    symbols: Optional[List[str]] = None,
    settings: Optional[Settings] = None,
    refresh: bool = False,
    synthetic: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV for every symbol in the universe (dict symbol -> frame)."""
    settings = settings or get_settings()
    symbols = symbols or load_universe()

    client = None
    if not synthetic:
        need_fetch = refresh or any(not _cache_path(settings, s).exists() for s in symbols)
        if need_fetch:
            client = make_data_client(settings)

    out: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            out[symbol] = load_symbol(symbol, settings, refresh, synthetic, client)
        except Exception as exc:  # noqa: BLE001 - keep going on per-symbol failures
            print(f"[warn] failed to load {symbol}: {exc}")
    return out


def equal_weight_index_close(ohlcv: Dict[str, pd.DataFrame]) -> pd.Series:
    """Build an equal-weight price index from the universe.

    Used as a stable, always-available proxy for SET50 in relative-strength
    features (avoids depending on a separate index symbol feed).
    """
    closes = {}
    for symbol, df in ohlcv.items():
        if df is None or df.empty:
            continue
        closes[symbol] = df.set_index("date")["close"]
    if not closes:
        return pd.Series(dtype=float)
    price = pd.DataFrame(closes).sort_index()
    daily = price.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    return (1 + daily).cumprod()
