"""Thin wrapper over the Settrade Open API (`settrade-v2`) market-data SDK.

Usage (real credentials required)::

    from settlex.config import get_settings
    from settlex.data.settrade_client import SettradeClient

    client = SettradeClient(get_settings().settrade)
    df = client.get_ohlcv("PTT", start="2015-01-01")

The SDK is imported lazily so the rest of settlex (and the synthetic data
path used in tests) works even when `settrade-v2` is not installed.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

from ..config import SettradeConfig

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class SettradeClient:
    def __init__(self, config: SettradeConfig):
        if not config.is_complete:
            raise ValueError(
                "Settrade credentials incomplete. Set SETTRADE_APP_ID, "
                "SETTRADE_APP_SECRET, SETTRADE_BROKER_ID and SETTRADE_APP_CODE "
                "in your .env, or use the synthetic data loader (--synthetic) "
                "for development."
            )
        self._config = config
        self._investor = None
        self._market = None

    def _get_investor(self):
        if self._investor is None:
            from settrade_v2 import Investor  # lazy import

            self._investor = Investor(
                app_id=self._config.app_id,
                app_secret=self._config.app_secret,
                broker_id=self._config.broker_id,
                app_code=self._config.app_code,
                is_auto_queue=False,
            )
        return self._investor

    def _market_data(self):
        if self._market is None:
            self._market = self._get_investor().MarketData()
        return self._market

    def get_account_equity(self) -> Optional[float]:
        """Return total account equity in THB from Settrade, or None on failure.

        Requires SETTRADE_ACCOUNT_NO to be set. Tries common field names
        returned by ``InvestorEquity.get_account_info()`` in priority order.
        """
        if not self._config.account_no:
            return None
        try:
            eq = self._get_investor().Equity(self._config.account_no)
            info = eq.get_account_info()
            for field_name in (
                "portEquity", "equity", "portValue",
                "totalPortValue", "totalEquity", "portValueWithCredit",
            ):
                val = info.get(field_name)
                if val is not None:
                    return float(val)
            return None
        except Exception:
            return None

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a tidy OHLCV frame for ``symbol`` between ``start`` and ``end``."""
        end = end or dt.date.today().isoformat()
        market = self._market_data()
        raw = market.get_candlestick(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            normalized=True,
        )
        return normalize_candlestick(raw)


def normalize_candlestick(raw) -> pd.DataFrame:
    """Normalise a Settrade candlestick payload into ``OHLCV_COLUMNS``.

    The SDK returns either a DataFrame or a dict of parallel lists keyed by
    ``time``/``open``/``high``/``low``/``close``/``volume``. ``time`` may be a
    unix epoch (seconds) or an ISO timestamp.
    """
    df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    cols = {c.lower(): c for c in df.columns}

    time_col = cols.get("time") or cols.get("datetime") or cols.get("date")
    if time_col is None:
        raise ValueError(f"Unexpected candlestick payload columns: {list(df.columns)}")

    ts = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(df[time_col], errors="coerce")
    ts = ts.dt.tz_localize(None) if getattr(ts.dt, "tz", None) is not None else ts
    ts = ts.dt.normalize()

    def col(name: str) -> pd.Series:
        src = cols.get(name)
        if src is None:
            raise ValueError(f"Candlestick payload missing '{name}': {list(df.columns)}")
        return pd.to_numeric(df[src], errors="coerce")

    out = pd.DataFrame(
        {
            "date": ts,
            "open": col("open"),
            "high": col("high"),
            "low": col("low"),
            "close": col("close"),
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
