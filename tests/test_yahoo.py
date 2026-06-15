"""Unit tests for the Yahoo Finance normaliser (no network required)."""
import pandas as pd

from settlex.data.settrade_client import OHLCV_COLUMNS
from settlex.data.yahoo_client import normalize_yahoo


def _yf_multiindex_frame() -> pd.DataFrame:
    """Mimic yfinance single-ticker output: MultiIndex (field, ticker) columns."""
    idx = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04"], name="Date"
    )
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["PTT.BK"]]
    )
    data = [
        [34.0, 34.5, 33.5, 34.25, 10_000_000],
        [34.25, 35.0, 34.0, 34.75, 12_000_000],
        [34.75, 35.5, 34.5, 35.25, 9_000_000],
    ]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_normalize_multiindex_columns():
    out = normalize_yahoo(_yf_multiindex_frame())
    assert list(out.columns) == OHLCV_COLUMNS
    assert len(out) == 3
    assert out["close"].iloc[-1] == 35.25
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    # dates sorted ascending, tz-naive, normalised to midnight
    assert out["date"].is_monotonic_increasing
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_normalize_flat_columns_and_adj_close_fallback():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Adj Close": [10.25, 11.25],  # no plain "Close" -> fallback
            "Volume": [1_000, 2_000],
        },
        index=idx,
    )
    out = normalize_yahoo(df)
    assert list(out.columns) == OHLCV_COLUMNS
    assert out["close"].tolist() == [10.25, 11.25]


def test_normalize_empty_returns_schema():
    out = normalize_yahoo(pd.DataFrame())
    assert list(out.columns) == OHLCV_COLUMNS
    assert len(out) == 0
