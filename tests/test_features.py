import numpy as np

from settlex.data.loader import synthetic_ohlcv
from settlex.features.technical import atr, build_technical_features, rsi


def _df():
    return synthetic_ohlcv("TEST", "2020-01-01")


def test_rsi_bounded_0_100():
    r = rsi(_df()["close"]).dropna()
    assert r.min() >= 0.0
    assert r.max() <= 100.0


def test_atr_positive():
    df = _df()
    a = atr(df["high"], df["low"], df["close"]).dropna()
    assert (a > 0).all()


def test_build_features_shape_and_columns():
    df = _df()
    feats = build_technical_features(df)
    assert len(feats) == len(df)
    assert "date" in feats.columns
    for col in ["rsi_14", "macd", "macd_hist", "adx_14", "atr_pct", "bb_pctb", "obv_chg_5", "dist_high_20"]:
        assert col in feats.columns
