import numpy as np

from settlex.config import Settings
from settlex.data.loader import equal_weight_index_close, synthetic_ohlcv
from settlex.features.pipeline import (
    _rolling_zscore,
    build_dataset,
    build_inference_matrix,
    build_symbol_features,
)


def _universe(symbols=("AAA", "BBB", "CCC")):
    return {s: synthetic_ohlcv(s, "2018-01-01") for s in symbols}


def _settings():
    s = Settings()
    s.lookback = 30
    s.horizon = 5
    return s


def test_build_dataset_shapes_and_no_nan():
    settings = _settings()
    data = _universe()
    ds = build_dataset(data, settings, index_close=equal_weight_index_close(data))
    assert ds.X_seq.shape[1] == settings.lookback
    assert ds.X_seq.shape[2] == len(ds.feature_names)
    assert ds.X_flat.shape[0] == ds.X_seq.shape[0] == len(ds.y)
    assert not np.isnan(ds.X_seq).any()
    assert not np.isnan(ds.y).any()


def test_inference_matrix_one_window_per_symbol():
    settings = _settings()
    data = _universe()
    X_seq, X_flat, syms, dates, names = build_inference_matrix(
        data, settings, index_close=equal_weight_index_close(data)
    )
    assert len(syms) == len(set(syms))
    assert X_seq.shape[1] == settings.lookback
    assert X_seq.shape[2] == len(names)


def test_rolling_zscore_has_no_lookahead():
    """The normalised value at row k must not change when future rows are removed."""
    feats = build_symbol_features(synthetic_ohlcv("ZZZ", "2019-01-01"))
    k = 400
    full = _rolling_zscore(feats).iloc[k - 1]
    truncated = _rolling_zscore(feats.iloc[:k]).iloc[k - 1]
    mask = ~(full.isna() | truncated.isna())
    assert np.allclose(full[mask].to_numpy(), truncated[mask].to_numpy(), atol=1e-8)
