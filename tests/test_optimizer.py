import numpy as np
import pandas as pd

from settlex.portfolio.optimizer import allocate_capital, optimize_weights
from settlex.portfolio.selection import select_top_n


def _price_history(symbols, n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    hist = {}
    for i, s in enumerate(symbols):
        rets = rng.normal(0.0005 * (i + 1), 0.02, n)
        close = 100 * np.exp(np.cumsum(rets))
        hist[s] = pd.DataFrame(
            {"date": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}
        )
    return hist


def test_select_top_n_positive_only():
    sel = select_top_n(["A", "B", "C"], [0.03, -0.01, 0.02], top_n=5)
    assert list(sel.index) == ["A", "C"]


def test_weights_obey_constraints_five_assets():
    syms = ["A", "B", "C", "D", "E"]
    sel = pd.Series([0.05, 0.04, 0.03, 0.02, 0.01], index=syms)
    w = optimize_weights(sel, _price_history(syms), max_weight=0.30, horizon=5)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()
    assert w.max() <= 0.30 + 1e-6


def test_weights_three_assets_cap_lifted_for_feasibility():
    syms = ["A", "B", "C"]
    sel = pd.Series([0.05, 0.03, 0.01], index=syms)
    w = optimize_weights(sel, _price_history(syms), max_weight=0.30, horizon=5)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_single_asset_gets_full_weight():
    w = optimize_weights(pd.Series([0.05], index=["A"]), {}, horizon=5)
    assert w.to_dict() == {"A": 1.0}


def test_empty_selection_returns_empty():
    w = optimize_weights(pd.Series(dtype=float), {}, horizon=5)
    assert w.empty


def test_allocate_capital_sums_to_capital():
    thb = allocate_capital(pd.Series({"A": 0.5, "B": 0.5}), 1_000_000)
    assert thb.sum() == 1_000_000
