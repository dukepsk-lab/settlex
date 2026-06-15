"""Tests for prediction-accuracy evaluation (pure, no network)."""
import pandas as pd

from settlex.evaluation import evaluate_predictions


def _frame(dates, closes):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": closes})


def _ohlcv():
    dates = ["2026-06-10", "2026-06-11", "2026-06-12"]
    return {
        "AAA": _frame(dates, [100.0, 102.0, 105.0]),  # +5% from signal date
        "BBB": _frame(dates, [50.0, 49.0, 48.0]),      # -4%
        "CCC": _frame(dates, [20.0, 20.0, 20.5]),      # +2.5% (not a pick)
    }


def _prev_signal():
    return {
        "date": "2026-06-10",
        "horizon": 5,
        "capital": 1_000_000,
        "positions": [
            {"symbol": "AAA", "weight": 0.6, "thb": 600_000, "pred_return": 0.01},
            {"symbol": "BBB", "weight": 0.4, "thb": 400_000, "pred_return": 0.008},
        ],
    }


def test_realized_returns_and_hit_rate():
    ev = evaluate_predictions(_prev_signal(), _ohlcv())
    by_sym = {r["symbol"]: r for r in ev["positions"]}
    assert round(by_sym["AAA"]["realized_return"], 4) == 0.05
    assert round(by_sym["BBB"]["realized_return"], 4) == -0.04
    # AAA predicted up and went up (hit); BBB predicted up but fell (miss) -> 50%
    assert by_sym["AAA"]["correct"] is True
    assert by_sym["BBB"]["correct"] is False
    assert ev["hit_rate"] == 0.5


def test_portfolio_vs_benchmark():
    ev = evaluate_predictions(_prev_signal(), _ohlcv())
    # weighted: 0.6*0.05 + 0.4*(-0.04) = 0.03 - 0.016 = 0.014
    assert round(ev["portfolio_return"], 4) == 0.014
    # benchmark = mean(0.05, -0.04, 0.025) = 0.011667
    assert round(ev["benchmark_return"], 4) == 0.0117
    assert round(ev["excess_return"], 4) == round(0.014 - 0.011667, 4)


def test_days_elapsed_and_eval_date():
    ev = evaluate_predictions(_prev_signal(), _ohlcv())
    assert ev["eval_date"] == "2026-06-12"
    assert ev["days_elapsed"] == 2  # Wed 10th -> Fri 12th = 2 business days


def test_missing_symbol_is_skipped():
    ohlcv = {"AAA": _ohlcv()["AAA"]}  # BBB absent
    ev = evaluate_predictions(_prev_signal(), ohlcv)
    syms = {r["symbol"] for r in ev["positions"]}
    assert syms == {"AAA"}
    # only AAA contributes; weighted by its own weight -> equals its return
    assert round(ev["portfolio_return"], 2) == 0.05
