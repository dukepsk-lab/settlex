"""Tests for the rebalance (turnover) diff between two signals — pure, no network."""
from settlex.rebalance import (
    compute_rebalance,
    has_changes,
    rebalance_summary_text,
)


def _sig(date, positions):
    return {"date": date, "capital": 1_000_000, "positions": positions}


def _yesterday():
    return _sig(
        "2026-06-12",
        [
            {"symbol": "PTT", "weight": 0.30, "thb": 300_000, "pred_return": 0.02},
            {"symbol": "AOT", "weight": 0.40, "thb": 400_000, "pred_return": 0.03},
            {"symbol": "BBL", "weight": 0.30, "thb": 300_000, "pred_return": 0.01},
        ],
    )


def _today():
    return _sig(
        "2026-06-15",
        [
            {"symbol": "AOT", "weight": 0.40, "thb": 400_000, "pred_return": 0.03},   # hold, unchanged
            {"symbol": "PTT", "weight": 0.20, "thb": 200_000, "pred_return": 0.01},   # hold, trimmed
            {"symbol": "CPALL", "weight": 0.40, "thb": 400_000, "pred_return": 0.04}, # new buy
            # BBL dropped -> sell
        ],
    )


def test_classifies_sell_hold_buy():
    reb = compute_rebalance(_yesterday(), _today())
    assert {r["symbol"] for r in reb["sell"]} == {"BBL"}
    assert {r["symbol"] for r in reb["hold"]} == {"PTT", "AOT"}
    assert {r["symbol"] for r in reb["buy"]} == {"CPALL"}
    assert reb["has_prev"] is True
    assert has_changes(reb) is True


def test_hold_deltas():
    reb = compute_rebalance(_yesterday(), _today())
    holds = {r["symbol"]: r for r in reb["hold"]}
    # PTT trimmed 30% -> 20%
    assert round(holds["PTT"]["weight_delta"], 4) == -0.10
    assert round(holds["PTT"]["thb_delta"], 0) == -100_000
    # AOT unchanged
    assert round(holds["AOT"]["weight_delta"], 6) == 0.0
    assert round(holds["AOT"]["thb_delta"], 0) == 0


def test_turnover_is_one_way_fraction():
    reb = compute_rebalance(_yesterday(), _today())
    # gross |dw|: BBL 0.30 + PTT 0.10 + AOT 0 + CPALL 0.40 = 0.80 -> one-way 0.40
    assert round(reb["turnover"], 4) == 0.40


def test_no_previous_signal_is_all_buys():
    reb = compute_rebalance(None, _today())
    assert reb["has_prev"] is False
    assert reb["sell"] == [] and reb["hold"] == []
    assert {r["symbol"] for r in reb["buy"]} == {"AOT", "PTT", "CPALL"}


def test_identical_portfolio_has_no_changes():
    reb = compute_rebalance(_yesterday(), _yesterday())
    assert reb["sell"] == [] and reb["buy"] == []
    assert all(abs(h["weight_delta"]) < 1e-9 for h in reb["hold"])
    assert has_changes(reb) is False
    # Every line is a "hold, unchanged" note; no sell/buy verbs appear.
    text = rebalance_summary_text(reb)
    assert "คงน้ำหนัก" in text
    assert "ขายปิด" not in text and "ซื้อใหม่" not in text


def test_summary_text_mentions_each_action():
    text = rebalance_summary_text(compute_rebalance(_yesterday(), _today()))
    assert "ขายปิด BBL" in text
    assert "ซื้อใหม่ CPALL" in text
    assert "PTT" in text  # trimmed hold appears
