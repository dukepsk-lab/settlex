from settlex.backtest import metrics


def test_max_drawdown_monotonic_up_is_zero():
    assert abs(metrics.max_drawdown([0.01] * 20)) < 1e-12


def test_max_drawdown_negative_when_decline():
    assert metrics.max_drawdown([0.1, -0.5, 0.05]) < 0


def test_profit_factor_no_losses_is_inf():
    assert metrics.profit_factor([0.01, 0.02, 0.0]) == float("inf")


def test_sharpe_zero_variance_is_zero():
    assert metrics.sharpe([0.01, 0.01, 0.01]) == 0.0


def test_hit_rate_counts_positive():
    assert metrics.hit_rate([0.1, -0.1, 0.2, 0.0]) == 0.5


def test_cagr_positive_for_gains():
    assert metrics.cagr([0.01] * 252, periods_per_year=252) > 0


def test_summary_has_all_keys():
    s = metrics.summary([0.01, -0.02, 0.03])
    for key in ["CAGR", "Sharpe", "Sortino", "MaxDrawdown", "ProfitFactor", "HitRate", "Periods"]:
        assert key in s
