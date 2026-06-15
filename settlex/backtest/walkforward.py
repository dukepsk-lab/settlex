"""Walk-forward (rolling-window) backtesting.

For each fold the model is retrained on a trailing window and evaluated strictly
out-of-sample on the following window, then the window rolls forward — this
adapts to regime change (e.g. the 2024 Uptick Rule) and, unlike K-fold CV,
respects chronology. An embargo between train and test prevents the overlapping
forward-return label windows from leaking future information.

The out-of-sample predictions then drive the live strategy logic (Top-N
selection + Markowitz weights) to produce a realised return series, benchmarked
against an equal-weight (1/N) basket of the same cross-section.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..data.loader import equal_weight_index_close
from ..features.pipeline import Dataset, build_dataset
from ..models.cnn_bilstm import CNNBiLSTMModel
from ..models.ensemble import Ensemble
from ..models.xgb import XGBModel
from ..portfolio.optimizer import optimize_weights
from ..portfolio.selection import select_top_n
from . import metrics

_MODELS = {"ensemble": Ensemble, "xgboost": XGBModel, "cnn_bilstm": CNNBiLSTMModel}


def _make_model(name: str, dataset: Dataset):
    cls = _MODELS[name]
    if name == "cnn_bilstm":
        return cls(lookback=int(dataset.X_seq.shape[1]))
    return cls()


def generate_oos_predictions(
    dataset: Dataset,
    settings: Settings,
    model_name: str = "ensemble",
    train_years: int = 3,
    test_months: int = 12,
    quick: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Roll the train/test window forward, returning OOS predictions.

    Columns: ``date``, ``symbol``, ``pred`` (predicted h-day return),
    ``fwd_return`` (realised h-day return).
    """
    dates = pd.to_datetime(dataset.dates)
    d_min, d_max = dates.min(), dates.max()
    horizon = settings.horizon
    embargo = pd.Timedelta(days=horizon * 2)

    train_end = pd.Timestamp(start) if start else d_min + pd.DateOffset(years=train_years)
    last = pd.Timestamp(end) if end else d_max
    test_delta = pd.DateOffset(months=test_months)

    rows: List[tuple] = []
    fold = 0
    while train_end < last:
        train_start = train_end - pd.DateOffset(years=train_years)
        test_end = min(train_end + test_delta, last + pd.Timedelta(days=1))
        train_mask = (dates >= train_start) & (dates <= train_end - embargo)
        test_mask = (dates > train_end) & (dates <= test_end)

        if int(train_mask.sum()) < 200 or int(test_mask.sum()) < 1:
            train_end = test_end
            continue

        train_ds = dataset.subset(np.asarray(train_mask))
        model = _make_model(model_name, train_ds)
        model.fit(train_ds, quick=quick, verbose=0)

        test_ds = dataset.subset(np.asarray(test_mask))
        preds = model.predict(test_ds.X_seq, test_ds.X_flat)
        for symbol, date, pred, actual in zip(test_ds.symbols, test_ds.dates, preds, test_ds.y):
            rows.append((pd.Timestamp(date), str(symbol), float(pred), float(actual)))

        fold += 1
        if verbose:
            print(
                f"[fold {fold}] train {train_start.date()}..{(train_end - embargo).date()} "
                f"test {train_end.date()}..{test_end.date()} "
                f"(train={int(train_mask.sum())}, test={int(test_mask.sum())})"
            )
        train_end = test_end

    return pd.DataFrame(rows, columns=["date", "symbol", "pred", "fwd_return"])


def simulate_portfolio(
    preds_df: pd.DataFrame,
    ohlcv: Dict[str, pd.DataFrame],
    settings: Settings,
    cost_bps: float = 15.0,
):
    """Turn OOS predictions into realised strategy & benchmark return series.

    Rebalances every ``horizon`` trading days (non-overlapping holds). Costs are
    charged on turnover (bps of traded weight). The benchmark is the equal-weight
    (1/N) return of the same day's cross-section.
    """
    horizon = settings.horizon
    preds_df = preds_df.sort_values("date")
    unique_dates = sorted(preds_df["date"].unique())
    rebalance_dates = unique_dates[::horizon]

    strat_rets, bench_rets = [], []
    prev_w: Dict[str, float] = {}
    for date in rebalance_dates:
        day = preds_df[preds_df["date"] == date]
        bench = day["fwd_return"].mean()
        bench_rets.append(0.0 if np.isnan(bench) else float(bench))

        selected = select_top_n(day["symbol"].to_numpy(), day["pred"].to_numpy(), top_n=settings.top_n)
        if len(selected) == 0:
            strat_rets.append(0.0)
            prev_w = {}
            continue

        weights = optimize_weights(
            selected,
            {s: ohlcv[s] for s in selected.index if s in ohlcv},
            max_weight=settings.max_weight,
            risk_free_rate=settings.risk_free_rate,
            horizon=horizon,
        )
        fwd_map = dict(zip(day["symbol"], day["fwd_return"]))
        ret = 0.0
        for symbol, weight in weights.items():
            fr = fwd_map.get(symbol, np.nan)
            ret += weight * (0.0 if np.isnan(fr) else fr)

        turnover = sum(abs(weights.get(s, 0.0) - prev_w.get(s, 0.0)) for s in set(weights) | set(prev_w))
        ret -= turnover * cost_bps / 1e4
        strat_rets.append(ret)
        prev_w = dict(weights)

    idx = pd.to_datetime(rebalance_dates)
    return (
        pd.Series(strat_rets, index=idx, name="strategy"),
        pd.Series(bench_rets, index=idx, name="benchmark_1overN"),
    )


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]

    def report(self) -> str:
        def fmt(m: Dict[str, float]) -> str:
            return (
                f"  CAGR        : {m['CAGR']:+.2%}\n"
                f"  Sharpe      : {m['Sharpe']:.2f}\n"
                f"  Sortino     : {m['Sortino']:.2f}\n"
                f"  MaxDrawdown : {m['MaxDrawdown']:.2%}\n"
                f"  ProfitFactor: {m['ProfitFactor']:.2f}\n"
                f"  HitRate     : {m['HitRate']:.2%}\n"
                f"  Periods     : {m['Periods']}"
            )

        ic = self.metrics.get("IC", float("nan"))
        return (
            "=== Walk-forward backtest ===\n"
            f"Out-of-sample predictions: {len(self.predictions)} | "
            f"Spearman IC (pred vs realised): {ic:.3f}\n\n"
            "Strategy (Top-N + Markowitz, long-only):\n"
            f"{fmt(self.metrics)}\n\n"
            "Benchmark (equal-weight 1/N):\n"
            f"{fmt(self.benchmark_metrics)}"
        )


def run_walkforward(
    ohlcv: Dict[str, pd.DataFrame],
    settings: Optional[Settings] = None,
    model_name: str = "ensemble",
    train_years: int = 3,
    test_months: int = 12,
    quick: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cost_bps: float = 15.0,
    verbose: bool = False,
) -> BacktestResult:
    settings = settings or get_settings()
    index_close = equal_weight_index_close(ohlcv)
    dataset = build_dataset(ohlcv, settings, index_close=index_close)

    preds = generate_oos_predictions(
        dataset, settings, model_name, train_years, test_months, quick, start, end, verbose
    )
    if preds.empty:
        raise RuntimeError(
            "Walk-forward produced no out-of-sample predictions; check the date "
            "range against available history / train_years."
        )

    strat, bench = simulate_portfolio(preds, ohlcv, settings, cost_bps)
    ppy = 252 / settings.horizon
    strat_metrics = metrics.summary(strat.to_numpy(), rf=settings.risk_free_rate, periods_per_year=ppy)
    bench_metrics = metrics.summary(bench.to_numpy(), rf=settings.risk_free_rate, periods_per_year=ppy)

    valid = preds[["pred", "fwd_return"]].dropna()
    strat_metrics["IC"] = (
        float(valid.corr(method="spearman").iloc[0, 1]) if len(valid) > 2 else float("nan")
    )

    return BacktestResult(preds, strat, bench, strat_metrics, bench_metrics)
