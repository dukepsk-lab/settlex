"""Orchestrate the live signal: data -> features -> ensemble -> Top-N -> weights."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..data.loader import equal_weight_index_close, load_universe_ohlcv
from ..features.pipeline import build_inference_matrix
from ..models.ensemble import Ensemble
from ..portfolio.optimizer import allocate_capital, optimize_weights
from ..portfolio.selection import select_top_n
from ..universe import load_universe


def generate_signal(
    settings: Optional[Settings] = None,
    synthetic: bool = False,
    capital: Optional[float] = None,
    model_path=None,
) -> Dict:
    """Produce today's signal as a dict (see :func:`signals.telegram.format_signal`)."""
    settings = settings or get_settings()

    symbols = load_universe()
    ohlcv = load_universe_ohlcv(symbols, settings, synthetic=synthetic)
    if not ohlcv:
        raise RuntimeError("No OHLCV data loaded for the universe.")

    if capital is None and not synthetic:
        from ..data.portfolio import load_manual_portfolio
        live_port = load_manual_portfolio(settings.data_dir)
        
        if live_port is not None:
            fetched = live_port["cash"]
            for sym, p in live_port["positions"].items():
                price = float(ohlcv[sym].iloc[-1]["close"]) if sym in ohlcv and not ohlcv[sym].empty else p.get("market_value", 0) / max(1, p.get("shares", 1))
                fetched += p.get("shares", 0) * price
            if fetched > 0:
                capital = fetched
    capital = settings.capital if capital is None else capital

    index_close = equal_weight_index_close(ohlcv)

    model = Ensemble.load(model_path or settings.ensemble_path)
    X_seq, X_flat, syms, dates, _ = build_inference_matrix(
        ohlcv, settings, index_close=index_close, feature_names=model.feature_names
    )
    if len(syms) == 0:
        raise RuntimeError("No symbols had enough history to build inference features.")

    preds = model.predict(X_seq, X_flat)
    as_of = pd.Timestamp(max(dates)).date().isoformat()

    # Optional: agreement between the two members (rank correlation of predictions).
    agreement = None
    try:
        import warnings

        from scipy.stats import ConstantInputWarning, spearmanr

        members = model.member_predictions(X_seq, X_flat)
        if len(members) == 2 and len(syms) >= 3:
            (a, b) = list(members.values())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                corr = spearmanr(a, b).correlation
            agreement = None if corr is None or np.isnan(corr) else float(corr)
    except Exception:
        agreement = None

    selected = select_top_n(syms, preds, top_n=settings.top_n)
    result: Dict = {
        "date": as_of,
        "horizon": settings.horizon,
        "capital": capital,
        "model_agreement": agreement,
        "positions": [],
    }
    if len(selected) == 0:
        return result

    weights = optimize_weights(
        selected,
        {s: ohlcv[s] for s in selected.index if s in ohlcv},
        max_weight=settings.max_weight,
        risk_free_rate=settings.risk_free_rate,
        horizon=settings.horizon,
    )
    thb = allocate_capital(weights, capital)
    pred_map = dict(zip(syms, preds))

    positions: List[Dict] = []
    for symbol in weights.sort_values(ascending=False).index:
        w = float(weights[symbol])
        if w <= 0:
            continue
            
        target_thb = float(thb.get(symbol, 0.0))
        price = None
        shares = None
        actual_thb = target_thb
        
        df = ohlcv.get(symbol)
        if df is not None and not df.empty:
            price = float(df.iloc[-1]["close"])
            if price > 0:
                import math
                shares = math.floor(target_thb / price)
                actual_thb = shares * price

        positions.append(
            {
                "symbol": symbol,
                "weight": w,
                "thb": actual_thb,
                "price": price,
                "shares": shares,
                "pred_return": float(pred_map.get(symbol, np.nan)),
            }
        )
    result["positions"] = positions
    return result
