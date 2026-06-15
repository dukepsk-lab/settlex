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

    # Auto-fetch account equity from Settrade when no override is given.
    if capital is None and not synthetic and settings.settrade.is_complete and settings.settrade.account_no:
        try:
            from ..data.settrade_client import SettradeClient
            fetched = SettradeClient(settings.settrade).get_account_equity()
            if fetched and fetched > 0:
                capital = fetched
        except Exception:
            pass
    capital = settings.capital if capital is None else capital

    symbols = load_universe()
    ohlcv = load_universe_ohlcv(symbols, settings, synthetic=synthetic)
    if not ohlcv:
        raise RuntimeError("No OHLCV data loaded for the universe.")

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
        positions.append(
            {
                "symbol": symbol,
                "weight": w,
                "thb": float(thb.get(symbol, 0.0)),
                "pred_return": float(pred_map.get(symbol, np.nan)),
            }
        )
    result["positions"] = positions
    return result
