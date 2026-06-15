"""Rank model predictions and select the long-only Top-N basket."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def rank_predictions(symbols: Sequence[str], preds: Sequence[float]) -> pd.Series:
    """Return predicted returns as a Series sorted high -> low."""
    return pd.Series(np.asarray(preds, dtype=float), index=list(symbols)).sort_values(ascending=False)


def select_top_n(
    symbols: Sequence[str],
    preds: Sequence[float],
    top_n: int = 5,
    require_positive: bool = True,
) -> pd.Series:
    """Select up to ``top_n`` names with the highest predicted return.

    With ``require_positive`` (default) only positive-expected-return names are
    eligible — the strategy is long-only, so a thin/negative cross-section
    simply yields fewer (or zero) positions, i.e. a partial cash allocation.
    """
    ranked = rank_predictions(symbols, preds)
    if require_positive:
        ranked = ranked[ranked > 0]
    return ranked.head(top_n)
