"""SET50 constituent universe.

The SET50 index is rebalanced semi-annually by the Stock Exchange of Thailand.
This module ships a recent snapshot for live signal generation. For bias-free
backtesting you should supply point-in-time constituents via a file referenced
by the ``SETTLEX_UNIVERSE_FILE`` env var (see :func:`load_universe`); otherwise
backtest results are subject to survivorship bias (the research report flags
this explicitly).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Snapshot of SET50 constituents (maintain via SET index announcements).
SET50_SYMBOLS: List[str] = [
    "ADVANC", "AOT", "AWC", "BANPU", "BBL", "BDMS", "BEM", "BGRIM", "BH",
    "BTS", "CBG", "CENTEL", "COM7", "CPALL", "CPF", "CPN", "CRC", "DELTA",
    "EA", "EGCO", "GLOBAL", "GPSC", "GULF", "HMPRO", "IVL", "KBANK",
    "KCE", "KTB", "KTC", "LH", "MINT", "MTC", "OR", "OSP", "PTT", "PTTEP",
    "PTTGC", "RATCH", "SAWAD", "SCB", "SCC", "SCGP", "TIDLOR", "TISCO", "TLI",
    "TOP", "TRUE", "TTB", "WHA",
]


def load_universe(path: Optional[str] = None) -> List[str]:
    """Return the trading universe.

    If ``path`` (or ``$SETTLEX_UNIVERSE_FILE``) points at a newline-delimited
    file of symbols, that overrides the built-in snapshot. Lines starting with
    ``#`` are ignored.
    """
    path = path or os.getenv("SETTLEX_UNIVERSE_FILE")
    if path and Path(path).exists():
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        symbols = [ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#")]
        if symbols:
            return symbols
    return list(SET50_SYMBOLS)
