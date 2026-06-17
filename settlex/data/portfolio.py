"""Manual portfolio loading."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional


def load_manual_portfolio(data_dir: Path) -> Optional[Dict]:
    """Load manual portfolio override if it exists.
    
    Expected format:
    {
      "cash": 945.22,
      "positions": {
        "BGRIM": { "shares": 100, "market_value": 1660.0, "average_cost": 16.50 },
        "PTT": { "shares": 50, "market_value": 1750.0, "average_cost": 34.00 }
      }
    }
    """
    path = data_dir / "portfolio.json"
    if not path.exists():
        return None
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        cash = float(data.get("cash", 0.0))
        positions = {}
        for sym, p in data.get("positions", {}).items():
            positions[sym] = {
                "shares": int(p.get("shares", 0)),
                "market_value": float(p.get("market_value", 0.0)),
                "average_cost": float(p.get("average_cost", 0.0))
            }
        
        return {"cash": cash, "positions": positions}
    except Exception as e:
        logging.error(f"Failed to load manual portfolio.json: {e}")
        return None
