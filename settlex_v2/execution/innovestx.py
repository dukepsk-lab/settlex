"""Auto-execute orders via InnovestX TradingView Webhook."""
import logging
import requests
from typing import Dict, List

from ..config import InnovestXConfig

def execute_orders(rebalance: Dict, config: InnovestXConfig, signal_date: str) -> List[dict]:
    """Loop through rebalance dict and send HTTP POST for each order."""
    if not config.is_complete:
        logging.warning("InnovestX config incomplete. Skipping execution.")
        return []

    responses = []
    sells = []
    buys = []
    
    # Collect all sells and buys
    for r in rebalance.get("sell", []):
        if r.get("prev_shares"):
            sells.append({"symbol": r["symbol"], "qty": r["prev_shares"]})

    for r in rebalance.get("hold", []):
        sd = r.get("shares_delta")
        if sd is not None and sd != 0:
            if sd < 0:
                sells.append({"symbol": r["symbol"], "qty": abs(sd)})
            else:
                buys.append({"symbol": r["symbol"], "qty": sd})

    for r in rebalance.get("buy", []):
        if r.get("shares"):
            buys.append({"symbol": r["symbol"], "qty": r["shares"]})

    # Execute Sells first to free up purchasing power
    for s in sells:
        responses.append(_send_order("Sell", s["symbol"], s["qty"], config, signal_date))
        
    # Execute Buys
    for b in buys:
        responses.append(_send_order("Buy", b["symbol"], b["qty"], config, signal_date))

    return responses

def _send_order(side: str, symbol: str, quantity: int, config: InnovestXConfig, signal_date: str) -> dict:
    if quantity <= 0:
        return {}
        
    payload = {
        "ticker": symbol,
        "side": side,
        "quantity": quantity,
        "order_type": "MP-MTL", 
        "comment": f"Auto SETTLEX {signal_date} - {side} {quantity} {symbol}",
        "api_secret": config.api_secret
    }
    
    headers = {
        "User-Agent": "TradingView",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        resp = requests.post(config.endpoint, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        logging.info(f"InnovestX {side} {quantity} {symbol}: {resp.status_code}")
        return {"symbol": symbol, "side": side, "status": resp.status_code, "text": resp.text}
    except Exception as e:
        logging.error(f"InnovestX order failed for {symbol}: {e}")
        return {"symbol": symbol, "side": side, "error": str(e)}
