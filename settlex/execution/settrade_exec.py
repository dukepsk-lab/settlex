from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional

from ..config import SettradeConfig

def execute_orders(rebalance: Dict[str, Any], config: SettradeConfig, signal_date: str) -> List[dict]:
    responses = []
    if not config.is_complete:
        logging.warning('Settrade config incomplete, skipping execution.')
        return responses
        
    try:
        from settrade_v2 import Investor
        investor = Investor(
            app_id=config.app_id,
            app_secret=config.app_secret,
            broker_id=config.broker_id,
            app_code=config.app_code,
            is_auto_queue=False
        )
        equity = investor.Equity(config.account_no)
        market = investor.MarketData()
    except Exception as e:
        logging.error(f'Failed to initialize Settrade SDK: {e}')
        return responses

    sells = []
    buys = []
    for r in rebalance.get('sell', []):
        if r.get('shares'):
            sells.append({'symbol': r['symbol'], 'qty': r['shares']})
            
    for r in rebalance.get('buy', []):
        if r.get('shares'):
            buys.append({'symbol': r['symbol'], 'qty': r['shares']})

    for s in sells:
        responses.extend(_send_order(equity, market, 'Sell', s['symbol'], s['qty']))
        
    for b in buys:
        responses.extend(_send_order(equity, market, 'Buy', b['symbol'], b['qty']))

    return responses

def _send_order(equity, market, side: str, symbol: str, quantity: int) -> list:
    if quantity <= 0:
        return []
        
    responses = []
    board_lot = (quantity // 100) * 100
    odd_lot = quantity % 100
        
    try:
        quote = market.get_quote_symbol(symbol)
        ref_price = quote.get('last') or quote.get('prior')
        
        if not ref_price:
            return [{'symbol': symbol, 'side': side, 'status': 'Failed', 'error': 'Could not get reference price'}]
            
        price = float(ref_price)
        
        import os
        pin = os.getenv('SETTRADE_PIN', '000000')
        
        if board_lot > 0:
            try:
                order = equity.place_order(
                    symbol=symbol,
                    price=price,
                    volume=board_lot,
                    side=side,
                    pin=pin
                )
                order_no = order.get('orderNo', 'Unknown')
                responses.append({'symbol': symbol, 'side': side, 'qty': board_lot, 'status': f'Success (No: {order_no})'})
            except Exception as e:
                error_msg = str(e)
                logging.error(f'Settrade {side} {board_lot} {symbol}: {error_msg}')
                responses.append({'symbol': symbol, 'side': side, 'qty': board_lot, 'status': 'Failed', 'error': error_msg})

        if odd_lot > 0:
            try:
                order = equity.place_order(
                    symbol=symbol,
                    price=price,
                    volume=odd_lot,
                    side=side,
                    pin=pin
                )
                order_no = order.get('orderNo', 'Unknown')
                responses.append({'symbol': symbol, 'side': side, 'qty': odd_lot, 'status': f'Success (No: {order_no})'})
            except Exception as e:
                error_msg = str(e)
                logging.error(f'Settrade {side} {odd_lot} {symbol}: {error_msg}')
                responses.append({'symbol': symbol, 'side': side, 'qty': odd_lot, 'status': 'Failed', 'error': error_msg})
                
        return responses
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f'Settrade {side} {quantity} {symbol}: {error_msg}')
        return [{'symbol': symbol, 'side': side, 'qty': quantity, 'status': 'Failed', 'error': error_msg}]

def fetch_live_portfolio(config: SettradeConfig) -> Optional[Dict]:
    """Fetch live cash balance and equity portfolio from Settrade API."""
    if not config.is_complete:
        return None
    try:
        from settrade_v2 import Investor
        investor = Investor(
            app_id=config.app_id,
            app_secret=config.app_secret,
            broker_id=config.broker_id,
            app_code=config.app_code,
            is_auto_queue=False
        )
        equity = investor.Equity(config.account_no)
        info = equity.get_account_info()
        port = equity.get_portfolios()
        
        cash = info.get("lineAvailable", 0.0)
        positions = {}
        for p in port.get("portfolioList", []):
            sym = p.get("symbol", "").strip()
            shares = p.get("actualVolume", 0)
            if shares > 0 and sym:
                positions[sym] = {
                    "shares": shares,
                    "market_value": p.get("marketValue", 0.0),
                    "average_cost": p.get("averagePrice", 0.0)
                }
        return {"cash": cash, "positions": positions}
    except Exception as e:
        logging.error(f"Failed to fetch live portfolio from Settrade: {e}")
        return None
