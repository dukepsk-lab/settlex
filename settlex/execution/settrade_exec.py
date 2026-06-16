from __future__ import annotations
import logging
from typing import Dict, Any, List

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
        responses.append(_send_order(equity, market, 'Sell', s['symbol'], s['qty']))
        
    for b in buys:
        responses.append(_send_order(equity, market, 'Buy', b['symbol'], b['qty']))

    return responses

def _send_order(equity, market, side: str, symbol: str, quantity: int) -> dict:
    if quantity <= 0:
        return {}
        
    try:
        quote = market.get_quote_symbol(symbol)
        ref_price = quote.get('last') or quote.get('prior')
        
        if not ref_price:
            return {'symbol': symbol, 'side': side, 'status': 'Failed', 'error': 'Could not get reference price'}
            
        price = float(ref_price)
        
        import os
        pin = os.getenv('SETTRADE_PIN', '000000')
        
        order = equity.place_order(
            symbol=symbol,
            price=price,
            volume=quantity,
            side=side,
            pin=pin
        )
        
        status = 'Success'
        order_no = order.get('orderNo', 'Unknown')
        return {'symbol': symbol, 'side': side, 'status': f'{status} (No: {order_no})'}
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f'Settrade {side} {quantity} {symbol}: {error_msg}')
        return {'symbol': symbol, 'side': side, 'status': 'Failed', 'error': error_msg}
