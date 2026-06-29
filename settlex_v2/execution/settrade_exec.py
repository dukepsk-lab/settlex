from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
import time

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

    sell_responses = []
    for s in sells:
        sell_responses.extend(_send_order(equity, market, 'Sell', s['symbol'], s['qty'], config.pin))
        
    responses.extend(sell_responses)
    
    # Wait & Check Status for Sells
    sell_order_nos = [
        resp.get('order_no') for resp in sell_responses 
        if resp.get('status', '').startswith('Success') and resp.get('order_no') and resp.get('order_no') != 'Unknown'
    ]
    
    if sell_order_nos:
        logging.info(f"Waiting for {len(sell_order_nos)} sell order(s) to match...")
        max_retries = 20
        for i in range(max_retries):
            all_matched = True
            try:
                orders_res = equity.get_orders()
                # Handle both list and dict returns for safety
                orders_list = orders_res if isinstance(orders_res, list) else orders_res.get('orderList', [])
                if not isinstance(orders_list, list):
                    orders_list = []
                    
                for order_no in sell_order_nos:
                    matched_order = next((o for o in orders_list if str(o.get('orderNo')) == str(order_no)), None)
                    if matched_order:
                        vol = matched_order.get('vol', matched_order.get('volume', 0))
                        match_vol = matched_order.get('matchVol', matched_order.get('matchVolume', 0))
                        status = str(matched_order.get('status', '')).upper()
                        
                        if vol > 0 and match_vol >= vol:
                            continue # Fully matched
                        if status in ['M', 'MATCHED', 'FULLY MATCHED']:
                            continue
                            
                        all_matched = False
                        break
                    else:
                        # If order not found in active list, it might be closed/matched, but we can't be 100% sure. 
                        # We'll assume it's not matched if we just submitted it, or matched if it disappeared.
                        # Settrade usually keeps orders in get_orders for the day.
                        pass
                        
            except Exception as e:
                logging.warning(f"Error checking order status: {e}")
                all_matched = False
                
            if all_matched:
                logging.info("All sell orders matched successfully!")
                break
                
            logging.info(f"Sell orders not fully matched yet, waiting... ({i+1}/{max_retries})")
            time.sleep(3)

    # VWAP Slicing & Hard Stop-Loss (5%)
    vwap_chunks = 5
    buy_interval_sec = 60 * 60  # 1 hour between chunks
    
    # Pre-calculate chunks for each buy order
    buy_plans = []
    for b in buys:
        sym = b['symbol']
        total_qty = b['qty']
        board_lot = (total_qty // 100) * 100
        odd_lot = total_qty % 100
        
        chunk_size = ((board_lot // vwap_chunks) // 100) * 100
        chunks = [chunk_size] * vwap_chunks
        
        remainder = board_lot - sum(chunks)
        if remainder > 0:
            chunks[0] += remainder
            
        chunks[-1] += odd_lot
        buy_plans.append({'symbol': sym, 'chunks': chunks})

    if buy_plans:
        logging.info("Starting VWAP Buy Execution (alive all day)...")
        for i in range(vwap_chunks):
            logging.info(f"VWAP Chunk {i+1}/{vwap_chunks}")
            chunk_responses = []
            
            for plan in buy_plans:
                qty = plan['chunks'][i]
                if qty > 0:
                    res = _send_order(equity, market, 'Buy', plan['symbol'], qty, config.pin)
                    chunk_responses.extend(res)
                    
                    # 5% Hard Stop-Loss
                    for r in res:
                        if r.get('status', '').startswith('Success'):
                            try:
                                quote = market.get_quote_symbol(plan['symbol'])
                                last_price = float(quote.get('last') or quote.get('prior'))
                                stop_price = round(last_price * 0.95, 2)
                                
                                stop_order = equity.place_order(
                                    symbol=plan['symbol'],
                                    price=stop_price,
                                    volume=r['qty'],
                                    side='Sell',
                                    pin=config.pin,
                                    stop_condition='LAST_PAID_OR_LOWER',
                                    stop_price=stop_price,
                                    stop_symbol=plan['symbol'],
                                    validity_type='Cancel' # Good Till Cancel
                                )
                                stop_no = stop_order.get('orderNo', 'Unknown')
                                logging.info(f"Placed 5% Stop-Loss for {plan['symbol']} at {stop_price} (No: {stop_no})")
                            except Exception as e:
                                logging.error(f"Failed to place Stop-Loss for {plan['symbol']}: {e}")
                                
            responses.extend(chunk_responses)
            if i < vwap_chunks - 1:
                logging.info(f"Waiting {buy_interval_sec} seconds for the next chunk...")
                time.sleep(buy_interval_sec)

    return responses

def _send_order(equity, market, side: str, symbol: str, quantity: int, pin: str) -> list:
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
                responses.append({'symbol': symbol, 'side': side, 'qty': board_lot, 'status': f'Success (No: {order_no})', 'order_no': order_no})
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
                responses.append({'symbol': symbol, 'side': side, 'qty': odd_lot, 'status': f'Success (No: {order_no})', 'order_no': order_no})
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
