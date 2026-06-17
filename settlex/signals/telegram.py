"""Format and deliver signals via the Telegram Bot API (plain `requests`)."""
from __future__ import annotations

from typing import Dict, List, Optional

import requests

from ..config import TelegramConfig

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_LEN = 4000  # Telegram hard limit is 4096; leave headroom


def _chunk(text: str, limit: int = MAX_LEN) -> List[str]:
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks or [""]


def send_message(text: str, config: TelegramConfig, parse_mode: str = "Markdown") -> List[dict]:
    """Send ``text`` to the configured chat, splitting if over the length limit."""
    if not config.is_complete:
        raise ValueError(
            "Telegram is not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
        )
    url = API_URL.format(token=config.bot_token)
    responses = []
    for chunk in _chunk(text):
        resp = requests.post(
            url,
            json={"chat_id": config.chat_id, "text": chunk, "parse_mode": parse_mode},
            timeout=30,
        )
        resp.raise_for_status()
        responses.append(resp.json())
    return responses


def format_signal(result: Dict, live_portfolio: Optional[Dict] = None, rebalance: Optional[Dict] = None) -> str:
    """Render a signal result dict (see signals.generate) as a Markdown message."""
    date = result.get("date", "?")
    horizon = result.get("horizon", "?")
    
    lines = [
        "*📈 SETTLEX — SET50 Signal*",
        f"_Date:_ {date}   _Horizon:_ {horizon} trading days",
        ""
    ]

    # 1. Current Portfolio
    if live_portfolio and "positions" in live_portfolio:
        total_value = live_portfolio["cash"] + sum(p["market_value"] for p in live_portfolio["positions"].values())
        lines.append(f"💼 *พอร์ตปัจจุบัน:* มูลค่ารวม ฿{total_value:,.0f}")
        for sym, p in live_portfolio["positions"].items():
            lines.append(f"  • {sym}: {p['shares']} หุ้น (฿{p['market_value']:,.0f})")
        lines.append(f"  • เงินสด: ฿{live_portfolio['cash']:,.0f}")
        lines.append("")
    else:
        capital = result.get("capital", 0)
        lines.append(f"💼 *เงินทุนเริ่มต้น (Capital):* ฿{capital:,.0f}")
        lines.append("")

    # 2. Target Signal
    positions = result.get("positions", [])
    if not positions:
        lines.append("📊 *พอร์ตเป้าหมาย:* ถือเงินสด (100%)")
    else:
        invested = sum(p["weight"] for p in positions)
        lines.append("📊 *พอร์ตเป้าหมาย (ML Model):*")
        lines.append("```")
        lines.append(f"{'#':<2}{'SYM':<6}{'WT':>6}{'SHARES':>8}{'THB':>10}{'PRICE':>8}{'PRED':>7}")
        for i, p in enumerate(positions, 1):
            s_text = str(p.get("shares", "-"))
            price = p.get("price")
            price_text = f"{price:>8.2f}" if price is not None else f"{'-':>8}"
            pred = p.get("pred_return")
            pred_text = f"{pred*100:>6.1f}%" if pred is not None else f"{'-':>7}"
            lines.append(
                f"{i:<2}{p['symbol']:<6}{p['weight']*100:>5.1f}%"
                f"{s_text:>8}{p['thb']:>10,.0f}{price_text}{pred_text}"
            )
        lines.append("```")
        cash = max(0.0, 1.0 - invested)
        lines.append(f"_เงินสดที่ต้องเหลือ:_ {cash*100:.1f}%")

    agreement = result.get("model_agreement")
    if agreement is not None:
        lines.append(f"_Model agreement (CNN-BiLSTM↔XGB):_ {agreement:.2f}")

    lines.append("")

    # 3. Action Recommendation
    if rebalance:
        from ..rebalance import rebalance_summary_text
        lines.append("📝 *คำแนะนำ (Action Checklist):*")
        lines.append(rebalance_summary_text(rebalance))

    return "\n".join(lines)


def format_run_message(
    result: Dict, 
    live_portfolio: Optional[Dict] = None, 
    rebalance: Optional[Dict] = None,
    evaluation: Optional[Dict] = None,
    news: Optional[str] = None,
    execution_responses: Optional[List[Dict]] = None,
    claude_summary: Optional[str] = None,
) -> str:
    """Format the unified clean daily run message."""
    date = result.get("date", "?")
    
    lines = [
        f"📈 *SETTLEX Report* ({date})",
        ""
    ]

    if news:
        lines.append("📰 *ข่าวเด่นวันนี้:*")
        lines.append(news)
        lines.append("")

    if evaluation:
        hit_rate = evaluation.get("hit_rate", 0.0)
        port_ret = evaluation.get("portfolio_return", 0.0)
        lines.append(f"📉 *รีวิวเมื่อวาน:* Hit rate {hit_rate*100:.1f}% (พอร์ต {port_ret*100:+.2f}%)")
        lines.append("")

    if live_portfolio and "positions" in live_portfolio:
        total_value = live_portfolio["cash"] + sum(p["market_value"] for p in live_portfolio["positions"].values())
        lines.append("💼 *พอร์ตปัจจุบัน:*")
        lines.append(f"มูลค่ารวม: ฿{total_value:,.0f}")
        for sym, p in live_portfolio["positions"].items():
            shares = max(1, p.get("shares", 1))
            avg_cost = p.get("average_cost", 0.0)
            market_price = p.get("market_value", 0.0) / shares
            cost_str = f"{avg_cost:.2f}" if avg_cost > 0 else "-"
            lines.append(f"- {sym}: {p['shares']} หุ้น (ต้นทุน: {cost_str} | ราคาตลาด: {market_price:.2f})")
        lines.append(f"เงินสดคงเหลือ: ฿{live_portfolio['cash']:,.0f}")
        lines.append("")
    else:
        capital = result.get("capital", 0)
        lines.append("💼 *พอร์ตปัจจุบัน (อิงจากทุนจำลอง):*")
        lines.append(f"มูลค่ารวม: ฿{capital:,.0f}")
        lines.append("")

    # Target Signal Table
    positions = result.get("positions", [])
    if positions:
        lines.append("📊 *พอร์ตเป้าหมาย:*")
        lines.append("```")
        lines.append(f"{'#':<2}{'SYM':<6}{'WT':>6}{'SHARES':>8}{'THB':>10}{'PRICE':>8}{'PRED':>7}")
        for i, p in enumerate(positions, 1):
            s_text = str(p.get("shares", "-"))
            price = p.get("price")
            price_text = f"{price:>8.2f}" if price is not None else f"{'-':>8}"
            pred = p.get("pred_return")
            pred_text = f"{pred*100:>6.1f}%" if pred is not None else f"{'-':>7}"
            lines.append(f"{i:<2}{p['symbol']:<6}{p['weight']*100:>5.1f}%{s_text:>8}{p['thb']:>10,.0f}{price_text}{pred_text}")
        lines.append("```")
        lines.append("")

    lines.append("📝 *แผนการเทรดวันนี้ (Action):*")
    if rebalance:
        lines.append("🔴 *SELL (ขายเพื่อทำกำไร/ตัดขาดทุน):*")
        lines.append("```")
        lines.append(f"{'ACTION':<7}{'SYM':<6}{'SHARES':>8}{'PRICE':>8}{'RESULT':>8}")
        has_sell = False
        for r in rebalance.get("sell", []):
            s_text = str(r.get("shares", "-"))
            result_str = f"{r['result_pct']*100:+.1f}%"
            lines.append(f"{'SELL':<7}{r['symbol']:<6}{s_text:>8}{r['price']:>8.2f}{result_str:>8}")
            has_sell = True
        if not has_sell:
            lines.append("ไม่มีรายการขาย")
        lines.append("```")
        lines.append("")
        
        lines.append(f"💵 รับเงินสดจากการขาย: ฿{rebalance.get('cash_from_sell', 0):,.0f}")
        lines.append(f"💰 รวมเงินสดพร้อมซื้อ (Cash Balance): ฿{rebalance.get('total_cash_for_buy', 0):,.0f}")
        lines.append("")

        lines.append("🟢 *BUY (ซื้อหุ้นเป้าหมาย):*")
        lines.append("```")
        lines.append(f"{'ACTION':<7}{'SYM':<6}{'SHARES':>8}{'PRICE':>8}{'PRED':>8}")
        has_buy = False
        for r in rebalance.get("buy", []):
            s_text = str(r.get("shares", "-"))
            pred_str = f"{r['pred_pct']*100:+.1f}%"
            lines.append(f"{'BUY':<7}{r['symbol']:<6}{s_text:>8}{r['price']:>8.2f}{pred_str:>8}")
            has_buy = True
        if not has_buy:
            lines.append("ไม่มีรายการซื้อ")
        lines.append("```")
    else:
        lines.append("ไม่มีข้อมูลเปรียบเทียบพอร์ต")

    if claude_summary:
        lines.append("")
        lines.append("🤖 *Claude AI Analysis:*")
        lines.append(claude_summary)

    if execution_responses is not None:
        lines.append("")
        success_count = sum(1 for r in execution_responses if r.get("error") is None)
        total = len(execution_responses)
        if total > 0:
            lines.append(f"⚡️ ยิงคำสั่งสำเร็จ {success_count}/{total} รายการ")
        else:
            lines.append(f"⚡️ ไม่มียอดที่ต้องส่งคำสั่งซื้อขาย")

    return "\n".join(lines)
