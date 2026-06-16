"""Daily morning briefing — multi-LLM advisory layer on top of the quant signal.

Pipeline (run pre-market, e.g. 07:00 via cron on a VPS):
  1. Load today's signal (ML Top-N + Markowitz weights) — the decision.
  2. Evaluate yesterday's signal vs realised prices (pure, no LLM).
  3. Diff yesterday's -> today's target into SELL/HOLD/BUY actions (pure, no LLM).
  4. DeepSeek  -> verify yesterday's prediction accuracy.
  5. Gemini    -> summarise market/stock news (Google Search grounding).
  6. Claude    -> synthesise an actionable order checklist from the rebalance delta.
  7. Assemble + deliver via Telegram.

LLMs are advisory only: they never change the Top-N or the weights. Every LLM
section is optional — a missing key or a failed call simply omits that section,
and the deterministic target-allocation table is always shown.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .advisory import _thai_date_str, load_last_signal, load_previous_signal
from .config import Settings, get_settings
from .data.loader import load_universe_ohlcv
from .evaluation import evaluate_predictions
from .rebalance import compute_rebalance, has_changes, rebalance_summary_text
from .universe import load_universe

NEWS_SYSTEM = (
    "You are a Thai equity market news analyst. Using web search, summarise today's "
    "most relevant, market-moving news for the given SET50 stocks and the overall "
    "Thai market (SET index, flows, macro). Respond in Thai, concise bullet points, "
    "facts only. Do NOT give buy/sell advice or price targets."
)
VERIFY_SYSTEM = (
    "You are a quantitative trading analyst reviewing the accuracy of a model's "
    "previous predictions against realised prices. Respond in Thai, concise. Comment "
    "on the hit rate, whether the recommended portfolio beat the equal-weight "
    "benchmark, and any notable hits/misses. Be objective — this is a performance "
    "review, not investment advice."
)
ORDERS_SYSTEM = (
    "You are a trading-operations assistant. The quantitative model has ALREADY "
    "decided today's target Top-N portfolio and weights — you must NOT change, "
    "reorder, add, or drop any of them. You are given the explicit rebalance delta "
    "versus yesterday's holdings (what to SELL, HOLD/ADJUST, BUY). Turn it into a "
    "clear, actionable order checklist the user executes manually before the open, "
    "in Thai: state each sell, each new buy with its THB size, and each hold "
    "(noting any add/trim amount). Use the news and verification only as short risk "
    "notes. End with: this is decision-support only, not financial advice, not "
    "auto-executed."
)


def _safe(fn: Callable[[], str]) -> Optional[str]:
    """Run an LLM call, swallowing any error (returns None so the section is skipped)."""
    try:
        text = fn()
        return text or None
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the briefing
        print(f"[warn] LLM call failed: {exc}")
        return None


def build_providers(settings: Settings) -> List:
    """Instantiate the three providers (each reports its own availability)."""
    from .llm.claude import ClaudeProvider
    from .llm.deepseek import DeepSeekProvider
    from .llm.gemini import GeminiProvider

    cfg = settings.llm
    return [ClaudeProvider(cfg), GeminiProvider(cfg), DeepSeekProvider(cfg)]


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x*100:+.1f}%" if isinstance(x, (int, float)) else "—"


def _fmt_rate(x: Optional[float]) -> str:
    """Plain (unsigned) percent — for ratios like hit rate."""
    return f"{x*100:.0f}%" if isinstance(x, (int, float)) else "—"


def _format_rebalance_lines(reb: Optional[Dict]) -> List[str]:
    """Deterministic SELL / HOLD / BUY action list (shown even without any LLM)."""
    if not reb or not reb.get("has_prev"):
        return []
    if not has_changes(reb):
        return ["*🔄 ปรับพอร์ต:* ถือพอร์ตเดิมทั้งหมด — ไม่ต้องส่งคำสั่งวันนี้"]
    out = [f"*🔄 ปรับพอร์ต (เทียบ {reb.get('prev_date')}):*"]
    for r in reb["sell"]:
        out.append(f"  🔴 ขายปิด {r['symbol']} (เคย {r['prev_weight']*100:.1f}%)")
    for r in reb["hold"]:
        if abs(r["weight_delta"]) <= 1e-9:
            out.append(f"  ⚪ ถือ {r['symbol']} {r['weight']*100:.1f}% (คงเดิม)")
        else:
            arrow = "🔺" if r["thb_delta"] > 0 else "🔻"
            out.append(
                f"  {arrow} {r['symbol']} {r['prev_weight']*100:.1f}%→{r['weight']*100:.1f}% "
                f"({r['thb_delta']:+,.0f}฿)"
            )
    for r in reb["buy"]:
        out.append(f"  🟢 ซื้อใหม่ {r['symbol']} {r['weight']*100:.1f}% (฿{r['thb']:,.0f})")
    out.append(f"  _turnover ~{reb['turnover']*100:.0f}%_")
    return out


def _news_prompt(signal: Dict) -> str:
    syms = ", ".join(p["symbol"] for p in signal.get("positions", []))
    return (
        f"วันที่ {signal.get('date')}. สรุปข่าวสำคัญวันนี้ของตลาดหุ้นไทย (SET) "
        f"และของหุ้นกลุ่มนี้โดยเฉพาะ: {syms}. เน้นข่าวที่อาจกระทบราคาในระยะ 3-5 วัน."
    )


def _verify_prompt(ev: Dict) -> str:
    lines = [
        f"ช่วงทบทวน: {ev.get('signal_date')} → {ev.get('eval_date')} "
        f"({ev.get('days_elapsed')} วันทำการ, horizon {ev.get('horizon')} วัน)",
        f"Hit rate: {_fmt_rate(ev.get('hit_rate'))}",
        f"ผลตอบแทนพอร์ตที่แนะนำ: {_fmt_pct(ev.get('portfolio_return'))}",
        f"benchmark (เฉลี่ยทั้ง universe): {_fmt_pct(ev.get('benchmark_return'))}",
        f"ส่วนต่าง (excess): {_fmt_pct(ev.get('excess_return'))}",
        "",
        "รายตัว (pred → realized):",
    ]
    for r in ev.get("positions", []):
        lines.append(
            f"  {r['symbol']}: pred {_fmt_pct(r['pred_return'])} → "
            f"จริง {_fmt_pct(r['realized_return'])}"
        )
    lines.append("\nช่วยประเมินสั้น ๆ ว่าโมเดลทำนายแม่นแค่ไหน และอะไรน่าสังเกต.")
    return "\n".join(lines)


def _orders_prompt(
    signal: Dict,
    news: Optional[str],
    verification: Optional[str],
    rebalance: Optional[Dict] = None,
) -> str:
    parts = [
        f"พอร์ตเป้าหมายวันนี้ ({signal.get('date')}), "
        f"เงินทุน ฿{signal.get('capital', 0):,.0f}:",
    ]
    for i, p in enumerate(signal.get("positions", []), 1):
        parts.append(
            f"  {i}. {p['symbol']} — น้ำหนัก {p['weight']*100:.1f}% "
            f"(฿{p['thb']:,.0f}), คาดการณ์ {_fmt_pct(p['pred_return'])}"
        )
    if rebalance and rebalance.get("has_prev"):
        parts.append(
            f"\nสิ่งที่ต้องปรับจากพอร์ตเมื่อวาน ({rebalance.get('prev_date')}):\n"
            f"{rebalance_summary_text(rebalance)}"
        )
    elif rebalance is not None and not rebalance.get("has_prev"):
        parts.append("\nวันแรก — เปิดสถานะตามตารางเป้าหมายทั้งหมด.")
    if verification:
        parts.append(f"\nผลทบทวนเมื่อวาน:\n{verification}")
    if news:
        parts.append(f"\nข่าววันนี้:\n{news}")
    parts.append(
        "\nเขียน checklist คำสั่งซื้อขายสำหรับวันนี้ให้ชัดเจน ปฏิบัติได้จริง "
        "(ขายตัวไหน / ถือตัวไหน / ซื้อใหม่ตัวไหนพร้อมจำนวนเงิน) "
        "ห้ามเปลี่ยนรายชื่อหุ้นหรือน้ำหนัก พร้อมหมายเหตุความเสี่ยงสั้น ๆ."
    )
    return "\n".join(parts)


def generate_briefing(
    settings: Optional[Settings] = None,
    synthetic: bool = False,
    providers: Optional[List] = None,
) -> Dict:
    """Assemble the morning briefing dict (see :func:`format_briefing`)."""
    settings = settings or get_settings()
    signal = load_last_signal(settings.data_dir)

    evaluation = None
    rebalance = None
    if signal:
        prev = load_previous_signal(settings.data_dir, before=signal["date"])
        live_portfolio = None
        if not synthetic:
            from .data.portfolio import load_manual_portfolio
            live_portfolio = load_manual_portfolio(settings.data_dir)
            
            if live_portfolio is None and settings.settrade.is_complete and settings.settrade.account_no:
                try:
                    from .data.settrade_client import SettradeClient
                    live_portfolio = SettradeClient(settings.settrade).get_live_portfolio()
                except Exception:
                    pass
        rebalance = compute_rebalance(prev, signal, live_portfolio=live_portfolio)
        if prev:
            symbols = load_universe()
            ohlcv = load_universe_ohlcv(symbols, settings, synthetic=synthetic)
            evaluation = evaluate_predictions(prev, ohlcv)

    providers = providers if providers is not None else build_providers(settings)
    pmap = {p.name: p for p in providers if p.is_available()}

    sections: Dict[str, Optional[str]] = {}
    if settings.llm.enabled and signal and signal.get("positions"):
        if "gemini" in pmap:
            sections["news"] = _safe(lambda: pmap["gemini"].complete(_news_prompt(signal), NEWS_SYSTEM))
        if "deepseek" in pmap and evaluation:
            sections["verification"] = _safe(
                lambda: pmap["deepseek"].complete(_verify_prompt(evaluation), VERIFY_SYSTEM)
            )
        if "claude" in pmap:
            sections["orders"] = _safe(
                lambda: pmap["claude"].complete(
                    _orders_prompt(
                        signal, sections.get("news"), sections.get("verification"), rebalance
                    ),
                    ORDERS_SYSTEM,
                )
            )

    return {
        "date": signal["date"] if signal else None,
        "signal": signal,
        "evaluation": evaluation,
        "rebalance": rebalance,
        "sections": sections,
        "providers_used": sorted(k for k, v in sections.items() if v),
    }


def format_briefing(briefing: Dict) -> str:
    """Render the briefing as a Markdown Telegram message (Thai)."""
    signal = briefing.get("signal")
    ev = briefing.get("evaluation")
    sections = briefing.get("sections", {})

    from datetime import date

    lines = [
        "*🌅 SETTLEX — Morning Briefing*",
        _thai_date_str(date.fromisoformat(briefing["date"]) if briefing.get("date") else date.today()),
        "",
    ]

    if ev:
        lines += [
            f"*📉 ทวนสอบเมื่อวาน* ({ev.get('signal_date')} → {ev.get('eval_date')}, "
            f"{ev.get('days_elapsed')} วันทำการ):",
            f"  • Hit rate: {_fmt_rate(ev.get('hit_rate'))}",
            f"  • พอร์ตแนะนำ {_fmt_pct(ev.get('portfolio_return'))} "
            f"vs benchmark {_fmt_pct(ev.get('benchmark_return'))} "
            f"(ส่วนต่าง {_fmt_pct(ev.get('excess_return'))})",
        ]
        if sections.get("verification"):
            lines += ["", f"_DeepSeek:_ {sections['verification']}"]
        lines.append("")

    if sections.get("news"):
        lines += ["*📰 ข่าววันนี้ (Gemini):*", sections["news"], ""]

    if signal and signal.get("positions"):
        lines.append("*📊 แผนวันนี้ — พอร์ตเป้าหมาย (ML model):*")
        lines.append("```")
        lines.append(f"{'#':<2}{'SYM':<6}{'WT':>6}{'SHARES':>8}{'THB':>10}")
        for i, p in enumerate(signal["positions"], 1):
            s_text = str(p.get("shares", "-"))
            lines.append(f"{i:<2}{p['symbol']:<6}{p['weight']*100:>5.1f}%{s_text:>8}{p['thb']:>10,.0f}")
        lines.append("```")
        reb_lines = _format_rebalance_lines(briefing.get("rebalance"))
        if reb_lines:
            lines += ["", *reb_lines]
        if sections.get("orders"):
            lines += ["", f"_Claude:_ {sections['orders']}"]
    elif signal:
        lines.append("*ไม่มีสัญญาณซื้อวันนี้ → ถือเงินสด*")
    else:
        lines.append("_ยังไม่มีสัญญาณ — รัน_ `settlex signal` _ก่อน_")

    lines += [
        "",
        "⚠️ _เพื่อการตัดสินใจเสริมเท่านั้น ไม่ใช่คำแนะนำการลงทุน — LLM ไม่ได้แก้สัญญาณของโมเดล_",
    ]
    return "\n".join(lines)
