"""settlex command-line interface.

Commands:
  fetch-data   Download (or synthesise) SET50 OHLCV into the parquet cache.
  train        Train the CNN-BiLSTM + XGBoost ensemble and save it.
  backtest     Run a walk-forward backtest and print a performance report.
  signal       Generate today's Top-N signal and send it to Telegram.
  advisory     Show today's trading-day status and action checklist.
  briefing     Pre-market multi-LLM briefing (verify yesterday + news + orders).

Run from the repo root, e.g.  `python -m settlex.cli signal --dry-run`.
"""
from __future__ import annotations

import argparse
import sys
import os

# Suppress noisy TensorFlow C++ warnings and info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from .config import get_settings


def cmd_fetch_data(args: argparse.Namespace) -> None:
    from .data.loader import load_universe_ohlcv
    from .universe import load_universe

    settings = get_settings()
    symbols = load_universe()
    print(f"Fetching {len(symbols)} symbols (synthetic={args.synthetic}, refresh={args.refresh}) ...")
    data = load_universe_ohlcv(symbols, settings, refresh=args.refresh, synthetic=args.synthetic)
    rows = {s: len(df) for s, df in data.items() if df is not None and not df.empty}
    if rows:
        print(f"Loaded {len(rows)}/{len(symbols)} symbols. rows min={min(rows.values())} max={max(rows.values())}")
    else:
        print("No symbols loaded.")
    print(f"Cache: {settings.data_dir / 'ohlcv'}")


def cmd_train(args: argparse.Namespace) -> None:
    from .data.loader import equal_weight_index_close, load_universe_ohlcv
    from .features.pipeline import build_dataset
    from .models.ensemble import Ensemble

    settings = get_settings()
    data = load_universe_ohlcv(None, settings, synthetic=args.synthetic)
    index_close = equal_weight_index_close(data)

    print("Building dataset ...")
    dataset = build_dataset(data, settings, index_close=index_close)
    print(
        f"Dataset: {len(dataset)} samples | {len(dataset.feature_names)} features | "
        f"seq shape {dataset.X_seq.shape}"
    )

    model = Ensemble()
    print(f"Training ensemble (quick={args.quick}) ...")
    model.fit(dataset, quick=args.quick, verbose=1 if args.verbose else 0)

    settings.ensure_dirs()
    model.save(settings.ensemble_path)
    print(f"Saved ensemble -> {settings.ensemble_path}")


def cmd_backtest(args: argparse.Namespace) -> None:
    from .backtest.walkforward import run_walkforward
    from .data.loader import load_universe_ohlcv

    settings = get_settings()
    data = load_universe_ohlcv(None, settings, synthetic=args.synthetic)
    result = run_walkforward(
        data,
        settings,
        model_name=args.model,
        train_years=args.train_years,
        test_months=args.test_months,
        quick=args.quick,
        start=args.start,
        end=args.end,
        cost_bps=args.cost_bps,
        verbose=True,
    )
    print()
    print(result.report())
    if args.out:
        result.predictions.to_csv(args.out, index=False)
        print(f"\nSaved predictions -> {args.out}")


def cmd_signal(args: argparse.Namespace) -> None:
    from .advisory import append_signal_history, save_last_signal
    from .signals.generate import generate_signal
    from .signals.telegram import format_signal, send_message

    settings = get_settings()
    result = generate_signal(settings, synthetic=args.synthetic, capital=args.capital)
    
    # Calculate live_portfolio and rebalance to pass to format_signal
    from .advisory import load_previous_signal
    from .rebalance import compute_rebalance
    
    live_portfolio = None
    if not args.synthetic:
        from .data.portfolio import load_manual_portfolio
        live_portfolio = load_manual_portfolio(settings.data_dir)
        
        if live_portfolio is None and settings.settrade.is_complete and settings.settrade.account_no:
            try:
                from .data.settrade_client import SettradeClient
                live_portfolio = SettradeClient(settings.settrade).get_live_portfolio()
            except Exception:
                pass

    prev = load_previous_signal(settings.data_dir, before=result["date"])
    reb = compute_rebalance(prev, result, live_portfolio=live_portfolio)

    message = format_signal(result, live_portfolio=live_portfolio, rebalance=reb)
    print(message)

    # Persist so `advisory`/`briefing` can recap and verify it next morning.
    if not args.synthetic:
        settings.ensure_dirs()
        save_last_signal(result, settings.data_dir)
        append_signal_history(result, settings.data_dir)

    if args.dry_run:
        print("\n[dry-run] Not sending to Telegram.")
    else:
        if not settings.telegram.is_complete:
            print("\n[error] Telegram not configured. Set TELEGRAM_BOT_TOKEN and "
                  "TELEGRAM_CHAT_ID, or use --dry-run.")
            sys.exit(2)
        send_message(message, settings.telegram)
        print("\n[sent] Signal delivered to Telegram.")

    if getattr(args, "execute", False):
        if args.dry_run:
            print("\n[dry-run] Not executing InnovestX orders.")
        else:
            print("\n[execute] Calculating order diff and sending to InnovestX...")
            from .execution.innovestx import execute_orders
            responses = execute_orders(reb, settings.innovestx, result["date"])
            
            print(f"\n[executed] Processed {len(responses)} order requests.")
            for resp in responses:
                err = resp.get("error")
                if err:
                    print(f"  ❌ {resp['side']} {resp['symbol']} - Error: {err}")
                else:
                    print(f"  ✅ {resp['side']} {resp['symbol']} - Status: {resp['status']}")


def cmd_advisory(args: argparse.Namespace) -> None:
    from .advisory import format_advisory, generate_advisory
    from .signals.telegram import send_message

    settings = get_settings()
    advisory = generate_advisory(settings)
    message = format_advisory(advisory)
    print(message)

    if args.dry_run:
        print("\n[dry-run] Not sending to Telegram.")
        return
    if not settings.telegram.is_complete:
        print("\n[error] Telegram not configured. Set TELEGRAM_BOT_TOKEN and "
              "TELEGRAM_CHAT_ID, or use --dry-run.")
        sys.exit(2)
    send_message(message, settings.telegram)
    print("\n[sent] Advisory delivered to Telegram.")


def cmd_briefing(args: argparse.Namespace) -> None:
    from .briefing import format_briefing, generate_briefing
    from .signals.telegram import send_message

    settings = get_settings()
    briefing = generate_briefing(settings, synthetic=args.synthetic)
    message = format_briefing(briefing)
    print(message)
    if briefing.get("providers_used"):
        print(f"\n[llm] sections from: {', '.join(briefing['providers_used'])}")
    else:
        print("\n[llm] no LLM sections (no keys configured or all calls failed)")

    if args.dry_run:
        print("\n[dry-run] Not sending to Telegram.")
        return
    if not settings.telegram.is_complete:
        print("\n[error] Telegram not configured. Set TELEGRAM_BOT_TOKEN and "
              "TELEGRAM_CHAT_ID, or use --dry-run.")
        sys.exit(2)
    send_message(message, settings.telegram)
    print("\n[sent] Briefing delivered to Telegram.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="settlex", description="SET50 ML trading-signal engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-data", help="download/synthesise SET50 OHLCV into the cache")
    p.add_argument("--synthetic", action="store_true", help="use deterministic synthetic data (no creds)")
    p.add_argument("--refresh", action="store_true", help="ignore cache and refetch")
    p.set_defaults(func=cmd_fetch_data)

    p = sub.add_parser("train", help="train and save the CNN-BiLSTM + XGBoost ensemble")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--quick", action="store_true", help="few epochs / small trees for a fast smoke run")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="run a walk-forward backtest")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--model", choices=["ensemble", "xgboost", "cnn_bilstm"], default="ensemble")
    p.add_argument("--train-years", type=int, default=3, dest="train_years")
    p.add_argument("--test-months", type=int, default=12, dest="test_months")
    p.add_argument("--cost-bps", type=float, default=15.0, dest="cost_bps")
    p.add_argument("--start", default=None, help="first train-end date, e.g. 2019-01-01")
    p.add_argument("--end", default=None, help="last date to evaluate, e.g. 2024-12-31")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=None, help="optional CSV path for OOS predictions")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("signal", help="generate today's signal and send to Telegram")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--capital", type=float, default=None, help="override capital in THB")
    p.add_argument("--dry-run", action="store_true", help="print only; do not send to Telegram or execute orders")
    p.add_argument("--execute", action="store_true", help="automatically execute order diff via InnovestX webhook")
    p.set_defaults(func=cmd_signal)

    p = sub.add_parser("advisory", help="show today's trading-day status and action checklist")
    p.add_argument("--dry-run", action="store_true", help="print only; do not send to Telegram")
    p.set_defaults(func=cmd_advisory)

    p = sub.add_parser("briefing", help="pre-market briefing: verify yesterday + news + order plan (LLMs)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print only; do not send to Telegram")
    p.set_defaults(func=cmd_briefing)

    p = sub.add_parser("run", help="unified daily run (signal + advisory + briefing + execute)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--capital", type=float, default=None, help="override capital in THB")
    p.add_argument("--dry-run", action="store_true", help="print only; do not send to Telegram or execute orders")
    p.add_argument("--execute", action="store_true", help="automatically execute order diff via InnovestX webhook")
    p.set_defaults(func=cmd_run)

    return parser


def cmd_run(args: argparse.Namespace) -> None:
    from .advisory import generate_advisory, load_previous_signal, save_last_signal, append_signal_history
    from .signals.generate import generate_signal
    from .signals.telegram import format_run_message, send_message
    from .rebalance import compute_rebalance
    from .data.portfolio import load_manual_portfolio
    from .evaluation import evaluate_predictions
    from .universe import load_universe
    from .data.loader import load_universe_ohlcv
    from .briefing import build_providers

    settings = get_settings()
    
    # 1. Check Advisory
    advisory = generate_advisory(settings)
    if not advisory.get("is_trading_day", True):
        msg = f"📉 *SETTLEX Report* ({advisory.get('date', '?')})\n\nวันนี้ตลาดปิดครับ พักผ่อนได้เลย 🏖️"
        print(msg)
        if not args.dry_run and settings.telegram.is_complete:
            send_message(msg, settings.telegram)
        return

    # 2. Generate today's target signal
    print("\n[run] Generating today's signal...")
    result = generate_signal(settings, synthetic=args.synthetic, capital=args.capital)

    # 3. Load Portfolio
    live_portfolio = None
    if not args.synthetic:
        live_portfolio = load_manual_portfolio(settings.data_dir)

    # 4. Compute rebalance
    prev = load_previous_signal(settings.data_dir, before=result["date"])
    reb = compute_rebalance(prev, result, live_portfolio=live_portfolio)

    # 5. Evaluate yesterday's prediction
    evaluation = None
    if prev:
        try:
            symbols = load_universe()
            ohlcv = load_universe_ohlcv(symbols, settings, synthetic=args.synthetic)
            evaluation = evaluate_predictions(prev, ohlcv)
        except Exception as e:
            print(f"\n[run] Failed to evaluate yesterday's signal: {e}")

    # 6. LLM News
    news_text = None
    try:
        providers = build_providers(settings)
        pmap = {p.name: p for p in providers if p.is_available()}
        
        if "gemini" in pmap:
            prompt = (
                f"Give a short 1-paragraph summary of today's SET50/Thai stock market sentiment "
                f"or key news on {result['date']}. Keep it concise, engaging, and in Thai."
            )
            news_text = pmap["gemini"].complete(prompt, "You are an expert Thai stock market analyst.")
    except Exception as e:
        print(f"\n[run] Failed to fetch LLM news: {e}")

    # 6.5 LLM Claude Summary
    claude_summary = None
    try:
        if "claude" in locals().get("pmap", {}) and reb and (reb.get("sell") or reb.get("buy")):
            from .rebalance import rebalance_summary_text
            reb_text = rebalance_summary_text(reb)
            system = (
                "You are an expert Thai quant trader. Summarize the rebalance actions (Sell and Buy) "
                "for today. Focus specifically on explaining WHY each stock is being sold (e.g., target weight decreased, "
                "taking profit, or cutting loss based on the price). Keep it concise, professional, and in Thai."
            )
            prompt = (
                f"Market News:\n{news_text or 'No news'}\n\n"
                f"Rebalance Actions:\n{reb_text}\n\n"
                f"Please summarize and explain the decisions, especially the SELL actions."
            )
            print("\n[run] Fetching Claude analysis...")
            claude_summary = pmap["claude"].complete(prompt, system=system)
    except Exception as e:
        print(f"\n[run] Failed to fetch Claude analysis: {e}")

    # 7. Execute via Settrade Open API
    execution_responses = None
    if getattr(args, "execute", False):
        if args.dry_run:
            print("\n[dry-run] Not executing Settrade orders.")
        else:
            print("\n[execute] Sending orders to Settrade Open API...")
            from .execution.settrade_exec import execute_orders
            execution_responses = execute_orders(reb, settings.settrade, result["date"])
            print(f"\n[executed] Processed {len(execution_responses)} order requests.")

    # 8. Format and Send message
    message = format_run_message(
        result=result,
        live_portfolio=live_portfolio,
        rebalance=reb,
        evaluation=evaluation,
        news=news_text,
        execution_responses=execution_responses,
        claude_summary=claude_summary
    )
    try:
        print("\n" + message)
    except UnicodeEncodeError:
        print("\n" + message.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

    # Persist signal
    if not args.synthetic:
        settings.ensure_dirs()
        save_last_signal(result, settings.data_dir)
        append_signal_history(result, settings.data_dir)

    # Send
    if args.dry_run:
        print("\n[dry-run] Not sending to Telegram.")
    else:
        if not settings.telegram.is_complete:
            print("\n[error] Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or use --dry-run.")
            sys.exit(2)
        send_message(message, settings.telegram)
        print("\n[sent] Daily Run Report delivered to Telegram.")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
