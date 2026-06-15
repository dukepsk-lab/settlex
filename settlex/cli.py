"""settlex command-line interface.

Commands:
  fetch-data   Download (or synthesise) SET50 OHLCV into the parquet cache.
  train        Train the CNN-BiLSTM + XGBoost ensemble and save it.
  backtest     Run a walk-forward backtest and print a performance report.
  signal       Generate today's Top-N signal and send it to Telegram.

Run from the repo root, e.g.  `python -m settlex.cli signal --dry-run`.
"""
from __future__ import annotations

import argparse
import sys

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
    from .signals.generate import generate_signal
    from .signals.telegram import format_signal, send_message

    settings = get_settings()
    result = generate_signal(settings, synthetic=args.synthetic, capital=args.capital)
    message = format_signal(result)
    print(message)

    if args.dry_run:
        print("\n[dry-run] Not sending to Telegram.")
        return
    if not settings.telegram.is_complete:
        print("\n[error] Telegram not configured. Set TELEGRAM_BOT_TOKEN and "
              "TELEGRAM_CHAT_ID, or use --dry-run.")
        sys.exit(2)
    send_message(message, settings.telegram)
    print("\n[sent] Signal delivered to Telegram.")


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
    p.add_argument("--dry-run", action="store_true", help="print only; do not send to Telegram")
    p.set_defaults(func=cmd_signal)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
