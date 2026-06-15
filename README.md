# settlex

**SET50 machine-learning trading-signal engine.** `settlex` implements the
strategy from the research report *"Feasibility and Architecture of Machine
Learning Trading Systems for the SET50 Index"* and delivers its output as a
**Telegram signal** — it tells you *what to hold*, it does **not** place orders.

> ⚠️ **Decision-support only. Not financial advice. Not auto-executed.**
> Expected metrics quoted in the research (CAGR 18–24%, Sharpe 1.6–1.8,
> Max Drawdown < 15%) are research expectations, not guarantees.

## Strategy at a glance

| Stage | Choice (from the report) |
|-------|--------------------------|
| Data | Yahoo Finance EOD daily OHLCV (default, no creds) or Settrade Open API |
| Features | Technical indicators (RSI, MACD, ADX, ATR, Bollinger, OBV, VWAP) + **fractionally-differenced** log-price + relative strength |
| Model | **CNN-BiLSTM + XGBoost ensemble** (averaged) |
| Target | **3–5 day forward return** (regression) |
| Selection | **Top 5** predicted names, **long-only** |
| Weighting | **Markowitz mean-variance** (max-Sharpe, Ledoit-Wolf cov, ≤30%/name) |
| Validation | **Walk-forward** (rolling retrain, embargoed, vs 1/N benchmark) |
| Delivery | **Telegram** message, run on demand |

Daily bars are used throughout, which sidesteps the SET midday-session gap and
matches the multi-day swing horizon (avoiding intraday price-band friction).

## Install

```bash
pip install -r requirements.txt        # or: pip install -e .
```

Python ≥ 3.9. The heavy dependency is `tensorflow-cpu` (CNN-BiLSTM).

> **Note:** on some images `settrade-v2`'s legacy `setup.py` deps (`stringcase`,
> `paho-mqtt`) fail to build with very new setuptools (a Debian `install_layout`
> quirk). If you hit that, install with stdlib distutils:
> ```bash
> SETUPTOOLS_USE_DISTUTILS=stdlib pip install settrade-v2
> ```

## Configure

```bash
cp .env.example .env
# then edit .env
```

- **Market data** — `SETTLEX_DATA_SOURCE=yahoo` (default) pulls free EOD daily
  bars from Yahoo Finance (Thai tickers via the `.BK` suffix), so you can run
  the whole pipeline with **no broker credentials**. Since signals are generated
  after the close to plan the next session, EOD data is all you need. Set
  `SETTLEX_DATA_SOURCE=settrade` to use the broker feed instead.
- **Settrade Open API** (optional) — request `APP_ID` / `APP_SECRET` from your
  Thai broker (e.g. Pi Securities) Settrade Open API console. Use
  `SETTRADE_APP_CODE=ALGO_EQ` for equities. Setting `SETTRADE_ACCOUNT_NO` lets
  `signal` auto-fetch your real portfolio equity as the capital base. Docs:
  <https://developer.settrade.com/open-api/>
- **Telegram** — create a bot with [@BotFather](https://t.me/BotFather) for the
  token, message your bot, then read your chat id from
  `https://api.telegram.org/bot<TOKEN>/getUpdates`.

No credentials at all? Every command also accepts `--synthetic` to run the full
pipeline on deterministic generated data.

## Usage

```bash
# 1. Cache OHLCV (Yahoo, no creds by default) ... or synthetic for a dry run
python -m settlex.cli fetch-data
python -m settlex.cli fetch-data --synthetic

# 2. Train and save the ensemble (use --quick for a fast smoke run)
python -m settlex.cli train
python -m settlex.cli train --synthetic --quick

# 3. Walk-forward backtest (prints CAGR / Sharpe / Sortino / MaxDD vs 1/N)
python -m settlex.cli backtest --start 2019-01-01 --end 2024-12-31
python -m settlex.cli backtest --synthetic --quick --model xgboost

# 4. Generate today's signal -> Telegram (--dry-run prints without sending)
python -m settlex.cli signal --dry-run
python -m settlex.cli signal

# 5. Daily advisory: is today a trading day, when to run, last signal recap
python -m settlex.cli advisory --dry-run
python -m settlex.cli advisory

# 6. Pre-market briefing (optional LLMs): verify yesterday + news + order plan
python -m settlex.cli briefing --dry-run
python -m settlex.cli briefing
```

If installed with `pip install -e .`, the `settlex` console script is available
(`settlex signal --dry-run`).

### Pre-market briefing (multi-LLM, optional)

`briefing` runs an **advisory** layer on top of the quant signal — ideal as a
~07:00 cron job before the open. It (1) verifies yesterday's predictions against
realised prices, (2) summarises news, and (3) writes an actionable order plan:

| Provider | Role | Key |
|----------|------|-----|
| **DeepSeek** | verify yesterday's prediction accuracy | `DEEPSEEK_API_KEY` |
| **Gemini** | summarise market/stock news (Google Search grounding) | `GEMINI_API_KEY` |
| **Claude** | synthesise today's order checklist | `ANTHROPIC_API_KEY` |

Install the SDKs with `pip install -e ".[llm]"`. Each provider is **independent
and optional** — a missing key (or a failed call) simply omits that section; the
deterministic target-allocation table always shows. **LLMs never change the
Top-N or weights** — the ML model remains the sole decision-maker. Disable all
LLM sections with `SETTLEX_LLM_ENABLED=0`.

Example VPS cron (07:00 Asia/Bangkok, weekdays):

```cron
0 7 * * 1-5  cd /path/to/settlex && /path/to/.venv/bin/settlex briefing >> briefing.log 2>&1
```

## How it fits together

```
settlex/
  config.py            # env-driven settings & secrets
  universe.py          # SET50 constituents (override via SETTLEX_UNIVERSE_FILE)
  data/                # Yahoo + Settrade clients, caching loader, synthetic fallback
  features/            # technical indicators, fractional differencing, dataset pipeline
  models/              # XGBoost, CNN-BiLSTM, and the averaging ensemble
  portfolio/           # Top-N selection + Markowitz mean-variance optimiser
  backtest/            # walk-forward engine + financial metrics
  signals/             # live signal orchestration + Telegram delivery
  llm/                 # optional Claude / Gemini / DeepSeek providers (briefing)
  evaluation.py        # prediction-accuracy scoring (picks vs realised prices)
  advisory.py          # daily trading-day status + action checklist
  briefing.py          # pre-market multi-LLM briefing orchestration
  cli.py               # fetch-data | train | backtest | signal | advisory | briefing
```

## Tests

```bash
python -m pytest          # fast unit tests (no model training / no network)
```

## Known limitations (see also the research report)

- **Survivorship bias** — backtests use the current SET50 list by default. For
  rigorous results supply point-in-time constituents via `SETTLEX_UNIVERSE_FILE`.
- **Corporate actions** — the Yahoo source uses `auto_adjust=True` (split/dividend
  adjusted). If you switch to Settrade, verify whether its candlesticks are
  adjusted and add an adjustment step if it returns raw prices.
- **Alternative data** — NVDR / foreign-flow features (the report's top alpha
  source) are **not** included in this version; they can be layered onto the
  feature pipeline later.
- **Compute** — TensorFlow training is CPU-bound; use `--quick` for smoke runs.
