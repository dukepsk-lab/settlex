"""settlex — SET50 ML trading-signal engine.

Pipeline: data (Settrade Open API) -> features (technical + fractional
differencing) -> CNN-BiLSTM + XGBoost ensemble -> Top-5 selection ->
Markowitz mean-variance optimisation -> Telegram signal.

The strategy is *advisory only*: it generates signals, it does not place
orders. See README.md for the research background and usage.
"""

__version__ = "0.1.0"
