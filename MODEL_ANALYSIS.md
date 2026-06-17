# Model Analysis for settlex

This document provides a comprehensive overview of how the `settlex` machine-learning trading-signal engine generates its predictions and how those predictions are utilized.

## What the Model is Predicting

The model's primary objective is to predict the **forward N-day simple return** of each stock in the universe. This is framed as a **regression** problem, where the target variable ($y$) is the percentage change in the asset's closing price over the next N days.

- **Target Horizon (`horizon`):** By default, it predicts the **3-to-5 day forward return** (configurable via `SETTLEX_HORIZON`, defaulting to 5).
- **Format:** A continuous real number representing expected forward performance.

## How the Model Predicts

The predictive engine employs a **Mixture-of-Experts (MoE) style ensemble** that averages the outputs of two distinct machine learning architectures:

1.  **CNN-BiLSTM (Deep Learning):**
    -   **Input:** Sequential time-series data (`X_seq`) with a shape of `(n_samples, lookback, n_features)`. The default `lookback` is 60 trading days.
    -   **Architecture:** Conv1D layers extract local temporal and cross-feature patterns. These are fed into stacked Bidirectional LSTMs (Long Short-Term Memory) to capture sequential dependencies in both directions.
    -   **Regularization:** Employs aggressive regularization (Dropout, L2, Early Stopping) to prevent overfitting on the relatively small SET50 dataset.

2.  **XGBoost (Gradient Boosting):**
    -   **Input:** Flat cross-sectional data (`X_flat`) representing the most recent feature values for the stock, with a shape of `(n_samples, n_features)`.
    -   **Role:** Acts as a robust "factor" model, handling outliers well and finding non-linear cross-sectional relationships across fundamental/technical features.

**Ensembling:** The predictions from both models (for the same forward return target) are averaged. This approach mitigates the single-model bias of each underlying architecture.

**Data Normalization & Leakage Prevention:**
-   Features are normalized using a **trailing rolling z-score** (typically a 252-day window).
-   Crucially, this z-score calculation strictly uses data *up to and including the current bar* to guarantee **no look-ahead bias** leaks into a sample's features.

## The Features Used

The models are fed a robust set of features, carefully engineered to capture price dynamics, momentum, and volatility without requiring external dependencies:

1.  **Technical Indicators:**
    -   *Momentum/Returns:* 1-day, 5-day, 10-day returns, RSI, MACD, ADX.
    -   *Volatility:* ATR (Average True Range), Bollinger Bands (width and %b).
    -   *Volume/Price Dynamics:* OBV (On-Balance Volume), VWAP deviation, volume z-scores.
    -   *Support/Resistance Proximity:* Distance from 20-day high and low.
2.  **Fractionally-Differenced Log Price:**
    -   Standard differencing (e.g., daily returns) makes prices stationary but destroys valuable long-term memory.
    -   Fractional differencing applies a non-integer order to make the series stationary *while preserving as much historical memory as possible*, which is highly beneficial for recurrent deep models.
3.  **Relative Strength:**
    -   The performance of each symbol compared to the equal-weighted index over a 20-session window.

## How the Predictions are Utilized

Once the ensemble produces forward return predictions for all symbols in the universe, the system translates these into a portfolio allocation:

1.  **Top-N Selection:**
    -   Predictions are ranked from highest to lowest.
    -   The strategy selects the **Top 5** (configurable via `SETTLEX_TOP_N`) predicted names.
    -   It is strictly **long-only**; it filters out names with negative expected returns. If fewer than 5 names have positive outlooks, the strategy takes a partial cash position.
2.  **Markowitz Mean-Variance Optimization:**
    -   The selected Top-N basket is passed to a portfolio optimizer (via PyPortfolioOpt).
    -   **Expected Returns:** The model's predicted forward returns.
    -   **Risk Model:** A Ledoit-Wolf shrunk sample covariance matrix computed from historical prices.
    -   **Objective:** Maximize the Sharpe ratio to find the optimal long-only weights.
    -   **Constraints:** Fully invested (up to the positive names found) with a strict maximum weight per name (e.g., 30%, configurable via `SETTLEX_MAX_WEIGHT`) to ensure diversification.
    -   **Fallback:** If the convex optimization is infeasible, it gracefully falls back to a capped prediction-proportional weighting.
3.  **Actionable Output:**
    -   The optimal weights are converted into absolute THB capital allocations and share counts.
    -   These targets are diffed against the previous day's holdings to generate explicit deterministic **SELL**, **HOLD**, and **BUY** actions.
