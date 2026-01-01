# Directional Forecasting of Cryptocurrency Markets Using Machine Learning

This project investigates the feasibility of predicting the **directional movement**
of cryptocurrency prices using supervised Machine Learning models, with a strong
focus on **temporal integrity**, **robust validation**, and **economic relevance**.

The study is designed as an end-to-end pipeline, covering data acquisition, feature
engineering, model training, walk-forward validation, and financial backtesting.

---

## Problem Definition

Given historical OHLCV data for BTC/USDT on a 4-hour timeframe, the task is to predict
whether the **next candle** will close higher or lower than the current one.

The problem is formulated as a **binary classification task**, where:

- `1` indicates upward price movement
- `0` indicates downward or neutral movement

---

## Data Source

Market data is obtained from the **Binance API**, including:

- Open
- High
- Low
- Close
- Volume

All data is sampled at a fixed **4-hour interval**, ensuring temporal consistency.

---

## Data Treatment Strategy

Special care is taken to ensure **data integrity** and prevent **information leakage**.

Key principles include:

- Validation of continuous 4h intervals and OHLC consistency
- Strict temporal alignment between features and target
- Transformation of prices into log-returns to stabilize variance
- Time-aware normalization, fitting scalers exclusively on training windows
- No aggressive denoising or artificial class balancing

This approach preserves the true dynamics of the market while enabling robust
statistical learning.

---

## Feature Engineering

The initial feature set includes:

- **Returns & Momentum**: log-returns, rolling momentum
- **Trend Indicators**: SMA, EMA
- **Volatility Measures**: rolling standard deviation, ATR
- **Oscillators**: RSI, MACD

All features are computed using **past information only**.

---

## Models

The following models are evaluated:

- Logistic Regression (baseline)
- Gradient Boosting (XGBoost)
- Long Short-Term Memory (LSTM)

Model complexity is intentionally controlled to prioritize robustness and
generalization.

---

## Validation Strategy

A **walk-forward validation** scheme is employed, reflecting real-world deployment
conditions:

- Training on historical data
- Testing on strictly future, unseen periods
- No random shuffling or standard cross-validation

---

## Financial Evaluation

Model predictions are evaluated not only through statistical metrics but also through
a simplified trading simulation:

- Long position when prediction = 1
- Short position when prediction = 0
- One trade per candle
- No leverage
- Fixed transaction cost

Key financial metrics include:

- Cumulative Return
- Sharpe Ratio
- Maximum Drawdown

---

## Results

Detailed performance metrics and equity curves are available in the `results/`
directory. A full technical discussion is provided in `report.md`.

---

## Limitations

- Single asset (BTC/USDT)
- Fixed timeframe (4h)
- Simplified execution model without slippage modeling

These limitations are intentional to ensure clarity and reproducibility.

---

## Reproducibility

All experiments can be reproduced by installing the required dependencies and
executing the notebooks in numerical order, or by running the modular scripts
available in the `src/` directory.

---

## Future Work

- Extension to multiple assets
- Regime detection
- Advanced denoising techniques
- Probabilistic position sizing
