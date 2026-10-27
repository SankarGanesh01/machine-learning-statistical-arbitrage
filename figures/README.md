# Figures - Statistical Arbitrage Analysis

This folder contains visual outputs from the statistical arbitrage study comparing machine-learning-based and traditional pair-selection strategies across the NIFTY 500 and S&P 500 universes.

The figures cover capital growth, equity curves, Sharpe ratio distributions, crisis-period behaviour, and comparisons between strategy approaches.

## 1. Capital Growth and Market Regimes

| Figure | Description |
|---|---|
| `Capital Growth Nifty Long.png` | Capital growth for the NIFTY 500 Long cohort, illustrating how the strategy's simulated wealth evolves over the study period. |
| `Capital Growth with Crisis Regimes.png` | Compares capital growth across the NIFTY 500 and S&P 500 cohorts and highlights selected crisis periods, including the Global Financial Crisis and COVID-19 crash. |

## 2. Equity Curves

| Figure | Description |
|---|---|
| `Equity Curve_ML Strategy(S&P 500 broad).png` | Simulated equity curve for the machine-learning strategy in the S&P 500 Broad cohort. |
| `Equity Curve-ML Starategy Nifty 500.png` | Simulated equity curve for the machine-learning strategy in the NIFTY 500 Long cohort. |
| `Equity Curve-Traditional Starategy Nifty 500.png` | Simulated equity curve for the traditional strategy in the NIFTY 500 Long cohort. |
| `Equity Curve-Traditional Starategy S&P 500.png` | Simulated equity curve for the traditional strategy in the S&P 500 Long cohort. |

Equity curves show the evolution of simulated portfolio value over time. They help illustrate cumulative performance and drawdowns but should be interpreted alongside transaction costs, risk metrics, and backtesting assumptions.

## 3. Sharpe Ratio Distributions

| Figure | Description |
|---|---|
| `Sharpe_Distribution(Nifty 500 Broad).png` | Distribution of Sharpe ratios for the NIFTY 500 Broad cohort. |
| `Sharpe_Distribution(Nifty 500 Long).png` | Distribution of Sharpe ratios for the NIFTY 500 Long cohort. |
| `Sharpe_Distribution(S&P 500 Long).png` | Distribution of Sharpe ratios for the S&P 500 Long cohort. |

These distributions help assess variation in risk-adjusted performance across the evaluated strategies or pairs. Negative Sharpe ratios indicate that the measured excess return was negative relative to the risk-free-rate assumption used in the calculation.

## 4. Strategy Comparisons

| Figure | Description |
|---|---|
| `Strategy Comparisson_Nifty 500.png` | Compares annualised Sharpe ratios for traditional and proposed machine-learning strategies in the developing-market cohort. |
| `Strategy Comparisson_S&P 500.png` | Compares annualised Sharpe ratios for traditional and proposed machine-learning strategies in the developed-market cohort. |

The comparison charts include error bars. Their precise interpretation should be taken from the plotting code or underlying analysis; they should not be assumed to represent confidence intervals without verification.

## Interpretation Notes

- These figures represent historical backtest results, not live-trading performance.
- Results depend on the selected stock universes, data treatment, pair-selection rules, trading assumptions, and transaction costs.
- Large differences in cumulative growth across cohorts may reflect differences in the underlying strategy and test setup; they should not be compared in isolation.
- The charts should be read alongside the corresponding exported tables and the methodology documented in the main repository README.

