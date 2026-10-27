# Statistical Arbitrage Enhanced by Machine Learning
### A Comparative Study of the NIFTY 500 and S&P 500

## Overview

This project investigates whether machine-learning-assisted statistical arbitrage can improve the selection of potentially mean-reverting equity pairs across Indian and US equity markets.

The research combines classical econometric methods with unsupervised machine learning to identify related stocks, reduce common market-factor exposure, test long-run relationships, and evaluate simulated trading performance under explicit cost and risk constraints.

The study compares the NIFTY 500, representing the Indian equity market, with the S&P 500, representing the US equity market.

## Research Objectives

- Develop a multi-stage statistical arbitrage framework using Principal Component Analysis (PCA), OPTICS clustering, and cointegration testing.
- Compare traditional and machine-learning-assisted pair-selection strategies.
- Evaluate strategy performance across long-term and more recent market cohorts.
- Incorporate transaction costs, stop-loss controls, and market-specific hurdle rates into backtesting.
- Examine the implications of market structure, survivorship bias, and changing market regimes for quantitative strategy evaluation.

## Market Cohorts

| Cohort | Market | Study period |
|---|---|---|
| NIFTY 500 Long | India | 2007–2025 |
| NIFTY 500 Broad | India | 2015–2025 |
| S&P 500 Long | United States | 2007–2025 |
| S&P 500 Broad | United States | 2015–2025 |

The long cohorts provide a broader historical perspective, while the broad cohorts focus on a more recent market period. Exact start and end dates depend on the observations available after preprocessing.

## Methodology

The proposed research workflow consists of the following stages:

**1. Data collection and preprocessing**

Historical equity data is prepared for the four market cohorts. Data quality, universe construction, corporate actions, missing observations, and survivorship bias are relevant considerations when building historical backtests.

**2. Factor neutralisation using PCA**

Principal Component Analysis is used to identify dominant common sources of return variation. Residual extraction is intended to reduce broad market-factor influence and help isolate stock-specific behaviour.

**3. Asset grouping using OPTICS**

OPTICS (Ordering Points To Identify the Clustering Structure) is used as a density-based clustering method. Unlike approaches that require a fixed number of clusters in advance, OPTICS can help identify clusters with different density structures and distinguish noise points.

**4. Cointegration testing**

The Engle–Granger two-step procedure is used to test whether candidate asset relationships exhibit a statistically supported long-run equilibrium relationship. Cointegration is not the same as correlation and does not guarantee that a relationship will persist in future markets.

**5. Trading strategy construction**

Candidate pairs are evaluated within a pairs-trading framework based on deviations from an estimated relationship and the expectation of mean reversion. The study compares a traditional baseline with a machine-learning-assisted selection approach.

**6. Backtesting and risk evaluation**

Simulated strategy performance is evaluated using portfolio growth, risk-adjusted performance, and other backtest outputs. The framework incorporates transaction-cost assumptions, a hard stop-loss, and market-specific hurdle rates.

**7. Model interpretation**

The research includes SHAP-based explainability to help investigate the factors associated with model outputs and improve the interpretability of the machine-learning component.

## Backtesting Assumptions

| Parameter | Research assumption |
|---|---|
| Indian market hurdle rate | 7.2% |
| US market hurdle rate | 4.2% |
| Transaction costs | 15 basis points, as defined in the research implementation |
| Hard stop-loss | −5% |
| Traditional benchmark | Conventional statistical-arbitrage/pairs-trading approach |
| Proposed approach | PCA-based factor neutralisation, OPTICS clustering, and cointegration validation |

These parameters reflect the assumptions described in the research report. The precise implementation of transaction costs, trade entry and exit, stop-loss handling, and hurdle-rate calculations should be checked against the backtesting code.

## Repository Structure

```text
.
├── README.md
├── data/
│   └── README.md
├── src/
│   └── [source code]
├── notebooks/
│   └── [analysis notebooks, if included]
├── figures/
│   └── README.md
├── tables/
│   └── README.md
├── requirements.txt
└── .gitignore
```

This is a reference structure. The actual filenames and folders in the repository should be kept consistent with the uploaded files.

## Results and Visualisations

The `figures/` directory contains selected visual outputs, including:

- Simulated capital-growth curves.
- Equity curves for traditional and machine-learning strategies.
- Sharpe ratio distributions.
- Strategy-level comparisons across the NIFTY 500 and S&P 500 cohorts.
- Selected crisis-regime comparisons.

The `tables/` directory contains exported CSV results associated with the strategy cohorts and clustering workflow.

See the README files inside `figures/` and `tables/` for descriptions of individual outputs. Performance claims should be verified against the underlying result tables and implementation before being treated as final findings.

## Key Research Considerations

- **Survivorship bias:** Historical universe construction can materially affect backtest results. A survivorship-bias adjustment should be supported by point-in-time constituent data and a documented procedure.
- **Look-ahead bias:** Pair selection, factor estimation, parameter fitting, and signal generation must use only information available at the relevant decision time.
- **Transaction costs:** Brokerage, taxes, slippage, liquidity, and market impact can reduce or eliminate apparent statistical-arbitrage profits.
- **Regime changes:** Cointegrated relationships can break down during market stress or structural changes.
- **Model risk:** Clustering and statistical significance alone do not establish economic profitability or future stability.
- **Reproducibility:** Results depend on data availability, preprocessing, parameter choices, execution assumptions, and the exact code used.

## Limitations

This is a historical research and backtesting project, not a live-trading system or investment recommendation. Simulated performance may differ from executable returns because of liquidity constraints, market impact, short-selling restrictions, data limitations, and changes in market structure.

The reliability of the results depends on verifying the implementation of the historical universe reconstruction, survivorship-bias treatment, signal timing, and transaction-cost model. The presence of a stop-loss does not eliminate gap risk or guarantee that a position can be closed at the specified threshold.

## Technologies

The research workflow uses Python-based quantitative analysis and statistical modelling, with machine-learning and econometric methods. The exact installation requirements are listed in `requirements.txt`; any additional R packages or external data dependencies should be documented separately if they are used by the final codebase.

## Data and Usage

Raw market data is not included by default. Refer to `data/README.md` for information about data sources, expected inputs, and local data organisation.

Users should independently review the applicable data-provider terms before downloading, redistributing, or publishing market data. Exported figures and summary tables should not expose restricted underlying datasets.

## Research Context

This repository accompanies a comparative research project on machine-learning-enhanced statistical arbitrage across the NIFTY 500 and S&P 500.

It is intended to demonstrate a quantitative research workflow spanning data preparation, factor modelling, unsupervised learning, econometric validation, backtesting, and performance interpretation.

