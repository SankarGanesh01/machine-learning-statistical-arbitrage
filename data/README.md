# Data Documentation - Machine Learning Statistical Arbitrage

This directory documents the market data used in the research project comparing statistical arbitrage strategies across the NIFTY 500 and S&P 500.

The analysis covers two historical windows for each market: a long cohort beginning in 2007 and a broad cohort beginning in 2015. The data pipeline downloads historical prices, constructs cleaned price and return matrices, and prepares inputs for PCA-based factor neutralisation, OPTICS clustering, cointegration testing, and backtesting.

**Raw market data is not committed to this repository.** Users must obtain the required inputs separately and comply with the relevant data-provider terms.

## 1. Data Sources

| Market | Source documented in the research | Main purpose |
|---|---|---|
| NIFTY 500 | National Stock Exchange of India (NSE) and financial databases with historical price coverage | Historical Indian equity prices and universe construction |
| S&P 500 | Yahoo Finance and historical constituent information | Historical US equity prices and universe construction |

The Python download script uses `yfinance` to retrieve historical prices. The source code also expects local ticker-list CSV files for both universes.

### Expected ticker-list files

The downloader references these files in the project root:

- `NIFTY-500-SYMBOLS.csv`
- `S&P 500 Symbols.csv`

These files supply the ticker universe used by the download process. Their constituent dates and historical coverage should be checked before interpreting the resulting dataset as point-in-time index membership.

## 2. Historical Coverage and Cohorts

The research defines four cohorts:

| Cohort | Market | Start date | End year |
|---|---|---:|---:|
| NIFTY 500 Long | India | 2007 | 2025 |
| NIFTY 500 Broad | India | 2015 | 2025 |
| S&P 500 Long | United States | 2007 | 2025 |
| S&P 500 Broad | United States | 2015 | 2025 |

The downloader configuration specifies a start date of `2007-01-01` and an end date of `2025-12-31`. The broad-cohort processing script uses `2015-01-01` as its start date.

These are configured date boundaries, not a guarantee that every stock has observations on every trading day. Actual coverage depends on listing history, ticker availability, missing prices, and the data returned by the provider.

## 3. Local Data Organisation

The supplied processing scripts expect the following directories relative to the project root:

```text
data/
└── README.md

data_raw/
├── data_raw_nifty500/
└── data_raw_sp500/

data_processed/
```

The `data_raw/` and `data_processed/` directories are local working directories. They are not required to be committed to GitHub.

The downloader saves historical ticker-level price files under the corresponding market-specific raw-data directory. The processing scripts read those files and produce cleaned matrices and derived outputs under `data_processed/`.

**Important:** The current code uses the project root to resolve these directories. If you change the repository layout or move scripts into another directory, verify the path configuration before running the pipeline.

## 4. Data Processing

### 4.1 Price matrix construction

The processing workflow loads individual stock price files and aligns observations by date to construct a market-level price matrix.

The resulting matrix is used to calculate returns and prepare the data for factor modelling and pair selection.

### 4.2 Missing-data filtering

The processing code applies different missingness thresholds to the long and broad cohorts.

- The general data-processing script uses a maximum missingness threshold of **20%**.
- The broad-cohort script uses a stricter maximum missingness threshold of **10%**.
- The broad-cohort script applies forward filling with a maximum limit of **three consecutive observations** after filtering.

These choices affect which securities remain in the analysis. Forward filling can also affect return estimates around gaps, so the treatment should be considered when interpreting the results.

### 4.3 Log-return calculation

For a price series \(P_t\), the pipeline calculates log returns as:

\[
r_t=\ln\left(\frac{P_t}{P_{t-1}}\right)
\]

The implementation replaces infinite return values with missing values after calculation.

Log returns provide the input representation for PCA-based factor analysis and subsequent statistical modelling.

### 4.4 Cohort construction

The broad cohort is created by restricting the available history to observations from 2015 onward and applying the broad-cohort missingness filter.

The long cohort covers the configured historical period beginning in 2007.

The actual stock universe retained in each cohort depends on the ticker lists, historical data availability, filtering rules, and preprocessing implementation.

## 5. Downstream Use of the Data

The prepared data supports the following research stages:

1. **PCA-based factor neutralisation:** Extract common return factors and construct residual-return inputs.
2. **OPTICS clustering:** Group securities based on the representation used by the pair-selection workflow.
3. **Cointegration testing:** Evaluate candidate relationships for evidence of a long-run equilibrium relationship.
4. **Strategy construction:** Generate candidate pairs and trading signals.
5. **Backtesting:** Evaluate simulated strategy performance under the documented transaction-cost and risk assumptions.

The data files alone do not establish that a candidate pair is tradable, cointegrated out of sample, or profitable after execution costs.

## 6. Survivorship Bias and Point-in-Time Membership

The research aims to address survivorship bias by reconstructing historical index universes. However, this requires historical constituent membership information, including securities that subsequently left an index, were delisted, or ceased trading.

A current ticker list combined with historical price data is not sufficient on its own to eliminate survivorship bias.

Before reproducing or extending the analysis, verify:

- Whether constituent membership is available as of each historical date or only at the beginning of each cohort.
- Whether delisted and removed securities are represented.
- Whether ticker changes and corporate actions are handled consistently.
- Whether PCA fitting, clustering, cointegration testing, and signal generation avoid look-ahead bias.

The reliability of any survivorship-bias correction depends on the actual membership data and implementation.

## 7. Reproducing the Data Pipeline

To prepare the inputs locally:

1. Obtain the required historical price data and authorised ticker lists.
2. Place the ticker-list CSV files in the project root using the filenames expected by the downloader.
3. Run the download script to populate the market-specific raw-data directories.
4. Run the data-processing script to create cleaned price and return matrices.
5. Run the cohort-processing script to construct the broad-cohort inputs.
6. Inspect the logs and generated missingness summaries to verify which securities were retained.
7. Run the analysis notebooks using the resulting datasets.

The exact execution order and generated filenames should be confirmed against the source scripts and notebook imports. The pipeline may require adjustments for current provider behaviour or missing historical data.

## 8. Data Licensing and Reproducibility

This repository does not distribute the full underlying historical market datasets.

Users should obtain data through authorised sources and review applicable licensing, access, and redistribution restrictions before downloading or publishing it. Public availability of a dataset does not automatically permit redistribution.

For reproducibility, record the source, retrieval date, ticker universe, date coverage, adjustment method, missing-data treatment, and relevant processing parameters for each dataset version.

---

**Scope note:** This README describes the data paths and processing rules identified in the supplied Python scripts. The historical-universe reconstruction and survivorship-bias treatment should be independently verified before the research is described as fully survivorship-bias-free.
