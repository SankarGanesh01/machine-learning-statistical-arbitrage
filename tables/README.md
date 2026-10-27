# Tables - Statistical Arbitrage Analysis

This folder contains CSV outputs from the statistical arbitrage research comparing traditional and machine-learning-based pair-selection approaches across NIFTY 500 and S&P 500 market cohorts.

The tables provide exported backtest results, pair-selection outputs, and clustering results for further inspection.

## 1. Strategy Results

| File | Description |
|---|---|
| `NIFTY_500_broad_results.csv` | Exported results for the NIFTY 500 Broad cohort. |
| `NIFTY_500_long_results.csv` | Exported results for the NIFTY 500 Long cohort. |
| `S&P_500_long_results.csv` | Exported results for the S&P 500 Long cohort. |

These files contain cohort-level results from the analysis. Consult their column names and the generating notebook to determine the exact metrics, units, and calculation methods.

## 2. OPTICS Clustering Outputs

| File | Description |
|---|---|
| `optics_nifty_broad.csv` | Pair-selection or clustering output associated with the NIFTY 500 Broad cohort. |
| `optics_nifty_long.csv` | Pair-selection or clustering output associated with the NIFTY 500 Long cohort. |
| `optics_sp500_broad.csv` | Pair-selection or clustering output associated with the S&P 500 Broad cohort. |
| `optics_sp500_long.csv` | Pair-selection or clustering output associated with the S&P 500 Long cohort. |

These files are intended to document the output of the OPTICS-based pair-selection workflow. Their exact columns and meaning should be confirmed against the relevant notebook or processing function.

## Usage and Interpretation

- CSV files are provided for inspection and analysis; they are not necessarily raw input datasets.
- Results depend on the underlying data, cohort definitions, clustering settings, cointegration filters, and backtesting assumptions.
- Pair-selection output does not, by itself, demonstrate that a pair is profitable or will remain cointegrated.
- Performance metrics should be interpreted together with transaction costs, risk measures, and the strategy implementation.
- The repository's main README describes the research objective, methodology, and limitations.

