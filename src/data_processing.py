import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

# --- Configuration & Path Setup ---
# Define PROJECT_ROOT as the parent of the src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input Paths
DATA_RAW_NIFTY = PROJECT_ROOT / "data_raw" / "data_raw_nifty500"
DATA_RAW_SP500 = PROJECT_ROOT / "data_raw" / "data_raw_sp500"

# Output Path
DATA_PROCESSED = PROJECT_ROOT / "data_processed"

# Ensure output directory exists
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("DataProcessor")
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(LOG_DIR / "data_processing.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler (Console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def load_parquet_directory_to_long(path: Path, ticker_from_filename: bool = True) -> pd.DataFrame:
    """
    Reads all .parquet files in a folder and combines them into a single long DataFrame.
    Optimized to use a list of DataFrames for memory efficiency before concatenation.
    """
    if not path.exists():
        logger.error(f"Directory not found: {path}")
        return pd.DataFrame()

    files = list(path.glob("*.parquet"))
    if not files:
        logger.warning(f"No parquet files found in {path}")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} files from {path.name}...")
    
    df_list = []
    
    for i, file_path in enumerate(files):
        try:
            # Read Parquet
            df = pd.read_parquet(file_path)
            
            # Basic validation
            if df.empty:
                continue

            # Add Ticker Column
            if ticker_from_filename:
                # filename "RELIANCE.NS.parquet" -> ticker "RELIANCE.NS"
                ticker = file_path.stem 
                df["Ticker"] = ticker

            # CRITICAL FIX: Normalize Date (Remove Timezone)
            # US stocks have EST, India has IST. We must make them timezone-naive to mix them.
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # Reduce memory: ensure consistent types and keep only needed columns
            cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            existing_cols = [c for c in cols if c in df.columns]
            df = df[existing_cols]

            df_list.append(df)

            # Log progress every 100 files
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1}/{len(files)} files...")

        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")

    if not df_list:
        return pd.DataFrame()

    logger.info("Concatenating DataFrames...")
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Sort for safety
    if 'Date' in full_df.columns:
        full_df = full_df.sort_values(by=['Ticker', 'Date'])
        
    return full_df


def build_price_matrix(df_long: pd.DataFrame, price_col: str = "Adj_Close") -> pd.DataFrame:
    """
    Pivots the long DataFrame to a wide matrix (Date x Ticker).
    Applies forward-fill to handle small data gaps (holidays).
    """
    logger.info(f"Building price matrix using column: {price_col}")
    
    # Pivot: Index=Date, Columns=Ticker, Values=Price
    price_matrix = df_long.pivot(index="Date", columns="Ticker", values=price_col)
    
    # Sort Index (Dates)
    price_matrix = price_matrix.sort_index()
      
    return price_matrix


def filter_by_missingness(price_matrix: pd.DataFrame, max_missing_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drops tickers that have too many missing values (e.g., newly listed stocks).
    Returns: (Filtered Price Matrix, Summary DataFrame)
    """
    logger.info(f"Filtering tickers with > {max_missing_frac*100}% missing data...")
    
    # Calculate missing fraction for each ticker
    missing_frac = price_matrix.isnull().mean()
    
    # Identify keepers and droppers
    tickers_to_keep = missing_frac[missing_frac <= max_missing_frac].index
    tickers_to_drop = missing_frac[missing_frac > max_missing_frac].index
    
    # Filter the matrix
    filtered_matrix = price_matrix[tickers_to_keep]
    
    # Create Summary for reporting
    summary_data = []
    for t in tickers_to_keep:
        summary_data.append({"Ticker": t, "Status": "Kept", "Missing_Frac": missing_frac[t]})
    for t in tickers_to_drop:
        summary_data.append({"Ticker": t, "Status": "Dropped", "Missing_Frac": missing_frac[t]})
        
    summary_df = pd.DataFrame(summary_data).sort_values("Missing_Frac")
    
    logger.info(f"Dropped {len(tickers_to_drop)} tickers. Kept {len(tickers_to_keep)}.")
    
    return filtered_matrix, summary_df


def compute_log_returns(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Log Returns: ln(Pt / Pt-1).
    Preferred over % returns for statistical properties (additivity).
    """
    logger.info("Computing log returns...")
    
    # np.log(Price / Prev_Price)
    # Using specific numpy handling to avoid errors with zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.log(price_matrix / price_matrix.shift(1))
    
    # The first row will be NaN (no previous day).
    # Infinite values (if price was 0) become NaNs.
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    return returns


def process_market(market_name: str, input_dir: Path, output_prefix: str):
    """
    Orchestrator function for a single market.
    """
    logger.info(f"--- Processing {market_name} ---")
    
    # 1. Load Data
    df_long = load_parquet_directory_to_long(input_dir)
    if df_long.empty:
        logger.error(f"No data found for {market_name}. Exiting.")
        return

    # 2. Build Price Matrix
    # Build Price Matrix
    prices = build_price_matrix(df_long,price_col="Adj_Close")

    # Measure TRUE missingness
    prices_clean, summary = filter_by_missingness(prices, max_missing_frac=0.2)

    # Only AFTER removing bad tickers,
    # forward-fill small gaps.
    prices_clean = prices_clean.ffill(limit=3)

    # Compute Returns
    returns_clean = compute_log_returns(prices_clean)
        
    # 5. Save Results
    price_path = DATA_PROCESSED / f"{output_prefix}_prices.parquet"
    ret_path = DATA_PROCESSED / f"{output_prefix}_returns.parquet"
    summary_path = DATA_PROCESSED / f"missingness_summary_{output_prefix}.csv"
    
    prices_clean.to_parquet(price_path)
    returns_clean.to_parquet(ret_path)
    summary.to_csv(summary_path, index=False)
    
    logger.info(f"Saved {market_name} prices shape: {prices_clean.shape}")
    logger.info(f"Saved {market_name} returns shape: {returns_clean.shape}")
    logger.info(f"Saved summary to {summary_path.name}")


def main():
    logger.info("Starting Data Processing Pipeline...")
    
    # Process NIFTY 500
    process_market(
        market_name="NIFTY 500", 
        input_dir=DATA_RAW_NIFTY, 
        output_prefix="nifty500"
    )
    
    # Process S&P 500
    process_market(
        market_name="S&P 500", 
        input_dir=DATA_RAW_SP500, 
        output_prefix="sp500"
    )
    
    logger.info("Data Processing Complete.")

if __name__ == "__main__":
    main()