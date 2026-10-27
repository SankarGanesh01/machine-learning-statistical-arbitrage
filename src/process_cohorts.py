import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_NIFTY = PROJECT_ROOT / "data_raw" / "data_raw_nifty500"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
LOG_DIR = PROJECT_ROOT / "logs"

# NEW START DATE for the "Broad Market" Cohort
COHORT_START_DATE = "2015-01-01"
# Maximum allowable missing observations
MAX_MISSING = 0.10

# Setup Logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "cohort_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CohortProcessor")

def load_and_process_cohort(input_dir: Path, market_name: str, start_date: str):
    logger.info(f"--- Processing {market_name} (Cohort: {start_date}+) ---")
    
    # 1. Load Raw Files
    files = list(input_dir.glob("*.parquet"))
    if not files:
        logger.error("No files found.")
        return

    logger.info(f"Loading {len(files)} files...")
    df_list = []
    
    for file_path in files:
        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                continue

            # Validate Required Columns
            required = {"Date", "Adj_Close"}

            if not required.issubset(df.columns):
                 logger.warning(
                     f"{file_path.name} missing required columns. "
                     f"Found: {list(df.columns)}"
                     )
                 continue

            # Normalize & Clean Dates
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

            # Sort chronologically
            df = df.sort_values("Date")

            # Remove duplicate trading days
            df = df.drop_duplicates(subset="Date", keep="last")

            # Validate Prices
            df = df[df["Adj_Close"] > 0]
            
            if df.empty:
                continue

            # Add Ticker
            df["Ticker"] = file_path.stem

            # Keep Required Columns
            df = df[["Date", "Ticker", "Adj_Close"]]

            # Filter to Cohort Period
            df = df[df["Date"] >= pd.to_datetime(start_date)]
            
            if not df.empty:
                df_list.append(df)
                
        except Exception as e:
            logger.warning(
                f"Failed loading {file_path.name}: {e}")
    if not df_list:
        logger.error(
            "No valid data after cleaning."
            )
    return
    # 2. Concatenate
    full_df = pd.concat(df_list, ignore_index=True)
    
    # 3. Pivot to Price Matrix
    logger.info("Building Price Matrix...")
    price_matrix = full_df.pivot(index="Date", columns="Ticker", values="Adj_Close")
    price_matrix = price_matrix.sort_index()
    price_matrix = price_matrix.ffill(limit=3)
    
    # 4. Filter Missingness
    # Since we sliced to 2015, stocks listed in 2014 will now look perfect
    missing_frac = price_matrix.isnull().mean()

    summary = pd.DataFrame({
        "Ticker": price_matrix.columns,
        "MissingFraction": missing_frac.values,
        "Kept":
            missing_frac.values <= MAX_MISSING})

    summary.to_csv( 
                   DATA_PROCESSED /
                   f"{market_name.lower()}_2015_missingness.csv",
                   index=False)

    tickers_to_keep = missing_frac[missing_frac <= MAX_MISSING].index # stricter 10% rule since shorter history
    clean_prices = price_matrix[tickers_to_keep]
    clean_prices = clean_prices.ffill(limit=3)
    
    logger.info(f"Original stocks      : {len(files)}")
    logger.info(f"Remaining stocks     : {len(clean_prices.columns)}")
    logger.info(f"Trading days         : {len(clean_prices)}")
    logger.info(f"Average missingness  : {missing_frac.mean():.2%}")

    
    # 5. Compute Returns
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.log(clean_prices.div(clean_prices.shift(1)))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # 6. Save with Special Suffix
    price_output = (
        DATA_PROCESSED /
        f"{market_name.lower()}_2015_prices.parquet")
    clean_prices.to_parquet(price_output,  compression="snappy")
    logger.info(
        f"Saved processed prices to: {price_output.name}")
    # 7. Save Log Returns
    returns_output = (
        DATA_PROCESSED /
        f"{market_name.lower()}_2015_returns.parquet")
    returns.to_parquet(returns_output, compression="snappy")
    logger.info(
        f"Saved log returns to: {returns_output.name}")
if __name__ == "__main__":
    # Run for NIFTY
    load_and_process_cohort(DATA_RAW_NIFTY, "NIFTY500", COHORT_START_DATE)
    # Run for S&P 500
    load_and_process_cohort(PROJECT_ROOT / "data_raw" / "data_raw_sp500", "SP500", COHORT_START_DATE)