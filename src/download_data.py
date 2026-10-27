import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from logging.handlers import RotatingFileHandler
import pandas as pd
import yfinance as yf

# --- Path Definitions ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Constants
DATA_RAW_NIFTY = PROJECT_ROOT / "data_raw" / "data_raw_nifty500"
DATA_RAW_SP500 = PROJECT_ROOT / "data_raw" / "data_raw_sp500"
LOG_DIR = PROJECT_ROOT / "logs"

# CORRECTED FILENAME: Matches "S&P 500 Symbols.csv"
NIFTY_SYMBOLS_CSV = PROJECT_ROOT / "NIFTY-500-SYMBOLS.csv"
SP500_SYMBOLS_CSV = PROJECT_ROOT / "S&P 500 Symbols.csv"

START_DATE = "2007-01-01"
END_DATE = "2025-12-31"

# --- Logging Setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file_path = LOG_DIR / "download_data.log"

logger = logging.getLogger("DataDownloader")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3
)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(file_formatter)
logger.addHandler(stream_handler)


def read_symbol_list(csv_path: Path, symbol_col: str = "Symbol") -> List[str]:
    if not csv_path.exists():
        logger.error(f"Symbol file not found: {csv_path}")
        return []
    try:
        df = pd.read_csv(csv_path)
        if symbol_col not in df.columns:
            logger.error(f"Column '{symbol_col}' not found. Found: {df.columns.tolist()}")
            return []
        return df[symbol_col].dropna().astype(str).str.strip().tolist()
    except Exception as e:
        logger.error(f"Failed to read symbol list from {csv_path}: {e}")
        return []

def build_yahoo_ticker_universe() -> Tuple[List[str], List[str]]:
    nifty_raw = read_symbol_list(NIFTY_SYMBOLS_CSV)
    sp500_raw = read_symbol_list(SP500_SYMBOLS_CSV)

    # NIFTY: Add .NS
    nifty_tickers = [f"{sym}.NS" for sym in nifty_raw]
    
    # S&P 500: Replace dots with dashes (BRK.B -> BRK-B)
    sp500_tickers = [sym.replace(".", "-") for sym in sp500_raw]

    return nifty_tickers, sp500_tickers

def save_ticker_parquet(df: pd.DataFrame, ticker: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize ticker (remove :)
    safe_ticker = ticker.replace(":", "")
    file_path = out_dir / f"{safe_ticker}.parquet"
    df.to_parquet(file_path, engine="pyarrow", compression="snappy")

def process_batch(tickers_batch: List[str], out_dir: Path, start: str, end: str, min_days: int) -> Dict:
    stats = {"success": 0, "failed": 0, "thin": 0, "failures_list": [], "thin_list": []}
    
    if not tickers_batch:
        return stats

    try:
        logger.info(f"Downloading batch of {len(tickers_batch)} tickers...")
        # Batch download
        data = yf.download(tickers_batch, start=start, end=end, group_by='ticker', threads=True, progress=False, auto_adjust=False)
        
        # Handle single ticker edge case (returns flat DF instead of MultiIndex)
        if len(tickers_batch) == 1:
            ticker = tickers_batch[0]
            data = pd.concat({ticker: data}, axis=1, names=["Ticker", "Price"])

        for ticker in tickers_batch:
            try:
                if ticker not in data.columns.levels[0]:
                    stats["failed"] += 1
                    stats["failures_list"].append(ticker)
                    continue
                
                df = data[ticker].copy()
                df = df.dropna(how='all') # Drop empty rows

                if df.empty:
                    stats["failed"] += 1
                    stats["failures_list"].append(ticker)
                    continue

                if len(df) < min_days:
                    stats["thin"] += 1
                    stats["thin_list"].append({'Symbol': ticker, 'Count': len(df)})
                    continue

                # Clean Columns
                df = df.reset_index()
                if 'Adj Close' in df.columns:
                    df = df.rename(columns={'Adj Close': 'Adj_Close'})
                if 'Date' not in df.columns and 'Datetime' in df.columns:
                    df = df.rename(columns={'Datetime': 'Date'})
                
                save_ticker_parquet(df, ticker, out_dir)
                stats["success"] += 1

            except Exception:
                stats["failed"] += 1
                stats["failures_list"].append(ticker)

    except Exception as e:
        logger.error(f"Batch download crash: {e}")
        for t in tickers_batch:
            stats["failed"] += 1
            stats["failures_list"].append(t)
            
    return stats

def process_universe(tickers: List[str], out_dir: Path, start: str, end: str, min_trading_days: int = 250):
    logger.info(f"Processing universe for output: {out_dir}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify what is already downloaded to skip it
    existing_files = {f.stem for f in out_dir.glob("*.parquet")}
    
    # Filter list
    tickers_to_process = []
    for t in tickers:
        safe_t = t.replace(":", "")
        # Check if safe_t (filename) exists
        if safe_t not in existing_files:
            tickers_to_process.append(t)
    
    skipped_count = len(tickers) - len(tickers_to_process)
    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} tickers (already downloaded).")

    # Process remaining in batches of 50
    BATCH_SIZE = 50
    total_stats = {"success": 0, "failed": 0, "thin": 0}
    all_failures = []
    all_thin = []

    for i in range(0, len(tickers_to_process), BATCH_SIZE):
        batch = tickers_to_process[i : i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(tickers_to_process)//BATCH_SIZE)+1}")
        
        batch_stats = process_batch(batch, out_dir, start, end, min_trading_days)
        
        total_stats["success"] += batch_stats["success"]
        total_stats["failed"] += batch_stats["failed"]
        total_stats["thin"] += batch_stats["thin"]
        all_failures.extend(batch_stats["failures_list"])
        all_thin.extend(batch_stats["thin_list"])
        
        time.sleep(1) # Be nice to API

    # Save Logs
    report_base = PROJECT_ROOT / "data_raw"
    report_base.mkdir(parents=True, exist_ok=True)
    
    fail_csv = report_base / f"{out_dir.name}_download_failures.csv"
    thin_csv = report_base / f"{out_dir.name}_thin_history.csv"
    
    if all_failures:
        pd.DataFrame({'Symbol': all_failures}).to_csv(fail_csv, mode='a', header=not fail_csv.exists(), index=False)
    if all_thin:
        pd.DataFrame(all_thin).to_csv(thin_csv, mode='a', header=not thin_csv.exists(), index=False)

    logger.info(f"Finished {out_dir.name}. Session Stats: {total_stats}")

def run_all_universes():
    logger.info("Starting High-Speed Data Download Pipeline...")
    nifty, sp500 = build_yahoo_ticker_universe()

    if nifty:
        print(f"Checking NIFTY download ({len(nifty)} symbols)...")
        process_universe(nifty, DATA_RAW_NIFTY, START_DATE, END_DATE)
    
    if sp500:
        print(f"Starting S&P 500 download ({len(sp500)} symbols)...")
        process_universe(sp500, DATA_RAW_SP500, START_DATE, END_DATE)

    logger.info("Pipeline Complete.")

if __name__ == "__main__":
    run_all_universes()