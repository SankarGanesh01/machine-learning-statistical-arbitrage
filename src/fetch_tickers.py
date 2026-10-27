import pandas as pd
import requests
import io
from pathlib import Path

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NIFTY_CSV_PATH = PROJECT_ROOT / "NIFTY-500-SYMBOLS.csv"
SP500_CSV_PATH = PROJECT_ROOT / "S-P-500-Symbols.csv"

# Reliable source for NIFTY 500 (NSE Archives)
NIFTY_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
# Standard source for S&P 500
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

def fetch_nifty500():
    print("Fetching NIFTY 500 symbols...")
    try:
        response = requests.get(NIFTY_URL, headers=get_headers())
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        
        if 'Symbol' not in df.columns:
            raise ValueError(f"Column 'Symbol' not found. Available columns: {df.columns.tolist()}")
            
        df_symbols = df[['Symbol']].copy()
        df_symbols = df_symbols.dropna().drop_duplicates().sort_values('Symbol')
        
        df_symbols.to_csv(NIFTY_CSV_PATH, index=False)
        print(f"✅ Success: Saved {len(df_symbols)} NIFTY tickers to {NIFTY_CSV_PATH.name}")
        
    except Exception as e:
        print(f"❌ Error fetching NIFTY 500: {e}")

def fetch_sp500():
    print("Fetching S&P 500 symbols...")
    try:
        # FIX: Use requests with headers first, then pass text to pandas
        response = requests.get(SP500_WIKI_URL, headers=get_headers())
        response.raise_for_status()
        
        # Use io.StringIO to avoid FutureWarning in newer pandas versions
        tables = pd.read_html(io.StringIO(response.text))
        
        # The constituents table is usually the first one
        df = tables[0]
        
        if 'Symbol' not in df.columns:
            raise ValueError("Column 'Symbol' not found in Wikipedia table.")
            
        # Clean Tickers: Yahoo Finance uses 'BRK-B', Wikipedia uses 'BRK.B'
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        
        df_symbols = df[['Symbol']].drop_duplicates().sort_values('Symbol')
        
        df_symbols.to_csv(SP500_CSV_PATH, index=False)
        print(f"✅ Success: Saved {len(df_symbols)} S&P 500 tickers to {SP500_CSV_PATH.name}")
        
    except Exception as e:
        print(f"❌ Error fetching S&P 500: {e}")

if __name__ == "__main__":
    fetch_nifty500()
    print("-" * 30)
    fetch_sp500()
    print("-" * 30)
    print("Ticker lists ready. You can now run 'python -m src.download_data'")