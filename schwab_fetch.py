"""
Schwab API Data Fetcher for ORB Analysis
==========================================
Downloads historical 5-min bars for NQ and ES futures.
Saves to CSV for use with orb_local_pipeline.py.

SETUP:
    1. Register app at developer.schwab.com
    2. Set environment variables:
         set SCHWAB_API_KEY=your_key
         set SCHWAB_APP_SECRET=your_secret
    3. Run this script — browser will open for first-time auth
    4. After auth, token is saved and reused automatically

USAGE:
    python schwab_fetch.py
"""

import os
import sys
import json
from datetime import datetime, timedelta

import pandas as pd
from schwab import auth, client

# ============================================================
# CONFIG
# ============================================================
CALLBACK_URL = "https://127.0.0.1:8182/callback"
TOKEN_PATH = "schwab_token.json"

# Futures symbols — front month continuous
SYMBOLS = ["/NQ", "/ES"]

# How far back to pull (in days). Schwab allows varying amounts
# for minute data. We'll try chunks and see what sticks.
LOOKBACK_DAYS = 365 * 2  # Try 2 years
CHUNK_DAYS = 10  # Schwab limits minute requests to ~10 days per call

OUTPUT_DIR = "data"


# ============================================================
# AUTH
# ============================================================
def get_client():
    """Authenticate with Schwab API."""
    api_key = os.environ.get("SCHWAB_API_KEY")
    app_secret = os.environ.get("SCHWAB_APP_SECRET")

    if not api_key or not app_secret:
        print("ERROR: Set SCHWAB_API_KEY and SCHWAB_APP_SECRET environment variables")
        print("  set SCHWAB_API_KEY=your_key")
        print("  set SCHWAB_APP_SECRET=your_secret")
        sys.exit(1)

    # Try loading existing token first
    try:
        c = auth.client_from_token_file(TOKEN_PATH, api_key, app_secret)
        print("Authenticated using saved token.")
        return c
    except Exception:
        pass

    # First-time auth — opens browser
    print("First-time authentication — a browser window will open.")
    print("Log in to Schwab, then copy the redirect URL back here.")
    try:
        c = auth.client_from_login_flow(
            api_key, app_secret, CALLBACK_URL, TOKEN_PATH
        )
        print("Authentication successful! Token saved.")
        return c
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("\nMake sure your callback URL matches what's registered in your Schwab app.")
        print(f"Current callback URL: {CALLBACK_URL}")
        sys.exit(1)


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_5min_bars(c, symbol, start_date, end_date):
    """
    Fetch 5-minute bars for a symbol in chunks.

    Schwab limits minute-level requests to ~10 days per call,
    so we loop in chunks.
    """
    all_candles = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_date)

        try:
            resp = c.get_price_history_every_five_minutes(
                symbol,
                start_datetime=current_start,
                end_datetime=current_end,
                need_extended_hours_data=False  # RTH only
            )

            if resp.status_code != 200:
                print(f"    Chunk {current_start.date()} -> {current_end.date()}: HTTP {resp.status_code}")
                current_start = current_end
                continue

            data = resp.json()
            candles = data.get("candles", [])

            if candles:
                all_candles.extend(candles)
                dates = [datetime.fromtimestamp(c["datetime"] / 1000) for c in candles]
                print(f"    {current_start.date()} -> {current_end.date()}: {len(candles)} bars")
            else:
                print(f"    {current_start.date()} -> {current_end.date()}: no data")

        except Exception as e:
            print(f"    {current_start.date()} -> {current_end.date()}: error - {e}")

        current_start = current_end

    return all_candles


def candles_to_dataframe(candles, symbol):
    """Convert Schwab candle data to a pandas DataFrame."""
    if not candles:
        return pd.DataFrame()

    rows = []
    for c in candles:
        ts = datetime.fromtimestamp(c["datetime"] / 1000)
        rows.append({
            "datetime": ts,
            "Open": c["open"],
            "High": c["high"],
            "Low": c["low"],
            "Close": c["close"],
            "Volume": c["volume"],
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("SCHWAB DATA FETCHER — NQ & ES 5-min bars")
    print("=" * 60)

    c = get_client()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for symbol in SYMBOLS:
        clean_name = symbol.replace("/", "")
        print(f"\nFetching {symbol} ({start_date.date()} -> {end_date.date()})...")

        candles = fetch_5min_bars(c, symbol, start_date, end_date)
        print(f"  Total candles: {len(candles)}")

        if not candles:
            print(f"  No data for {symbol}, skipping.")
            continue

        df = candles_to_dataframe(candles, symbol)
        print(f"  DataFrame: {len(df)} rows, {df.index[0]} -> {df.index[-1]}")

        # Save to CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{clean_name}_5min.csv")
        df.to_csv(csv_path)
        print(f"  Saved to {csv_path}")

        # Also save daily bars (derived from 5-min for ATR calc)
        daily = df.resample("D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()
        daily_path = os.path.join(OUTPUT_DIR, f"{clean_name}_daily.csv")
        daily.to_csv(daily_path)
        print(f"  Saved daily to {daily_path}")

    print(f"\nDone! CSVs saved to {OUTPUT_DIR}/")
    print("Now run: python orb_local_pipeline.py --csv")


if __name__ == "__main__":
    main()
