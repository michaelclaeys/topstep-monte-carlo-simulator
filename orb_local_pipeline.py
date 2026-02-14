"""
ORB BREAKOUT — NQ & ES FUTURES
================================
Two-boundary evaluation: after breakout, set stop at opposite OR boundary,
track whether target (1R, 1.5R, 2R, 3R) or stop gets hit first.

Uses yfinance for minute-level data (~60 days).

USAGE:
    pip install yfinance pandas numpy scikit-learn scipy
    python orb_local_pipeline.py
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
SYMBOLS = ["NQ=F", "ES=F"]

USE_5MIN = True
OR_MINUTES = 30           # Opening range window (minutes from open)
MIN_OR_SIZE_PTS = {       # Minimum OR size in points (filter out tiny ranges)
    "NQ=F": 30,
    "ES=F": 5,
}
MAX_OR_SIZE_PTS = {       # Maximum OR size (filter out insane gaps/news)
    "NQ=F": 500,
    "ES=F": 60,
}
ATR_PERIOD = 14

# R:R targets to evaluate (multiples of risk where risk = OR range)
RR_TARGETS = [0.5, 1.0, 1.5, 2.0, 3.0]

OUTPUT_FILE = "orb_events.json"


# ============================================================
# DATA COLLECTION
# ============================================================
def download_data(symbols, use_5min=True):
    """Download intraday + daily data from yfinance."""
    interval = "5m" if use_5min else "1m"
    period = "60d"

    print(f"Downloading {interval} data for {len(symbols)} symbols ({period} lookback)...")

    intraday = {}
    daily = {}

    for ticker in symbols:
        try:
            tk = yf.Ticker(ticker)

            df = tk.history(period=period, interval=interval)
            if df.empty:
                print(f"  {ticker}: No intraday data, skipping")
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York")
            else:
                df.index = df.index.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
            intraday[ticker] = df
            print(f"  {ticker}: {len(df)} intraday bars ({df.index[0].date()} -> {df.index[-1].date()})")

            # Daily data (for ATR)
            dfd = tk.history(period="2y", interval="1d")
            if dfd.index.tz is not None:
                dfd.index = dfd.index.tz_convert("America/New_York").tz_localize(None)
            daily[ticker] = dfd

        except Exception as e:
            print(f"  {ticker}: Error - {e}")

    return intraday, daily


def load_csv_data(data_dir="data"):
    """Load intraday + daily data from CSVs (from schwab_fetch.py output)."""
    import os
    import glob

    print(f"Loading CSV data from {data_dir}/...")

    intraday = {}
    daily = {}

    # Map clean names back to ticker format
    ticker_map = {"NQ": "NQ=F", "ES": "ES=F"}

    for csv_path in glob.glob(os.path.join(data_dir, "*_5min.csv")):
        basename = os.path.basename(csv_path)
        clean_name = basename.replace("_5min.csv", "")
        ticker = ticker_map.get(clean_name, clean_name)

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.empty:
            print(f"  {ticker}: Empty CSV, skipping")
            continue

        # Localize to ET if naive
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        intraday[ticker] = df
        print(f"  {ticker}: {len(df)} intraday bars ({df.index[0].date()} -> {df.index[-1].date()})")

        # Load or derive daily
        daily_path = os.path.join(data_dir, f"{clean_name}_daily.csv")
        if os.path.exists(daily_path):
            dfd = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        else:
            dfd = df.resample("D").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
        # Daily must be tz-naive for date comparisons
        if dfd.index.tz is not None:
            dfd.index = dfd.index.tz_localize(None)
        daily[ticker] = dfd

    return intraday, daily


def compute_daily_indicators(daily_df, atr_period=14):
    """Compute ATR, SMA, RSI on daily data."""
    df = daily_df.copy()

    df["prev_close"] = df["Close"].shift(1)
    df["tr"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["prev_close"]),
            abs(df["Low"] - df["prev_close"])
        )
    )
    df["atr"] = df["tr"].rolling(atr_period).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["bb_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    return df


def get_vix_data():
    """Download VIX for regime context."""
    try:
        vix = yf.Ticker("^VIX")
        df = vix.history(period="2y", interval="1d")
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        return df["Close"]
    except:
        return pd.Series(dtype=float)


def evaluate_two_boundary(post_breakout_df, breakout_price, stop_price,
                          breakout_direction, or_range, rr_targets):
    """
    Bar-by-bar: check if target or stop gets hit first.

    Returns dict with results for each R:R target.
    For each target: 'win', 'loss', or 'timeout' (EOD).
    Also returns MFE/MAE in R-multiples.
    """
    risk = abs(breakout_price - stop_price)
    if risk <= 0:
        return None

    results = {}
    target_prices = {}
    target_hit = {}

    for rr in rr_targets:
        if breakout_direction == "long":
            target_prices[rr] = breakout_price + rr * risk
        else:
            target_prices[rr] = breakout_price - rr * risk
        target_hit[rr] = None  # None = not resolved yet

    mfe = 0.0  # max favorable excursion in R
    mae = 0.0  # max adverse excursion in R

    for _, bar in post_breakout_df.iterrows():
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])

        # Update MFE/MAE
        if breakout_direction == "long":
            favorable = (bar_high - breakout_price) / risk
            adverse = (breakout_price - bar_low) / risk
        else:
            favorable = (breakout_price - bar_low) / risk
            adverse = (bar_high - breakout_price) / risk
        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        # Check stop first (conservative — if both hit same bar, stop wins)
        stop_hit_this_bar = False
        if breakout_direction == "long" and bar_low <= stop_price:
            stop_hit_this_bar = True
        elif breakout_direction == "short" and bar_high >= stop_price:
            stop_hit_this_bar = True

        for rr in rr_targets:
            if target_hit[rr] is not None:
                continue  # Already resolved

            if stop_hit_this_bar:
                target_hit[rr] = "loss"
                continue

            # Check target
            if breakout_direction == "long" and bar_high >= target_prices[rr]:
                target_hit[rr] = "win"
            elif breakout_direction == "short" and bar_low <= target_prices[rr]:
                target_hit[rr] = "win"

        # If stop was hit, all unresolved targets are losses
        if stop_hit_this_bar:
            for rr in rr_targets:
                if target_hit[rr] is None:
                    target_hit[rr] = "loss"
            break

    # Anything still None = timeout (EOD close)
    # Calculate EOD P&L
    if len(post_breakout_df) > 0:
        eod_close = float(post_breakout_df["Close"].iloc[-1])
        if breakout_direction == "long":
            eod_pnl_r = (eod_close - breakout_price) / risk
        else:
            eod_pnl_r = (breakout_price - eod_close) / risk
    else:
        eod_pnl_r = 0

    for rr in rr_targets:
        if target_hit[rr] is None:
            target_hit[rr] = "timeout"

    results["outcomes"] = target_hit
    results["mfe_r"] = round(mfe, 4)
    results["mae_r"] = round(mae, 4)
    results["eod_pnl_r"] = round(eod_pnl_r, 4)
    results["risk_pts"] = round(risk, 2)
    return results


def collect_orb_events(intraday, daily, vix_data, use_5min=True):
    """Scan NQ/ES and collect ORB breakout events with two-boundary outcomes."""
    bar_minutes = 5 if use_5min else 1
    or_bars_needed = OR_MINUTES // bar_minutes

    all_events = []

    for ticker, idf in intraday.items():
        daily_ind = compute_daily_indicators(daily[ticker])
        min_or = MIN_OR_SIZE_PTS.get(ticker, 5)
        max_or = MAX_OR_SIZE_PTS.get(ticker, 500)

        idf["date"] = idf.index.date
        trading_days = idf.groupby("date")

        prev_day_stats = deque(maxlen=20)
        prev_close = None
        prev_high = None
        prev_low = None
        prev_volume = None

        for day_date, day_df in trading_days:
            # Regular trading hours for futures: 9:30 - 16:00 ET
            day_df = day_df.between_time("09:30", "15:59")
            if len(day_df) < or_bars_needed + 2:
                if len(day_df) > 0:
                    prev_close = float(day_df["Close"].iloc[-1])
                    prev_high = float(day_df["High"].max())
                    prev_low = float(day_df["Low"].min())
                    prev_volume = float(day_df["Volume"].sum())
                continue

            # -- Build Opening Range --
            or_df = day_df.iloc[:or_bars_needed]
            or_high = float(or_df["High"].max())
            or_low = float(or_df["Low"].min())
            or_open = float(or_df["Open"].iloc[0])
            or_close = float(or_df["Close"].iloc[-1])
            or_range = or_high - or_low
            or_volume = float(or_df["Volume"].sum())

            # Filter by OR size in points
            if or_range < min_or or or_range > max_or:
                prev_close = float(day_df["Close"].iloc[-1])
                prev_high = float(day_df["High"].max())
                prev_low = float(day_df["Low"].min())
                prev_volume = float(day_df["Volume"].sum())
                continue

            # Get daily ATR
            try:
                day_pd = pd.Timestamp(day_date)
                if day_pd in daily_ind.index:
                    atr_val = float(daily_ind.loc[day_pd, "atr"])
                else:
                    prior = daily_ind.index[daily_ind.index <= day_pd]
                    if len(prior) == 0:
                        prev_close = float(day_df["Close"].iloc[-1])
                        prev_high = float(day_df["High"].max())
                        prev_low = float(day_df["Low"].min())
                        prev_volume = float(day_df["Volume"].sum())
                        continue
                    atr_val = float(daily_ind.loc[prior[-1], "atr"])
            except:
                prev_close = float(day_df["Close"].iloc[-1])
                prev_high = float(day_df["High"].max())
                prev_low = float(day_df["Low"].min())
                prev_volume = float(day_df["Volume"].sum())
                continue

            if np.isnan(atr_val) or atr_val <= 0:
                prev_close = float(day_df["Close"].iloc[-1])
                prev_high = float(day_df["High"].max())
                prev_low = float(day_df["Low"].min())
                prev_volume = float(day_df["Volume"].sum())
                continue

            or_range_atr = or_range / atr_val

            # -- Detect Breakout --
            post_or_df = day_df.iloc[or_bars_needed:]
            breakout_triggered = False
            breakout_direction = None
            breakout_idx = None
            breakout_price = None
            breakout_bar_vol = 0

            for i, (ts, row) in enumerate(post_or_df.iterrows()):
                if float(row["Close"]) > or_high:
                    breakout_triggered = True
                    breakout_direction = "long"
                    breakout_idx = i
                    breakout_price = float(row["Close"])
                    breakout_bar_vol = float(row["Volume"])
                    break
                elif float(row["Close"]) < or_low:
                    breakout_triggered = True
                    breakout_direction = "short"
                    breakout_idx = i
                    breakout_price = float(row["Close"])
                    breakout_bar_vol = float(row["Volume"])
                    break

            if not breakout_triggered:
                prev_close = float(day_df["Close"].iloc[-1])
                prev_high = float(day_df["High"].max())
                prev_low = float(day_df["Low"].min())
                prev_volume = float(day_df["Volume"].sum())
                continue

            # -- Two-Boundary Evaluation --
            # Stop = opposite OR boundary
            if breakout_direction == "long":
                stop_price = or_low
            else:
                stop_price = or_high

            # Evaluate from breakout bar+1 through end of day
            post_breakout = post_or_df.iloc[breakout_idx + 1:]

            if len(post_breakout) < 1:
                prev_close = float(day_df["Close"].iloc[-1])
                prev_high = float(day_df["High"].max())
                prev_low = float(day_df["Low"].min())
                prev_volume = float(day_df["Volume"].sum())
                continue

            boundary_result = evaluate_two_boundary(
                post_breakout, breakout_price, stop_price,
                breakout_direction, or_range, RR_TARGETS
            )
            if boundary_result is None:
                prev_close = float(day_df["Close"].iloc[-1])
                prev_high = float(day_df["High"].max())
                prev_low = float(day_df["Low"].min())
                prev_volume = float(day_df["Volume"].sum())
                continue

            # -- Feature Extraction --
            or_up_bars = sum(1 for _, r in or_df.iterrows() if r["Close"] > r["Open"])
            or_down_bars = or_bars_needed - or_up_bars
            or_high_bar_idx = int(or_df["High"].argmax())
            or_low_bar_idx = int(or_df["Low"].argmin())
            or_trend = (or_close - or_open) / or_range if or_range > 0 else 0

            half = or_bars_needed // 2
            first_half_vol = float(or_df.iloc[:half]["Volume"].sum())
            second_half_vol = float(or_df.iloc[half:]["Volume"].sum())
            or_vol_ratio = first_half_vol / second_half_vol if second_half_vol > 0 else 1.0

            breakout_time_min = OR_MINUTES + breakout_idx * bar_minutes
            avg_or_bar_vol = or_volume / or_bars_needed if or_bars_needed > 0 else 1
            breakout_vol_ratio = breakout_bar_vol / avg_or_bar_vol if avg_or_bar_vol > 0 else 1

            if breakout_direction == "long":
                breakout_distance_pct = (breakout_price - or_high) / or_range
            else:
                breakout_distance_pct = (or_low - breakout_price) / or_range

            gap = 0
            gap_pct = 0
            if prev_close and prev_close > 0:
                gap = or_open - prev_close
                gap_pct = gap / prev_close * 100

            gap_vs_or = abs(gap) / or_range if or_range > 0 else 0
            gap_aligned = 0
            if gap > 0 and breakout_direction == "long":
                gap_aligned = 1
            elif gap < 0 and breakout_direction == "short":
                gap_aligned = 1

            prev_range_atr = 0
            prev_close_position = 0
            if prev_high and prev_low and prev_close:
                prev_range = prev_high - prev_low
                prev_range_atr = prev_range / atr_val
                if prev_range > 0:
                    prev_close_position = (prev_close - prev_low) / prev_range

            consecutive_up = 0
            consecutive_down = 0
            avg_daily_range = 0
            avg_daily_volume = 0
            if prev_day_stats:
                for d in reversed(list(prev_day_stats)):
                    if d["close"] > d["open"]:
                        consecutive_up += 1
                    else:
                        break
                for d in reversed(list(prev_day_stats)):
                    if d["close"] < d["open"]:
                        consecutive_down += 1
                    else:
                        break
                avg_daily_range = np.mean([d["range"] for d in prev_day_stats])
                avg_daily_volume = np.mean([d["volume"] for d in prev_day_stats])

            time_fraction = OR_MINUTES / 390
            vol_vs_avg_adj = (or_volume / time_fraction) / avg_daily_volume if avg_daily_volume > 0 else 1

            # Daily indicators
            try:
                day_pd = pd.Timestamp(day_date)
                prior = daily_ind.index[daily_ind.index <= day_pd]
                if len(prior) > 0:
                    d_row = daily_ind.loc[prior[-1]]
                    sma20 = float(d_row["sma_20"]) if not np.isnan(d_row["sma_20"]) else None
                    sma50 = float(d_row["sma_50"]) if not np.isnan(d_row["sma_50"]) else None
                    rsi = float(d_row["rsi"]) if not np.isnan(d_row["rsi"]) else None
                    bb_width = float(d_row["bb_width"]) if not np.isnan(d_row["bb_width"]) else None
                else:
                    sma20 = sma50 = rsi = bb_width = None
            except:
                sma20 = sma50 = rsi = bb_width = None

            price_vs_sma20 = (breakout_price - sma20) / atr_val if sma20 else None
            price_vs_sma50 = (breakout_price - sma50) / atr_val if sma50 else None
            sma_trend = 1 if (sma20 and sma50 and sma20 > sma50) else 0

            vix_val = None
            try:
                day_pd = pd.Timestamp(day_date)
                prior_vix = vix_data.index[vix_data.index.date <= day_date]
                if len(prior_vix) > 0:
                    vix_val = float(vix_data.loc[prior_vix[-1]])
            except:
                pass

            # VWAP approx from OR
            vwap_or = (or_df["Close"] * or_df["Volume"]).sum() / or_volume if or_volume > 0 else or_close
            price_vs_vwap = (breakout_price - vwap_or) / atr_val

            # OR boundary touches
            touch_threshold = 0.1 * atr_val
            touches_high = sum(1 for _, r in or_df.iterrows() if abs(float(r["High"]) - or_high) < touch_threshold)
            touches_low = sum(1 for _, r in or_df.iterrows() if abs(float(r["Low"]) - or_low) < touch_threshold)

            or_vs_prev_high = (or_high - prev_high) / atr_val if prev_high else None
            or_vs_prev_low = (or_low - prev_low) / atr_val if prev_low else None

            dow = pd.Timestamp(day_date).weekday()

            # -- Build event --
            event = {
                "symbol": ticker,
                "date": str(day_date),
                "direction": breakout_direction,
                "breakout_price": round(breakout_price, 2),
                "stop_price": round(stop_price, 2),
                "or_high": round(or_high, 2),
                "or_low": round(or_low, 2),
                "or_range_pts": round(or_range, 2),
                "risk_pts": boundary_result["risk_pts"],
                "mfe_r": boundary_result["mfe_r"],
                "mae_r": boundary_result["mae_r"],
                "eod_pnl_r": boundary_result["eod_pnl_r"],
                # Outcomes at each R:R
                **{f"rr_{rr}_outcome": boundary_result["outcomes"][rr] for rr in RR_TARGETS},
                # Features
                "or_range_atr": round(or_range_atr, 4),
                "or_trend": round(or_trend, 4),
                "or_up_bars": or_up_bars,
                "or_down_bars": or_down_bars,
                "or_high_bar_idx": or_high_bar_idx,
                "or_low_bar_idx": or_low_bar_idx,
                "or_vol_ratio_halves": round(or_vol_ratio, 4),
                "or_volume": round(or_volume, 0),
                "touches_high": touches_high,
                "touches_low": touches_low,
                "breakout_time_min": breakout_time_min,
                "breakout_vol_ratio": round(breakout_vol_ratio, 4),
                "breakout_distance_pct": round(breakout_distance_pct, 4),
                "gap_pct": round(gap_pct, 4),
                "gap_vs_or": round(gap_vs_or, 4),
                "gap_aligned": gap_aligned,
                "prev_range_atr": round(prev_range_atr, 4),
                "prev_close_position": round(prev_close_position, 4),
                "consecutive_up_days": consecutive_up,
                "consecutive_down_days": consecutive_down,
                "vol_vs_avg_adj": round(vol_vs_avg_adj, 4),
                "price_vs_sma20_atr": round(price_vs_sma20, 4) if price_vs_sma20 is not None else None,
                "price_vs_sma50_atr": round(price_vs_sma50, 4) if price_vs_sma50 is not None else None,
                "sma_trend_up": sma_trend,
                "rsi_14": round(rsi, 2) if rsi else None,
                "bb_width": round(bb_width, 4) if bb_width is not None else None,
                "vix": round(vix_val, 2) if vix_val else None,
                "day_of_week": dow,
                "price_vs_vwap_atr": round(price_vs_vwap, 4),
                "or_vs_prev_high_atr": round(or_vs_prev_high, 4) if or_vs_prev_high is not None else None,
                "or_vs_prev_low_atr": round(or_vs_prev_low, 4) if or_vs_prev_low is not None else None,
            }
            all_events.append(event)

            # Update prev day stats
            prev_close = float(day_df["Close"].iloc[-1])
            prev_high = float(day_df["High"].max())
            prev_low = float(day_df["Low"].min())
            prev_volume = float(day_df["Volume"].sum())
            prev_day_stats.append({
                "open": float(day_df["Open"].iloc[0]),
                "close": prev_close,
                "high": prev_high,
                "low": prev_low,
                "volume": prev_volume,
                "range": prev_high - prev_low,
            })

        print(f"  {ticker}: {sum(1 for e in all_events if e['symbol'] == ticker)} events")

    return all_events


# ============================================================
# ML ANALYSIS
# ============================================================
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text
from scipy import stats


def run_analysis(events):
    """Run full analysis on collected events."""
    df = pd.DataFrame(events)

    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "date", "direction"], keep="first")
    after = len(df)
    if before != after:
        print(f"\nDeduplicated: {before} -> {after} events")

    print(f"\n{'='*70}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*70}")
    print(f"Total events:    {len(df)}")
    print(f"Symbols:         {', '.join(df['symbol'].unique())}")
    print(f"Date range:      {df['date'].min()} -> {df['date'].max()}")

    # ── 0. TWO-BOUNDARY RESULTS ──
    print(f"\n{'='*70}")
    print("0. TWO-BOUNDARY RESULTS (Stop at opposite OR boundary)")
    print(f"{'='*70}")

    print(f"\n  Risk = OR range (stop at opposite boundary)")
    print(f"  Avg risk: {df['risk_pts'].mean():.1f} pts")
    print(f"  Avg MFE:  {df['mfe_r'].mean():.2f} R")
    print(f"  Avg MAE:  {df['mae_r'].mean():.2f} R")
    print(f"  Avg EOD:  {df['eod_pnl_r'].mean():.3f} R")

    print(f"\n  {'R:R Target':<12} {'Win%':>8} {'Loss%':>8} {'Timeout%':>10} {'EV (R)':>10}  n={len(df)}")
    print(f"  {'-'*55}")

    best_rr = None
    best_ev = -999

    for rr in RR_TARGETS:
        col = f"rr_{rr}_outcome"
        if col not in df.columns:
            continue
        wins = (df[col] == "win").sum()
        losses = (df[col] == "loss").sum()
        timeouts = (df[col] == "timeout").sum()
        n = len(df)
        win_pct = wins / n
        loss_pct = losses / n
        timeout_pct = timeouts / n

        # EV = win% * reward - loss% * 1R - timeout uses EOD PnL
        timeout_mask = df[col] == "timeout"
        avg_timeout_pnl = df.loc[timeout_mask, "eod_pnl_r"].mean() if timeout_mask.sum() > 0 else 0
        ev = win_pct * rr - loss_pct * 1.0 + timeout_pct * avg_timeout_pnl

        if ev > best_ev:
            best_ev = ev
            best_rr = rr

        print(f"  {rr:>5.1f}R      {win_pct:>7.1%} {loss_pct:>8.1%} {timeout_pct:>9.1%}  {ev:>9.3f} R")

    if best_rr:
        print(f"\n  Best R:R: {best_rr}R (EV = {best_ev:.3f}R per trade)")

    # Per-symbol breakdown
    for sym in df["symbol"].unique():
        sub = df[df["symbol"] == sym]
        print(f"\n  --- {sym} ({len(sub)} events, avg risk: {sub['risk_pts'].mean():.1f} pts) ---")
        print(f"  {'R:R Target':<12} {'Win%':>8} {'Loss%':>8} {'Timeout%':>10} {'EV (R)':>10}")
        for rr in RR_TARGETS:
            col = f"rr_{rr}_outcome"
            wins = (sub[col] == "win").sum()
            losses = (sub[col] == "loss").sum()
            timeouts = (sub[col] == "timeout").sum()
            n = len(sub)
            win_pct = wins / n
            loss_pct = losses / n
            timeout_pct = timeouts / n
            timeout_mask = sub[col] == "timeout"
            avg_to = sub.loc[timeout_mask, "eod_pnl_r"].mean() if timeout_mask.sum() > 0 else 0
            ev = win_pct * rr - loss_pct * 1.0 + timeout_pct * avg_to
            print(f"  {rr:>5.1f}R      {win_pct:>7.1%} {loss_pct:>8.1%} {timeout_pct:>9.1%}  {ev:>9.3f} R")

    # Direction breakdown
    for direction in ["long", "short"]:
        sub = df[df["direction"] == direction]
        if len(sub) < 5:
            continue
        print(f"\n  --- {direction.upper()} ({len(sub)} events) ---")
        print(f"  {'R:R Target':<12} {'Win%':>8} {'Loss%':>8} {'EV (R)':>10}")
        for rr in RR_TARGETS:
            col = f"rr_{rr}_outcome"
            wins = (sub[col] == "win").sum()
            losses = (sub[col] == "loss").sum()
            timeouts = (sub[col] == "timeout").sum()
            n = len(sub)
            win_pct = wins / n
            loss_pct = losses / n
            timeout_pct = timeouts / n
            timeout_mask = sub[col] == "timeout"
            avg_to = sub.loc[timeout_mask, "eod_pnl_r"].mean() if timeout_mask.sum() > 0 else 0
            ev = win_pct * rr - loss_pct * 1.0 + timeout_pct * avg_to
            print(f"  {rr:>5.1f}R      {win_pct:>7.1%} {loss_pct:>8.1%}  {ev:>9.3f} R")

    # ── Use best R:R as the target for ML analysis ──
    # Pick 1R as the ML target (most balanced)
    ml_target_rr = 1.0
    target_col = f"rr_{ml_target_rr}_outcome"
    # For ML: success = win at this R:R
    df["success"] = (df[target_col] == "win").astype(int)
    success_rate = df["success"].mean()

    feature_cols = [
        "or_range_atr", "or_range_pts", "or_trend", "or_up_bars", "or_down_bars",
        "or_high_bar_idx", "or_low_bar_idx", "or_vol_ratio_halves",
        "or_volume", "touches_high", "touches_low",
        "breakout_time_min", "breakout_vol_ratio", "breakout_distance_pct",
        "gap_pct", "gap_vs_or", "gap_aligned",
        "prev_range_atr", "prev_close_position",
        "consecutive_up_days", "consecutive_down_days",
        "vol_vs_avg_adj",
        "price_vs_sma20_atr", "price_vs_sma50_atr", "sma_trend_up",
        "rsi_14", "bb_width",
        "vix", "day_of_week", "price_vs_vwap_atr",
        "or_vs_prev_high_atr", "or_vs_prev_low_atr",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"\n{'='*70}")
    print(f"ML TARGET: Win at {ml_target_rr}R (stop at opposite OR boundary)")
    print(f"{'='*70}")
    print(f"Features:        {len(feature_cols)}")
    print(f"Base win rate:   {success_rate:.1%}")

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    X = df[feature_cols].values
    y = df["success"].values

    # ── 1. UNIVARIATE ANALYSIS ──
    print(f"\n{'='*70}")
    print("1. UNIVARIATE ANALYSIS: Winners vs Losers")
    print(f"{'='*70}")

    winners = df[df["success"] == 1]
    losers = df[df["success"] == 0]
    uni_results = []

    for col in feature_cols:
        w = winners[col].dropna()
        l = losers[col].dropna()
        if len(w) < 5 or len(l) < 5:
            continue
        try:
            _, pval = stats.mannwhitneyu(w, l, alternative="two-sided")
        except:
            continue
        pooled_std = np.sqrt((w.std()**2 + l.std()**2) / 2)
        d = (w.mean() - l.mean()) / pooled_std if pooled_std > 0 else 0
        uni_results.append({"feature": col, "win_mean": w.mean(), "loss_mean": l.mean(),
                            "cohens_d": d, "p_value": pval})

    if uni_results:
        uni_df = pd.DataFrame(uni_results).sort_values("p_value")
        print(f"\n{'Feature':<30} {'Win Mean':>10} {'Loss Mean':>10} {'Cohen d':>10} {'p-value':>10}")
        print("-" * 75)
        for _, r in uni_df.iterrows():
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
            print(f"{r['feature']:<30} {r['win_mean']:>10.4f} {r['loss_mean']:>10.4f} {r['cohens_d']:>10.4f} {r['p_value']:>10.4f} {sig}")

        sig_count = sum(1 for _, r in uni_df.iterrows() if r["p_value"] < 0.05)
        print(f"\n{sig_count} / {len(uni_df)} features statistically significant (p < 0.05)")
    else:
        print("\n  Not enough winners/losers for univariate analysis (need >= 5 each)")

    # ── 2. ML FEATURE IMPORTANCE ──
    print(f"\n{'='*70}")
    print("2. ML FEATURE IMPORTANCE")
    print(f"{'='*70}")

    min_class_count = min(y.sum(), len(y) - y.sum())
    if min_class_count < 10:
        print(f"\n  Skipping ML: only {int(y.sum())} wins / {int(len(y) - y.sum())} non-wins")
        print("  Need at least 10 in each class.")
        all_imp = {}
    else:
        n_splits = min(5, int(min_class_count))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=8,
                              min_samples_leaf=10, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                  min_samples_leaf=10, learning_rate=0.05, subsample=0.8, random_state=42),
        }

        all_imp = {}
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            print(f"\n{name}: ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

            model.fit(X, y)
            perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="roc_auc")
            all_imp[name] = perm.importances_mean

            top_idx = np.argsort(perm.importances_mean)[::-1][:15]
            print(f"  Top 15 features:")
            for rank, idx in enumerate(top_idx, 1):
                print(f"    {rank:>2}. {feature_cols[idx]:<30} {perm.importances_mean[idx]:.4f}")

        avg_imp = np.mean(list(all_imp.values()), axis=0)
        ranking = np.argsort(avg_imp)[::-1]
        print(f"\nCONSENSUS RANKING:")
        for rank, idx in enumerate(ranking[:20], 1):
            print(f"  {rank:>2}. {feature_cols[idx]:<30} {avg_imp[idx]:.4f}")

    # ── 3. DECISION TREE RULES ──
    print(f"\n{'='*70}")
    print("3. DECISION TREE RULES")
    print(f"{'='*70}")

    if min_class_count < 10:
        print("  Skipping: insufficient class balance.")
    else:
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
        tree.fit(X, y)
        print(export_text(tree, feature_names=feature_cols, max_depth=4, decimals=3))

    # ── 4. REGIME ANALYSIS ──
    print(f"\n{'='*70}")
    print("4. REGIME ANALYSIS (at 1R target)")
    print(f"{'='*70}")

    print("\nBy Direction:")
    for d in ["long", "short"]:
        sub = df[df["direction"] == d]
        if len(sub) > 0:
            print(f"  {d:<6} WR={sub['success'].mean():.1%}  avg_mfe={sub['mfe_r'].mean():.2f}R  avg_mae={sub['mae_r'].mean():.2f}R  n={len(sub)}")

    print("\nBy Day of Week:")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for dow in range(5):
        sub = df[df["day_of_week"] == dow]
        if len(sub) >= 5:
            print(f"  {days[dow]:<5} WR={sub['success'].mean():.1%}  avg_eod={sub['eod_pnl_r'].mean():.3f}R  n={len(sub)}")

    print("\nBy Breakout Time:")
    df["bo_bucket"] = pd.cut(df["breakout_time_min"],
        bins=[0, 35, 45, 60, 90, 120, 390],
        labels=["30-35m", "35-45m", "45-60m", "60-90m", "90-120m", "120m+"])
    for bucket in ["30-35m", "35-45m", "45-60m", "60-90m", "90-120m", "120m+"]:
        sub = df[df["bo_bucket"] == bucket]
        if len(sub) >= 5:
            print(f"  {bucket:<10} WR={sub['success'].mean():.1%}  avg_eod={sub['eod_pnl_r'].mean():.3f}R  n={len(sub)}")

    print("\nBy Symbol:")
    for sym in df["symbol"].unique():
        sub = df[df["symbol"] == sym]
        if len(sub) >= 5:
            print(f"  {sym:<8} WR={sub['success'].mean():.1%}  avg_eod={sub['eod_pnl_r'].mean():.3f}R  n={len(sub)}")

    # ── 5. TOP SETUPS ──
    print(f"\n{'='*70}")
    print("5. TOP SETUPS (Best Single-Feature Filters at 1R)")
    print(f"{'='*70}")

    base_wr = success_rate
    filter_results = []

    for col in feature_cols:
        for pct in [25, 75]:
            threshold = df[col].quantile(pct / 100)
            mask = df[col] <= threshold if pct == 25 else df[col] >= threshold
            label = f"{col} {'<=' if pct == 25 else '>='} {threshold:.3f}"
            sub = df[mask]
            if len(sub) >= 15:
                wr = sub["success"].mean()
                avg_eod = sub["eod_pnl_r"].mean()
                edge = wr - base_wr
                if abs(edge) > 0.03:
                    filter_results.append({"filter": label, "win_rate": wr, "edge": edge,
                                           "avg_eod_r": avg_eod, "n": len(sub)})

    filter_results.sort(key=lambda x: x["edge"], reverse=True)

    print(f"\nBase: WR={base_wr:.1%}\n")
    if filter_results:
        print("BEST filters:")
        for r in filter_results[:10]:
            print(f"  {r['filter']:<45} WR={r['win_rate']:.1%} (edge={r['edge']:+.1%})  EOD={r['avg_eod_r']:.3f}R  n={r['n']}")

        print("\nWORST filters (avoid):")
        for r in filter_results[-5:]:
            print(f"  {r['filter']:<45} WR={r['win_rate']:.1%} (edge={r['edge']:+.1%})  EOD={r['avg_eod_r']:.3f}R  n={r['n']}")
    else:
        print("  No filters found with significant edge.")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ORB BREAKOUT — NQ & ES FUTURES")
    print("=" * 70)

    use_csv = "--csv" in sys.argv
    json_file = None
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            json_file = arg

    if json_file:
        # Load pre-computed events
        print(f"\nLoading existing data from {json_file}...")
        with open(json_file) as f:
            events = json.load(f)
        print(f"Loaded {len(events)} events")
    elif use_csv:
        # Load from Schwab CSV export
        vix_data = get_vix_data()
        intraday, daily = load_csv_data("data")
        print(f"\nCollecting ORB events from CSV data...")
        events = collect_orb_events(intraday, daily, vix_data, USE_5MIN)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(events, f, indent=2)
        print(f"\nSaved {len(events)} events to {OUTPUT_FILE}")
    else:
        # Download from yfinance (60-day limit)
        vix_data = get_vix_data()
        intraday, daily = download_data(SYMBOLS, USE_5MIN)
        print(f"\nCollecting ORB events...")
        events = collect_orb_events(intraday, daily, vix_data, USE_5MIN)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(events, f, indent=2)
        print(f"\nSaved {len(events)} events to {OUTPUT_FILE}")

    run_analysis(events)
