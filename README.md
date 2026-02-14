# TopStep 50K Monte Carlo Simulator

Monte Carlo simulation framework for modeling the complete lifecycle of a TopStep 50K prop firm account. Built to answer one question: **what is the expected value of trading a given strategy through TopStep's rules?**

## What It Does

Simulates every phase of the TopStep 50K account lifecycle with accurate rule enforcement:

- **Trading Combine** -- Trailing MLL, daily loss limits, consistency rule (50%), profit target ($3K), minimum trading days
- **Express Funded Account (XFA)** -- Scaling plan, benchmark days, payout mechanics (50% split, $125 fee), MLL lock after first payout
- **Back2Funded (B2F)** -- Expected value calculation for the $599 reactivation option
- **Full Lifecycle** -- Combine attempts + XFA trading + payouts + blow + B2F decision, chained together with realistic cost accounting

The simulator also includes an **ORB (Opening Range Breakout) analysis pipeline** for backtesting NQ and ES futures breakout strategies using historical data.

## Project Structure

```
topstep_sim/
    config.py            # All TopStep rules, fees, scaling tables
    trade_generator.py   # Parametric + bootstrap trade generation
    combine_sim.py       # Trading Combine simulator (vectorized batch)
    xfa_sim.py           # XFA simulator with payout strategies
    lifecycle_sim.py     # Full lifecycle: Combine -> XFA -> B2F -> repeat
    optimizer.py         # Parameter sweep to maximize lifecycle EV
    analytics.py         # Visualization and reporting
    backtest_loader.py   # Load real trade data for bootstrap simulation
    main.py              # CLI interface
    tests/               # Unit tests for MLL, consistency, payouts, scaling

    orb_local_pipeline.py   # ORB breakout analysis for NQ/ES futures
    schwab_fetch.py         # Schwab API data fetcher for historical bars
```

## Quick Start

### Install

```bash
pip install numpy pandas matplotlib
```

### Run a Simulation

```bash
# Default parameters (68% WR, $30 avg win, $55 avg loss)
python -m topstep_sim simulate

# Custom strategy parameters
python -m topstep_sim simulate --win-rate 0.57 --avg-win 37 --avg-loss 38 --trades-per-day 1

# Combine phase only
python -m topstep_sim combine --n-sims 100000

# Parameter optimization
python -m topstep_sim optimize --quick

# Evaluate your own backtest CSV
python -m topstep_sim backtest --input my_trades.csv
```

### Example Output

```
TOPSTEP 50K LIFECYCLE SIMULATION
============================================================
Strategy: 57.0% WR | $37.00 avg win | $38.00 avg loss
Daily EV: $2.34 | Sharpe (daily): 0.12

Lifecycle Results (100,000 sims):
  Lifecycle EV:     $2,047.23
  Combine Pass Rate: 72.1%
  Avg Days to Pass:  34.2
  XFA Avg Payouts:   $1,891.50
  B2F Usage Rate:    41.3%
```

## ORB Breakout Analysis

The ORB pipeline evaluates Opening Range Breakout setups on NQ and ES futures using a two-boundary approach: stop at the opposite OR boundary, targets at 0.5R through 3R.

```bash
# Using free yfinance data (60-day limit)
python orb_local_pipeline.py

# Using Schwab API data (up to 2 years)
python schwab_fetch.py          # Downloads NQ/ES 5-min bars to data/
python orb_local_pipeline.py --csv
```

Features:
- Per-symbol OR size filters (points-based)
- R:R analysis at multiple target levels
- Univariate feature significance testing (Mann-Whitney U, Cohen's d)
- ML feature importance (Random Forest, permutation importance)
- Regime analysis (VIX, day-of-week, OR size quintiles)

## Key Design Decisions

- **Intraday MLL checks**: Breach is checked trade-by-trade within each day, not just at EOD
- **Post-payout MLL lock**: After first XFA payout, MLL permanently locks at $0
- **Consistency rule**: Best single profitable day cannot exceed 50% of total profit
- **Vectorized combine sim**: Batch combine simulation is fully vectorized with NumPy for performance
- **XFA sim uses loop**: Payout state transitions (MLL resets, benchmark day counters) make full vectorization impractical, so XFA uses a per-sim loop with vectorized daily trade generation

## Tests

```bash
pytest tests/ -v
```

## Requirements

- Python 3.10+
- numpy, pandas
- matplotlib (optional, for plots)
- scikit-learn, scipy (for ORB analysis)
- schwab-py (for Schwab API data fetching)
