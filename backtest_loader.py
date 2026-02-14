"""
Backtest Data Loader

Load real backtest trade data from CSV files and extract
distribution parameters for simulation.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd

from .trade_generator import StrategyParams, BootstrapTradeGenerator


@dataclass
class BacktestStats:
    """Statistics extracted from backtest data."""
    total_trades: int
    total_days: int
    trades_per_day: float
    win_rate: float
    avg_win: float
    avg_loss: float
    win_std: float
    loss_std: float
    per_trade_ev: float
    per_trade_std: float
    daily_ev: float
    daily_std: float
    max_win: float
    max_loss: float
    profit_factor: float
    sharpe_ratio: float  # Annualized, assuming 252 trading days

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Backtest Statistics:\n"
            f"  Total Trades:    {self.total_trades:,}\n"
            f"  Total Days:      {self.total_days}\n"
            f"  Trades/Day:      {self.trades_per_day:.1f}\n"
            f"\n"
            f"  Win Rate:        {self.win_rate:.1%}\n"
            f"  Avg Win:         ${self.avg_win:.2f}\n"
            f"  Avg Loss:        ${self.avg_loss:.2f}\n"
            f"  Win Std:         ${self.win_std:.2f}\n"
            f"  Loss Std:        ${self.loss_std:.2f}\n"
            f"\n"
            f"  Per-Trade EV:    ${self.per_trade_ev:.2f}\n"
            f"  Per-Trade Std:   ${self.per_trade_std:.2f}\n"
            f"  Daily EV:        ${self.daily_ev:.2f}\n"
            f"  Daily Std:       ${self.daily_std:.2f}\n"
            f"\n"
            f"  Max Win:         ${self.max_win:.2f}\n"
            f"  Max Loss:        ${self.max_loss:.2f}\n"
            f"  Profit Factor:   {self.profit_factor:.2f}\n"
            f"  Sharpe Ratio:    {self.sharpe_ratio:.2f}"
        )


def load_backtest_csv(
    filepath: str,
    date_column: str = 'date',
    pnl_column: str = 'trade_pnl',
    date_format: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load backtest trade data from CSV file.

    Expected CSV format:
        date,trade_pnl
        2024-01-02,45.50
        2024-01-02,-30.00
        ...

    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        pnl_column: Name of P&L column
        date_format: Optional date format string for parsing

    Returns:
        Tuple of (trade_pnl_array, full_dataframe)
    """
    df = pd.read_csv(filepath)

    # Parse dates
    if date_format:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])

    # Extract P&L values
    trades = df[pnl_column].values.astype(float)

    return trades, df


def compute_backtest_stats(
    trades: np.ndarray,
    df: Optional[pd.DataFrame] = None,
    date_column: str = 'date'
) -> BacktestStats:
    """
    Compute comprehensive statistics from backtest trades.

    Args:
        trades: Array of trade P&L values
        df: Optional DataFrame with date information
        date_column: Name of date column in df

    Returns:
        BacktestStats with all computed statistics
    """
    trades = np.asarray(trades)

    # Basic counts
    total_trades = len(trades)

    # Compute daily stats if we have date info
    if df is not None and date_column in df.columns:
        daily_pnl = df.groupby(df[date_column].dt.date).sum()
        total_days = len(daily_pnl)
        trades_per_day = total_trades / total_days if total_days > 0 else 0
        daily_ev = daily_pnl.mean().iloc[0] if len(daily_pnl) > 0 else 0
        daily_std = daily_pnl.std().iloc[0] if len(daily_pnl) > 1 else 0
    else:
        # Estimate from trade count
        trades_per_day = 40  # Default assumption
        total_days = total_trades // trades_per_day
        daily_ev = trades.mean() * trades_per_day
        daily_std = trades.std() * np.sqrt(trades_per_day)

    # Win/loss separation
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = -losses.mean() if len(losses) > 0 else 0  # Store as positive
    win_std = wins.std() if len(wins) > 1 else 0
    loss_std = (-losses).std() if len(losses) > 1 else 0

    # Overall stats
    per_trade_ev = trades.mean() if total_trades > 0 else 0
    per_trade_std = trades.std() if total_trades > 1 else 0

    # Extremes
    max_win = wins.max() if len(wins) > 0 else 0
    max_loss = -losses.min() if len(losses) > 0 else 0  # Store as positive

    # Profit factor
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = -losses.sum() if len(losses) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Sharpe ratio (annualized, assuming 252 trading days)
    if daily_std > 0:
        sharpe_ratio = (daily_ev / daily_std) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    return BacktestStats(
        total_trades=total_trades,
        total_days=total_days,
        trades_per_day=trades_per_day,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_std=win_std,
        loss_std=loss_std,
        per_trade_ev=per_trade_ev,
        per_trade_std=per_trade_std,
        daily_ev=daily_ev,
        daily_std=daily_std,
        max_win=max_win,
        max_loss=max_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio
    )


def backtest_to_strategy_params(
    stats: BacktestStats,
    trades_per_day: Optional[int] = None,
    max_loss_per_trade: float = 100.0,
    base_contracts: int = 20
) -> StrategyParams:
    """
    Convert backtest statistics to StrategyParams.

    Args:
        stats: Backtest statistics
        trades_per_day: Override trades per day (uses stats value if None)
        max_loss_per_trade: Maximum loss per trade
        base_contracts: Base contract count

    Returns:
        StrategyParams matching the backtest statistics
    """
    tpd = trades_per_day if trades_per_day is not None else int(round(stats.trades_per_day))

    return StrategyParams(
        trades_per_day=max(1, tpd),
        win_rate=max(0.01, min(0.99, stats.win_rate)),
        avg_win=max(0.01, stats.avg_win),
        avg_loss=max(0.01, stats.avg_loss),
        win_std=max(0.01, stats.win_std),
        loss_std=max(0.01, stats.loss_std),
        max_loss_per_trade=max_loss_per_trade,
        base_contracts=base_contracts
    )


def create_bootstrap_generator(
    filepath: str,
    trades_per_day: int = 40,
    date_column: str = 'date',
    pnl_column: str = 'trade_pnl',
    rng: Optional[np.random.Generator] = None
) -> Tuple[BootstrapTradeGenerator, BacktestStats]:
    """
    Create a bootstrap trade generator from CSV file.

    Args:
        filepath: Path to CSV file
        trades_per_day: Trades to sample per day
        date_column: Name of date column
        pnl_column: Name of P&L column
        rng: Random number generator

    Returns:
        Tuple of (BootstrapTradeGenerator, BacktestStats)
    """
    trades, df = load_backtest_csv(filepath, date_column, pnl_column)
    stats = compute_backtest_stats(trades, df, date_column)

    generator = BootstrapTradeGenerator(
        historical_trades=trades,
        trades_per_day=trades_per_day,
        rng=rng
    )

    return generator, stats


def validate_backtest_data(trades: np.ndarray) -> Dict[str, Any]:
    """
    Validate backtest data and return potential issues.

    Args:
        trades: Array of trade P&L values

    Returns:
        Dictionary with validation results and warnings
    """
    issues = []
    warnings = []

    trades = np.asarray(trades)

    # Check for NaN/inf
    nan_count = np.isnan(trades).sum()
    inf_count = np.isinf(trades).sum()

    if nan_count > 0:
        issues.append(f"Found {nan_count} NaN values")
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values")

    # Check for reasonable values
    if len(trades) < 100:
        warnings.append(f"Only {len(trades)} trades - may not be statistically significant")

    if len(trades) > 0:
        # Check for outliers
        mean = trades.mean()
        std = trades.std()
        outliers = np.abs(trades - mean) > 5 * std
        outlier_count = outliers.sum()

        if outlier_count > 0:
            warnings.append(f"Found {outlier_count} extreme outliers (>5 std from mean)")

        # Check for suspiciously good performance
        wins = trades[trades > 0]
        losses = trades[trades < 0]

        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(trades)
            if win_rate > 0.90:
                warnings.append(f"Win rate {win_rate:.1%} is suspiciously high")

            avg_win = wins.mean()
            avg_loss = -losses.mean()
            if avg_win > 3 * avg_loss:
                warnings.append("Average win >> average loss - verify this is realistic")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'trade_count': len(trades)
    }


def load_and_analyze(
    filepath: str,
    date_column: str = 'date',
    pnl_column: str = 'trade_pnl'
) -> Tuple[np.ndarray, BacktestStats, Dict[str, Any]]:
    """
    Load backtest data, compute stats, and validate.

    Convenience function that does everything in one call.

    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        pnl_column: Name of P&L column

    Returns:
        Tuple of (trades, stats, validation_results)
    """
    trades, df = load_backtest_csv(filepath, date_column, pnl_column)
    validation = validate_backtest_data(trades)
    stats = compute_backtest_stats(trades, df, date_column)

    return trades, stats, validation
