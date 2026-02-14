"""
Trade Generator Module

Generates synthetic trade sequences from parameterized distributions.
Supports both parametric generation and bootstrap resampling from historical data.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class StrategyParams:
    """
    Parameters defining a trading strategy's statistical properties.

    All monetary values are in dollars.
    """
    trades_per_day: int = 40            # Number of trades executed per day
    win_rate: float = 0.68              # Probability of a winning trade
    avg_win: float = 30.0               # Average profit on winning trades
    avg_loss: float = 55.0              # Average loss on losing trades (positive value)
    win_std: float = 10.0               # Std dev of winning trade sizes
    loss_std: float = 15.0              # Std dev of losing trade sizes
    max_loss_per_trade: float = 10000.0  # Maximum single trade loss (stop loss cap)
    max_daily_trades: int = 60          # Hard cap on trades per day
    base_contracts: int = 20            # Base number of micros the strategy uses

    def __post_init__(self):
        """Validate parameters."""
        if not 0 < self.win_rate < 1:
            raise ValueError("win_rate must be between 0 and 1")
        if self.trades_per_day <= 0:
            raise ValueError("trades_per_day must be positive")
        if self.avg_win <= 0:
            raise ValueError("avg_win must be positive")
        if self.avg_loss <= 0:
            raise ValueError("avg_loss must be positive")

    @property
    def per_trade_ev(self) -> float:
        """Expected value per trade."""
        return self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss

    @property
    def daily_ev(self) -> float:
        """Expected daily P&L."""
        return self.per_trade_ev * self.trades_per_day

    @property
    def per_trade_variance(self) -> float:
        """Variance of per-trade P&L."""
        # E[X^2] - E[X]^2
        # For a mixture: E[X^2] = p * E[W^2] + (1-p) * E[L^2]
        p = self.win_rate
        e_w_sq = self.avg_win**2 + self.win_std**2
        e_l_sq = self.avg_loss**2 + self.loss_std**2
        e_x_sq = p * e_w_sq + (1 - p) * e_l_sq
        return e_x_sq - self.per_trade_ev**2

    @property
    def per_trade_std(self) -> float:
        """Standard deviation of per-trade P&L."""
        return np.sqrt(self.per_trade_variance)

    @property
    def daily_std(self) -> float:
        """Standard deviation of daily P&L (assuming independent trades)."""
        return self.per_trade_std * np.sqrt(self.trades_per_day)

    def summary(self) -> str:
        """Return a summary string of key statistics."""
        return (
            f"Strategy Parameters:\n"
            f"  Trades/Day:      {self.trades_per_day}\n"
            f"  Win Rate:        {self.win_rate:.1%}\n"
            f"  Avg Win:         ${self.avg_win:.2f}\n"
            f"  Avg Loss:        ${self.avg_loss:.2f}\n"
            f"  Per-Trade EV:    ${self.per_trade_ev:.2f}\n"
            f"  Daily EV:        ${self.daily_ev:.2f}\n"
            f"  Daily Std Dev:   ${self.daily_std:.2f}"
        )


class TradeGenerator:
    """
    Generates synthetic trade P&L sequences.

    Uses numpy for vectorized generation to support batch simulations.
    """

    def __init__(self, params: StrategyParams, rng: Optional[np.random.Generator] = None):
        """
        Initialize the trade generator.

        Args:
            params: Strategy parameters
            rng: Random number generator (creates new one if None)
        """
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_trades(self, n_trades: int, contract_scale: float = 1.0) -> np.ndarray:
        """
        Generate a sequence of trade P&L values.

        Args:
            n_trades: Number of trades to generate
            contract_scale: Scale factor for position sizing (0-1)
                           Used when XFA scaling limits contracts

        Returns:
            Array of trade P&L values
        """
        # Determine wins vs losses
        is_win = self.rng.random(n_trades) < self.params.win_rate

        # Generate trade sizes
        trades = np.zeros(n_trades)

        # Winning trades: positive values from |Normal(avg_win, win_std)|
        n_wins = is_win.sum()
        if n_wins > 0:
            wins = np.abs(self.rng.normal(
                self.params.avg_win,
                self.params.win_std,
                n_wins
            ))
            trades[is_win] = wins

        # Losing trades: negative values from -|Normal(avg_loss, loss_std)|
        n_losses = n_trades - n_wins
        if n_losses > 0:
            losses = -np.abs(self.rng.normal(
                self.params.avg_loss,
                self.params.loss_std,
                n_losses
            ))
            # Clamp to max loss per trade
            losses = np.maximum(losses, -self.params.max_loss_per_trade)
            trades[~is_win] = losses

        # Apply contract scaling
        if contract_scale != 1.0:
            trades *= contract_scale

        return trades

    def generate_day(self, contract_scale: float = 1.0,
                     daily_loss_limit: Optional[float] = None) -> Tuple[float, np.ndarray]:
        """
        Generate trades for a single trading day.

        Args:
            contract_scale: Scale factor for position sizing
            daily_loss_limit: Stop trading if daily loss exceeds this (positive value)

        Returns:
            Tuple of (daily_pnl, trades_array)
        """
        n_trades = min(self.params.trades_per_day, self.params.max_daily_trades)
        trades = self.generate_trades(n_trades, contract_scale)

        if daily_loss_limit is not None:
            # Simulate intraday stop: stop adding trades once limit hit
            cumsum = np.cumsum(trades)
            # Find first index where we breach the limit
            breach_idx = np.where(cumsum <= -daily_loss_limit)[0]
            if len(breach_idx) > 0:
                stop_idx = breach_idx[0] + 1  # Include the trade that caused breach
                trades = trades[:stop_idx]
                # Cap the daily loss at the limit
                daily_pnl = max(trades.sum(), -daily_loss_limit)
                return daily_pnl, trades

        return trades.sum(), trades

    def generate_days_batch(self, n_days: int, n_sims: int,
                            contract_scale: float = 1.0) -> np.ndarray:
        """
        Generate daily P&L for multiple days across multiple simulations.

        Optimized for batch simulation - generates all random numbers upfront.

        Args:
            n_days: Number of trading days
            n_sims: Number of parallel simulations
            contract_scale: Scale factor for position sizing

        Returns:
            Array of shape (n_sims, n_days) with daily P&L values
        """
        n_trades = min(self.params.trades_per_day, self.params.max_daily_trades)
        total_trades = n_days * n_sims * n_trades

        # Generate all wins/losses at once
        is_win = self.rng.random(total_trades) < self.params.win_rate

        # Generate all trade sizes
        trades = np.zeros(total_trades)

        # Wins
        n_wins = is_win.sum()
        if n_wins > 0:
            trades[is_win] = np.abs(self.rng.normal(
                self.params.avg_win, self.params.win_std, n_wins
            ))

        # Losses
        n_losses = total_trades - n_wins
        if n_losses > 0:
            losses = -np.abs(self.rng.normal(
                self.params.avg_loss, self.params.loss_std, n_losses
            ))
            losses = np.maximum(losses, -self.params.max_loss_per_trade)
            trades[~is_win] = losses

        # Apply scaling
        trades *= contract_scale

        # Reshape and sum to get daily P&L
        # Shape: (n_sims, n_days, n_trades) -> sum over trades -> (n_sims, n_days)
        trades = trades.reshape(n_sims, n_days, n_trades)
        daily_pnl = trades.sum(axis=2)

        return daily_pnl


class BootstrapTradeGenerator:
    """
    Generates trades by resampling from historical trade data.

    Use this when you have real backtest results and want to preserve
    the empirical distribution including any fat tails or skewness.
    """

    def __init__(self, historical_trades: np.ndarray,
                 trades_per_day: int = 40,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize bootstrap generator.

        Args:
            historical_trades: Array of historical trade P&L values
            trades_per_day: Number of trades to sample per day
            rng: Random number generator
        """
        self.historical_trades = np.asarray(historical_trades)
        self.trades_per_day = trades_per_day
        self.rng = rng if rng is not None else np.random.default_rng()

        # Compute statistics from historical data
        self._compute_stats()

    def _compute_stats(self):
        """Compute statistics from historical trades."""
        trades = self.historical_trades
        wins = trades[trades > 0]
        losses = trades[trades < 0]

        self.win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        self.avg_win = wins.mean() if len(wins) > 0 else 0
        self.avg_loss = -losses.mean() if len(losses) > 0 else 0
        self.per_trade_ev = trades.mean() if len(trades) > 0 else 0
        self.per_trade_std = trades.std() if len(trades) > 0 else 0

    def generate_trades(self, n_trades: int, contract_scale: float = 1.0) -> np.ndarray:
        """
        Generate trades by resampling with replacement.

        Args:
            n_trades: Number of trades to generate
            contract_scale: Scale factor for position sizing

        Returns:
            Array of trade P&L values
        """
        indices = self.rng.integers(0, len(self.historical_trades), n_trades)
        trades = self.historical_trades[indices] * contract_scale
        return trades

    def generate_day(self, contract_scale: float = 1.0) -> Tuple[float, np.ndarray]:
        """Generate trades for a single day."""
        trades = self.generate_trades(self.trades_per_day, contract_scale)
        return trades.sum(), trades

    def generate_days_batch(self, n_days: int, n_sims: int,
                            contract_scale: float = 1.0) -> np.ndarray:
        """Generate daily P&L for batch simulation."""
        total_trades = n_days * n_sims * self.trades_per_day
        indices = self.rng.integers(0, len(self.historical_trades), total_trades)
        trades = self.historical_trades[indices] * contract_scale

        # Reshape and sum
        trades = trades.reshape(n_sims, n_days, self.trades_per_day)
        daily_pnl = trades.sum(axis=2)

        return daily_pnl

    def to_strategy_params(self) -> StrategyParams:
        """Convert bootstrap statistics to StrategyParams for compatibility."""
        wins = self.historical_trades[self.historical_trades > 0]
        losses = self.historical_trades[self.historical_trades < 0]

        return StrategyParams(
            trades_per_day=self.trades_per_day,
            win_rate=self.win_rate,
            avg_win=self.avg_win if self.avg_win > 0 else 1.0,
            avg_loss=self.avg_loss if self.avg_loss > 0 else 1.0,
            win_std=wins.std() if len(wins) > 1 else 1.0,
            loss_std=(-losses).std() if len(losses) > 1 else 1.0,
        )

    def summary(self) -> str:
        """Return summary of historical trade statistics."""
        return (
            f"Bootstrap Trade Generator:\n"
            f"  Historical Trades: {len(self.historical_trades)}\n"
            f"  Trades/Day:        {self.trades_per_day}\n"
            f"  Win Rate:          {self.win_rate:.1%}\n"
            f"  Avg Win:           ${self.avg_win:.2f}\n"
            f"  Avg Loss:          ${self.avg_loss:.2f}\n"
            f"  Per-Trade EV:      ${self.per_trade_ev:.2f}\n"
            f"  Per-Trade Std:     ${self.per_trade_std:.2f}"
        )


def create_generator(params: StrategyParams,
                     historical_trades: Optional[np.ndarray] = None,
                     rng: Optional[np.random.Generator] = None):
    """
    Factory function to create appropriate trade generator.

    Args:
        params: Strategy parameters (used if no historical data)
        historical_trades: Optional historical trade data for bootstrap
        rng: Random number generator

    Returns:
        TradeGenerator or BootstrapTradeGenerator
    """
    if historical_trades is not None and len(historical_trades) > 0:
        return BootstrapTradeGenerator(
            historical_trades,
            trades_per_day=params.trades_per_day,
            rng=rng
        )
    return TradeGenerator(params, rng=rng)
