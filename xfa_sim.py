"""
Express Funded Account (XFA) Simulator

Simulates the XFA trading phase with scaling plan, benchmark days,
payout mechanics, and MLL management.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Any
from enum import Enum
import numpy as np

from .config import (
    XFARules,
    calculate_xfa_mll,
    calculate_payout_amount,
    calculate_net_payout,
    DEFAULT_MAX_TRADING_DAYS
)
from .trade_generator import StrategyParams, TradeGenerator, BootstrapTradeGenerator


class PayoutStrategyType(Enum):
    """Types of payout strategies."""
    IMMEDIATE = "immediate"           # Take payout as soon as eligible
    TARGET_BALANCE = "target_balance" # Wait for specific balance before payout
    MAX_PAYOUT = "max_payout"         # Wait for max payout ($10K balance)
    ONE_AND_DONE = "one_and_done"     # Take one payout and stop/accept fragility


@dataclass
class PayoutStrategy:
    """Configuration for payout decision strategy."""
    strategy_type: PayoutStrategyType = PayoutStrategyType.IMMEDIATE
    target_balance: float = 5_000.0   # For TARGET_BALANCE strategy
    stop_after_payout: bool = False   # For ONE_AND_DONE - stop trading after payout

    @classmethod
    def immediate(cls) -> 'PayoutStrategy':
        """Take payout immediately when eligible."""
        return cls(strategy_type=PayoutStrategyType.IMMEDIATE)

    @classmethod
    def target_balance(cls, target: float) -> 'PayoutStrategy':
        """Wait until balance reaches target before payout."""
        return cls(strategy_type=PayoutStrategyType.TARGET_BALANCE, target_balance=target)

    @classmethod
    def max_payout(cls) -> 'PayoutStrategy':
        """Wait until balance is high enough for max payout ($5K)."""
        return cls(
            strategy_type=PayoutStrategyType.MAX_PAYOUT,
            target_balance=10_000.0  # 50% of $10K = $5K max
        )

    @classmethod
    def one_and_done(cls, stop: bool = True) -> 'PayoutStrategy':
        """Take one payout then stop or continue with fragile account."""
        return cls(
            strategy_type=PayoutStrategyType.ONE_AND_DONE,
            stop_after_payout=stop
        )

    def should_take_payout(self, balance: float, benchmark_days: int,
                           rules: XFARules, num_payouts_taken: int,
                           balance_after_last_payout: float = 0.0) -> bool:
        """
        Determine if payout should be taken given current state.

        Args:
            balance: Current account balance
            benchmark_days: Benchmark days accumulated since last payout
            rules: XFA rules
            num_payouts_taken: Number of payouts already taken
            balance_after_last_payout: Balance immediately after last payout
                                       (for profit-since-last-payout rule)

        Returns:
            True if payout should be taken
        """
        # Basic eligibility
        if benchmark_days < rules.benchmark_days_required:
            return False
        if balance <= 0:
            return False

        # CRITICAL: Profit since last payout rule
        # After first payout, must be above the balance after last payout
        if num_payouts_taken > 0 and balance <= balance_after_last_payout:
            return False

        # Strategy-specific logic
        if self.strategy_type == PayoutStrategyType.IMMEDIATE:
            return True

        elif self.strategy_type == PayoutStrategyType.TARGET_BALANCE:
            return balance >= self.target_balance

        elif self.strategy_type == PayoutStrategyType.MAX_PAYOUT:
            # Wait for max payout amount
            potential_payout = calculate_payout_amount(balance, rules)
            return potential_payout >= rules.max_payout_amount

        elif self.strategy_type == PayoutStrategyType.ONE_AND_DONE:
            return num_payouts_taken == 0  # Only take first payout

        return True


@dataclass
class XFAResult:
    """Result of a single XFA lifecycle."""
    total_payouts: float              # Sum of all net payouts received
    num_payouts: int                  # Number of payouts taken
    days_active: int                  # Trading days before closure
    final_balance: float              # Balance at closure
    peak_balance: float               # Highest balance achieved
    payout_history: List[float]       # List of net payout amounts
    reason_closed: str                # 'mll_breach', 'timeout', 'stopped'
    benchmark_days_total: int         # Total benchmark days accumulated
    took_payout: bool                 # Whether any payout was taken (affects B2F)
    daily_pnls: List[float] = field(default_factory=list)

    @property
    def gross_payouts(self) -> float:
        """Estimated gross payout before split and fees."""
        # Reverse engineer from net payouts
        if not self.payout_history:
            return 0.0
        return sum(self.payout_history)  # Already net, just sum

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"XFA Result:\n"
            f"  Days Active:      {self.days_active}\n"
            f"  Total Payouts:    ${self.total_payouts:,.2f}\n"
            f"  Num Payouts:      {self.num_payouts}\n"
            f"  Peak Balance:     ${self.peak_balance:,.2f}\n"
            f"  Final Balance:    ${self.final_balance:,.2f}\n"
            f"  Reason Closed:    {self.reason_closed}\n"
            f"  B2F Eligible:     {not self.took_payout}"
        )


@dataclass
class XFABatchResults:
    """Aggregate results from batch XFA simulations."""
    n_sims: int
    avg_total_payouts: float
    std_total_payouts: float
    avg_num_payouts: float
    avg_days_active: float
    avg_peak_balance: float
    pct_with_payout: float           # % of sims that got at least one payout
    pct_b2f_eligible: float          # % that closed without taking payout
    payouts_distribution: np.ndarray
    days_distribution: np.ndarray

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"XFA Batch Results ({self.n_sims:,} simulations):\n"
            f"  Avg Payouts:      ${self.avg_total_payouts:,.2f} ± ${self.std_total_payouts:,.2f}\n"
            f"  Avg Num Payouts:  {self.avg_num_payouts:.2f}\n"
            f"  Avg Days Active:  {self.avg_days_active:.1f}\n"
            f"  Avg Peak Balance: ${self.avg_peak_balance:,.2f}\n"
            f"  % With Payout:    {self.pct_with_payout:.1%}\n"
            f"  % B2F Eligible:   {self.pct_b2f_eligible:.1%}"
        )


def get_contract_scale(balance: float, base_contracts: int, rules: XFARules) -> float:
    """
    Calculate contract scaling factor based on current balance.

    Args:
        balance: Current account balance
        base_contracts: Number of micros the strategy is designed for
        rules: XFA rules with scaling plan

    Returns:
        Scale factor (0-1) for trade P&L
    """
    max_mini, max_micro = rules.get_max_contracts(balance)
    allowed = max_micro
    return min(allowed / base_contracts, 1.0) if base_contracts > 0 else 1.0


def simulate_xfa(
    generator: Union[TradeGenerator, BootstrapTradeGenerator],
    rules: Optional[XFARules] = None,
    payout_strategy: Optional[PayoutStrategy] = None,
    base_contracts: int = 20,
    max_days: int = DEFAULT_MAX_TRADING_DAYS
) -> XFAResult:
    """
    Simulate a single XFA lifecycle from activation to closure.

    Args:
        generator: Trade generator to use
        rules: XFA rules (uses defaults if None)
        payout_strategy: Payout decision strategy (uses immediate if None)
        base_contracts: Number of micros the strategy is calibrated for
        max_days: Maximum trading days before timeout

    Returns:
        XFAResult with outcome details
    """
    if rules is None:
        rules = XFARules()
    if payout_strategy is None:
        payout_strategy = PayoutStrategy.immediate()

    # Initialize state
    balance = rules.starting_balance  # $0
    high_watermark = 0.0
    benchmark_days_since_payout = 0
    benchmark_days_total = 0
    total_payouts = 0.0
    payout_history: List[float] = []
    daily_pnls: List[float] = []
    took_payout = False
    peak_balance = 0.0
    balance_after_last_payout = 0.0  # Track for profit-since-last-payout rule
    mll_locked_at_zero = False  # After first payout, MLL permanently locks at $0

    for day in range(max_days):
        # Determine contract scaling based on current balance
        contract_scale = get_contract_scale(balance, base_contracts, rules)

        # MLL level is fixed for the day, based on previous EOD high watermark
        # After first payout, MLL is permanently locked at $0
        if mll_locked_at_zero:
            mll_level = 0.0
        else:
            mll_level = calculate_xfa_mll(high_watermark, rules)

        # Generate trades for the day
        daily_pnl, trades = generator.generate_day(
            contract_scale=contract_scale,
            daily_loss_limit=rules.daily_loss_limit
        )

        # Check intraday MLL breach (real-time, trade by trade)
        intraday_cumsum = np.cumsum(trades)
        intraday_balances = balance + intraday_cumsum
        breach_indices = np.where(intraday_balances <= mll_level)[0]

        if len(breach_indices) > 0:
            # MLL breached during the trading day — account closed
            breach_idx = breach_indices[0]
            actual_pnl = float(intraday_cumsum[breach_idx])
            balance += actual_pnl
            daily_pnls.append(actual_pnl)
            return XFAResult(
                total_payouts=total_payouts,
                num_payouts=len(payout_history),
                days_active=len(daily_pnls),
                final_balance=balance,
                peak_balance=peak_balance,
                payout_history=payout_history,
                reason_closed='mll_breach',
                benchmark_days_total=benchmark_days_total,
                took_payout=took_payout,
                daily_pnls=daily_pnls
            )

        # No intraday breach — update balance with full day's P&L
        balance += daily_pnl
        daily_pnls.append(daily_pnl)

        # Check for benchmark day ($150+ profit)
        if daily_pnl >= rules.benchmark_day_threshold:
            benchmark_days_since_payout += 1
            benchmark_days_total += 1

        # Update high watermark (EOD)
        high_watermark = max(high_watermark, balance)
        peak_balance = max(peak_balance, balance)

        # Check payout eligibility and strategy
        if payout_strategy.should_take_payout(
            balance, benchmark_days_since_payout, rules, len(payout_history),
            balance_after_last_payout
        ):
            # Calculate and process payout
            gross_payout = calculate_payout_amount(balance, rules)
            net_payout = calculate_net_payout(gross_payout, rules)

            # Update balance - deduct gross payout
            balance -= gross_payout

            # After first payout, MLL is permanently locked at $0
            mll_locked_at_zero = True
            high_watermark = balance

            # Track balance after payout for profit-since-last-payout rule
            balance_after_last_payout = balance

            # Record payout
            total_payouts += net_payout
            payout_history.append(net_payout)
            took_payout = True
            benchmark_days_since_payout = 0

            # Check if we should stop (one_and_done strategy)
            if (payout_strategy.strategy_type == PayoutStrategyType.ONE_AND_DONE
                and payout_strategy.stop_after_payout):
                return XFAResult(
                    total_payouts=total_payouts,
                    num_payouts=len(payout_history),
                    days_active=len(daily_pnls),
                    final_balance=balance,
                    peak_balance=peak_balance,
                    payout_history=payout_history,
                    reason_closed='stopped',
                    benchmark_days_total=benchmark_days_total,
                    took_payout=took_payout,
                    daily_pnls=daily_pnls
                )

    # Timeout
    return XFAResult(
        total_payouts=total_payouts,
        num_payouts=len(payout_history),
        days_active=max_days,
        final_balance=balance,
        peak_balance=peak_balance,
        payout_history=payout_history,
        reason_closed='timeout',
        benchmark_days_total=benchmark_days_total,
        took_payout=took_payout,
        daily_pnls=daily_pnls
    )


def simulate_xfa_batch(
    params: StrategyParams,
    rules: Optional[XFARules] = None,
    payout_strategy: Optional[PayoutStrategy] = None,
    n_sims: int = 100_000,
    max_days: int = DEFAULT_MAX_TRADING_DAYS,
    rng: Optional[np.random.Generator] = None
) -> XFABatchResults:
    """
    Simulate multiple XFA lifecycles.

    Note: Due to the complex state transitions (payouts, MLL resets),
    this uses a loop over simulations rather than full vectorization.
    Still reasonably fast due to vectorized daily trade generation.

    Args:
        params: Strategy parameters
        rules: XFA rules
        payout_strategy: Payout decision strategy
        n_sims: Number of simulations
        max_days: Maximum days per simulation
        rng: Random number generator

    Returns:
        XFABatchResults with aggregate statistics
    """
    if rules is None:
        rules = XFARules()
    if payout_strategy is None:
        payout_strategy = PayoutStrategy.immediate()
    if rng is None:
        rng = np.random.default_rng()

    # Results arrays
    total_payouts = np.zeros(n_sims)
    num_payouts = np.zeros(n_sims, dtype=int)
    days_active = np.zeros(n_sims, dtype=int)
    peak_balances = np.zeros(n_sims)
    took_payouts = np.zeros(n_sims, dtype=bool)

    for i in range(n_sims):
        generator = TradeGenerator(params, rng=rng)
        result = simulate_xfa(
            generator, rules, payout_strategy,
            base_contracts=params.base_contracts,
            max_days=max_days
        )

        total_payouts[i] = result.total_payouts
        num_payouts[i] = result.num_payouts
        days_active[i] = result.days_active
        peak_balances[i] = result.peak_balance
        took_payouts[i] = result.took_payout

    # Compute statistics
    return XFABatchResults(
        n_sims=n_sims,
        avg_total_payouts=total_payouts.mean(),
        std_total_payouts=total_payouts.std(),
        avg_num_payouts=num_payouts.mean(),
        avg_days_active=days_active.mean(),
        avg_peak_balance=peak_balances.mean(),
        pct_with_payout=(num_payouts > 0).mean(),
        pct_b2f_eligible=(~took_payouts).mean(),
        payouts_distribution=total_payouts,
        days_distribution=days_active
    )


def estimate_xfa_ev(
    params: StrategyParams,
    rules: Optional[XFARules] = None,
    payout_strategy: Optional[PayoutStrategy] = None,
    n_sims: int = 50_000,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Estimate expected value of a single XFA lifecycle.

    This is E[total_payouts] for a fresh XFA start.
    Used for B2F decision calculations.

    Args:
        params: Strategy parameters
        rules: XFA rules
        payout_strategy: Payout strategy
        n_sims: Number of simulations
        rng: Random number generator

    Returns:
        Expected total payouts from XFA
    """
    results = simulate_xfa_batch(
        params, rules, payout_strategy, n_sims, rng=rng
    )
    return results.avg_total_payouts


def should_use_b2f(
    xfa_ev: float,
    b2f_cost: float = 499.0,
    margin: float = 1.0
) -> bool:
    """
    Determine if Back2Funded is worth using.

    Args:
        xfa_ev: Expected payouts from fresh XFA
        b2f_cost: Cost of B2F
        margin: Required EV margin over cost (default 1.0 = break-even)

    Returns:
        True if B2F is +EV
    """
    return xfa_ev > b2f_cost * margin
