"""
Trading Combine Simulator

Simulates the TopStep Trading Combine phase with accurate MLL trailing
and consistency rule enforcement.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
import numpy as np

from .config import (
    CombineRules,
    calculate_combine_mll,
    check_consistency_rule,
    DEFAULT_MAX_TRADING_DAYS
)
from .trade_generator import StrategyParams, TradeGenerator, BootstrapTradeGenerator


@dataclass
class CombineResult:
    """Result of a single Trading Combine attempt."""
    passed: bool
    days_to_complete: int
    final_balance: float
    peak_balance: float
    daily_pnls: List[float]
    reason_failed: Optional[str] = None  # 'mll_breach' or None
    consistency_satisfied: bool = True

    @property
    def total_profit(self) -> float:
        """Total profit (or loss) from the combine."""
        return sum(self.daily_pnls)

    @property
    def best_day_profit(self) -> float:
        """Best single day profit."""
        profitable_days = [p for p in self.daily_pnls if p > 0]
        return max(profitable_days) if profitable_days else 0.0


@dataclass
class CombineBatchResults:
    """Aggregate results from batch combine simulations."""
    n_sims: int
    pass_rate: float
    avg_days_to_pass: float
    avg_days_to_fail: float
    std_days_to_pass: float
    std_days_to_fail: float
    avg_final_balance_pass: float
    avg_final_balance_fail: float
    days_distribution_pass: np.ndarray
    days_distribution_fail: np.ndarray
    final_balances: np.ndarray
    passed_mask: np.ndarray

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Combine Batch Results ({self.n_sims:,} simulations):\n"
            f"  Pass Rate:           {self.pass_rate:.1%}\n"
            f"  Avg Days (pass):     {self.avg_days_to_pass:.1f} ± {self.std_days_to_pass:.1f}\n"
            f"  Avg Days (fail):     {self.avg_days_to_fail:.1f} ± {self.std_days_to_fail:.1f}\n"
            f"  Avg Balance (pass):  ${self.avg_final_balance_pass:,.2f}\n"
            f"  Avg Balance (fail):  ${self.avg_final_balance_fail:,.2f}"
        )


def simulate_combine(
    generator: Union[TradeGenerator, BootstrapTradeGenerator],
    rules: Optional[CombineRules] = None,
    max_days: int = DEFAULT_MAX_TRADING_DAYS
) -> CombineResult:
    """
    Simulate a single Trading Combine attempt.

    The simulation runs day-by-day, checking MLL and pass conditions
    at the end of each day.

    Args:
        generator: Trade generator to use
        rules: Combine rules (uses defaults if None)
        max_days: Maximum days before timeout

    Returns:
        CombineResult with outcome details
    """
    if rules is None:
        rules = CombineRules()

    # Initialize state
    balance = rules.starting_balance
    high_watermark = rules.starting_balance
    daily_pnls: List[float] = []

    for day in range(max_days):
        # MLL level is fixed for the day, based on previous EOD high watermark
        # (value is set at EOD, but checked in real-time during trading)
        mll_level = calculate_combine_mll(high_watermark, rules)

        # Generate trades for the day
        daily_pnl, trades = generator.generate_day(
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
            return CombineResult(
                passed=False,
                days_to_complete=len(daily_pnls),
                final_balance=balance,
                peak_balance=high_watermark,
                daily_pnls=daily_pnls,
                reason_failed='mll_breach',
                consistency_satisfied=False
            )

        # No intraday breach — update balance with full day's P&L
        balance += daily_pnl
        daily_pnls.append(daily_pnl)

        # Update high watermark (EOD)
        high_watermark = max(high_watermark, balance)

        # Check pass conditions
        if balance >= rules.pass_balance:
            # Check minimum trading days
            if len(daily_pnls) >= rules.min_trading_days:
                # Check consistency rule
                if check_consistency_rule(daily_pnls, rules):
                    return CombineResult(
                        passed=True,
                        days_to_complete=len(daily_pnls),
                        final_balance=balance,
                        peak_balance=high_watermark,
                        daily_pnls=daily_pnls,
                        reason_failed=None,
                        consistency_satisfied=True
                    )
                # Consistency not satisfied - must continue trading to dilute best day

    # Timeout - didn't pass or fail definitively
    return CombineResult(
        passed=False,
        days_to_complete=max_days,
        final_balance=balance,
        peak_balance=high_watermark,
        daily_pnls=daily_pnls,
        reason_failed='timeout',
        consistency_satisfied=check_consistency_rule(daily_pnls, rules)
    )


def simulate_combine_batch(
    params: StrategyParams,
    rules: Optional[CombineRules] = None,
    n_sims: int = 100_000,
    max_days: int = DEFAULT_MAX_TRADING_DAYS,
    rng: Optional[np.random.Generator] = None
) -> CombineBatchResults:
    """
    Simulate multiple combine attempts in parallel using vectorization.

    This is the performance-optimized path for Monte Carlo simulation.

    Args:
        params: Strategy parameters
        rules: Combine rules
        n_sims: Number of simulations to run
        max_days: Maximum days per simulation
        rng: Random number generator

    Returns:
        CombineBatchResults with aggregate statistics
    """
    if rules is None:
        rules = CombineRules()
    if rng is None:
        rng = np.random.default_rng()

    generator = TradeGenerator(params, rng=rng)

    # Pre-generate all daily P&L values
    # Shape: (n_sims, max_days)
    daily_pnl = generator.generate_days_batch(max_days, n_sims)

    # Compute cumulative balance for each simulation
    # Starting balance added to cumulative sum of daily P&L
    cumulative_pnl = np.cumsum(daily_pnl, axis=1)
    balances = rules.starting_balance + cumulative_pnl

    # Track high watermark (running max along days axis)
    high_watermarks = np.maximum.accumulate(balances, axis=1)

    # Calculate MLL level for each day
    # MLL = min(starting_balance, max(floor, high_watermark - max_loss))
    floor = rules.starting_balance - rules.max_loss_limit
    trailing_mll = high_watermarks - rules.max_loss_limit
    mll_levels = np.minimum(rules.starting_balance, np.maximum(floor, trailing_mll))

    # Check MLL breach: balance <= mll_level
    mll_breach = balances <= mll_levels

    # Check profit target reached
    target_reached = balances >= rules.pass_balance

    # Check minimum days requirement (implicit - need at least min_trading_days)
    # Create a mask for days that meet minimum requirement
    day_indices = np.arange(max_days)
    min_days_met = day_indices >= (rules.min_trading_days - 1)  # 0-indexed

    # Consistency rule check is complex - need to track best day and total profit
    # For each simulation, compute running best day and running total profit
    # Then check if best_day / total_profit <= 0.50

    # Running max of positive daily P&L (best profitable day so far)
    positive_pnl = np.maximum(daily_pnl, 0)
    running_best_day = np.maximum.accumulate(positive_pnl, axis=1)

    # Running total profit
    running_total = cumulative_pnl

    # Consistency check: best_day / total_profit <= 0.50
    # Handle division carefully - need positive total profit
    with np.errstate(divide='ignore', invalid='ignore'):
        consistency_ratio = running_best_day / running_total
        consistency_ratio = np.where(running_total > 0, consistency_ratio, np.inf)
    consistency_met = consistency_ratio <= rules.consistency_rule

    # Combine conditions for passing
    # Pass on first day where: target reached AND min days met AND consistency met AND no prior MLL breach
    pass_conditions = target_reached & min_days_met[np.newaxis, :] & consistency_met

    # Find first MLL breach day for each simulation
    # Use argmax on breach mask - returns first True index (or 0 if no breach)
    mll_breach_any = mll_breach.any(axis=1)
    first_mll_breach = np.where(
        mll_breach_any,
        np.argmax(mll_breach, axis=1),
        max_days  # No breach
    )

    # Find first pass day for each simulation
    pass_any = pass_conditions.any(axis=1)
    first_pass_day = np.where(
        pass_any,
        np.argmax(pass_conditions, axis=1),
        max_days  # No pass
    )

    # Determine outcome: passed if pass day comes before MLL breach
    passed = pass_any & (first_pass_day < first_mll_breach)

    # Days to complete
    days_to_complete = np.where(passed, first_pass_day + 1, first_mll_breach + 1)
    days_to_complete = np.minimum(days_to_complete, max_days)

    # Final balances at completion day
    final_day_indices = np.minimum(days_to_complete - 1, max_days - 1)
    final_balances = balances[np.arange(n_sims), final_day_indices]

    # Compute statistics
    pass_rate = passed.mean()

    passed_mask = passed
    failed_mask = ~passed

    # Days statistics
    if passed_mask.sum() > 0:
        days_pass = days_to_complete[passed_mask]
        avg_days_to_pass = days_pass.mean()
        std_days_to_pass = days_pass.std()
        avg_final_balance_pass = final_balances[passed_mask].mean()
    else:
        avg_days_to_pass = 0.0
        std_days_to_pass = 0.0
        avg_final_balance_pass = 0.0
        days_pass = np.array([])

    if failed_mask.sum() > 0:
        days_fail = days_to_complete[failed_mask]
        avg_days_to_fail = days_fail.mean()
        std_days_to_fail = days_fail.std()
        avg_final_balance_fail = final_balances[failed_mask].mean()
    else:
        avg_days_to_fail = 0.0
        std_days_to_fail = 0.0
        avg_final_balance_fail = 0.0
        days_fail = np.array([])

    return CombineBatchResults(
        n_sims=n_sims,
        pass_rate=pass_rate,
        avg_days_to_pass=avg_days_to_pass,
        avg_days_to_fail=avg_days_to_fail,
        std_days_to_pass=std_days_to_pass,
        std_days_to_fail=std_days_to_fail,
        avg_final_balance_pass=avg_final_balance_pass,
        avg_final_balance_fail=avg_final_balance_fail,
        days_distribution_pass=days_pass,
        days_distribution_fail=days_fail,
        final_balances=final_balances,
        passed_mask=passed_mask
    )


def estimate_combine_pass_probability(
    params: StrategyParams,
    rules: Optional[CombineRules] = None,
    n_sims: int = 50_000,
    rng: Optional[np.random.Generator] = None
) -> Tuple[float, float, float]:
    """
    Quick estimate of combine pass probability and average cost.

    Args:
        params: Strategy parameters
        rules: Combine rules
        n_sims: Number of simulations
        rng: Random number generator

    Returns:
        Tuple of (pass_rate, avg_days_to_pass, avg_days_to_fail)
    """
    results = simulate_combine_batch(params, rules, n_sims, rng=rng)
    return results.pass_rate, results.avg_days_to_pass, results.avg_days_to_fail


def calculate_combine_ev(
    params: StrategyParams,
    rules: Optional[CombineRules] = None,
    cost_per_attempt: float = 49.0,
    max_attempts: int = 20,
    n_sims: int = 50_000,
    rng: Optional[np.random.Generator] = None
) -> Tuple[float, float, float]:
    """
    Calculate expected cost and attempts to pass the combine.

    Models the combine as a geometric distribution of attempts.

    Args:
        params: Strategy parameters
        rules: Combine rules
        cost_per_attempt: Cost per combine attempt (reset fee)
        max_attempts: Maximum attempts before giving up
        n_sims: Simulations for pass rate estimation
        rng: Random number generator

    Returns:
        Tuple of (expected_attempts, expected_cost, pass_probability_total)
    """
    results = simulate_combine_batch(params, rules, n_sims, rng=rng)
    p = results.pass_rate

    if p <= 0:
        return max_attempts, max_attempts * cost_per_attempt, 0.0

    # Expected attempts follows geometric distribution
    # E[attempts] = 1/p for first success
    # But we cap at max_attempts
    expected_attempts = 0.0
    prob_still_trying = 1.0
    total_pass_prob = 0.0

    for attempt in range(1, max_attempts + 1):
        # Probability of passing on this attempt
        prob_pass_this = prob_still_trying * p
        total_pass_prob += prob_pass_this

        # Add expected attempts
        expected_attempts += attempt * prob_pass_this

        # Update probability of still trying
        prob_still_trying *= (1 - p)

    # Add contribution from those who never pass
    expected_attempts += max_attempts * prob_still_trying

    expected_cost = expected_attempts * cost_per_attempt

    return expected_attempts, expected_cost, total_pass_prob
