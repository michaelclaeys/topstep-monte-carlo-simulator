"""
Full Lifecycle Simulator

Simulates the complete TopStep account lifecycle:
Combine attempts → XFA trading → Payouts → Blow → B2F decision → repeat
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np

from .config import (
    TopStepRules,
    CostStructure,
    get_default_rules,
    DEFAULT_MAX_TRADING_DAYS,
    DEFAULT_MAX_COMBINE_ATTEMPTS
)
from .trade_generator import StrategyParams, TradeGenerator
from .combine_sim import simulate_combine, CombineResult
from .xfa_sim import (
    simulate_xfa, XFAResult, PayoutStrategy, PayoutStrategyType,
    estimate_xfa_ev, should_use_b2f
)


class EventType(Enum):
    """Types of lifecycle events."""
    COMBINE_START = "combine_start"
    COMBINE_PASS = "combine_pass"
    COMBINE_FAIL = "combine_fail"
    XFA_START = "xfa_start"
    XFA_PAYOUT = "xfa_payout"
    XFA_BLOW = "xfa_blow"
    B2F_USED = "b2f_used"
    LIFECYCLE_END = "lifecycle_end"


@dataclass
class LifecycleEvent:
    """A single event in the lifecycle timeline."""
    event_type: EventType
    day: int
    balance: float
    cost: float = 0.0
    revenue: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleResult:
    """Result of a complete account lifecycle."""
    total_revenue: float              # Sum of all payouts received
    total_cost: float                 # Sum of all fees paid
    net_profit: float                 # revenue - cost
    combine_attempts: int             # Number of combine attempts
    combine_passed: bool              # Whether combine was passed
    xfa_payouts: float                # Total payouts from XFA(s)
    xfa_b2f_uses: int                 # Number of B2F uses
    xfa_days_active: int              # Total days active in XFA
    total_days: int                   # Total calendar days in lifecycle
    timeline: List[LifecycleEvent] = field(default_factory=list)

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Lifecycle Result:\n"
            f"  Combine Attempts:  {self.combine_attempts}\n"
            f"  Combine Passed:    {self.combine_passed}\n"
            f"  XFA Days Active:   {self.xfa_days_active}\n"
            f"  XFA Payouts:       ${self.xfa_payouts:,.2f}\n"
            f"  B2F Uses:          {self.xfa_b2f_uses}\n"
            f"  Total Cost:        ${self.total_cost:,.2f}\n"
            f"  Total Revenue:     ${self.total_revenue:,.2f}\n"
            f"  -------------------------------\n"
            f"  NET PROFIT:        ${self.net_profit:,.2f}\n"
            f"  -------------------------------"
        )


@dataclass
class LifecycleBatchResults:
    """Aggregate results from batch lifecycle simulations."""
    n_sims: int
    mean_net_profit: float            # THE KEY METRIC (lifecycle EV)
    median_net_profit: float
    std_net_profit: float
    p_profitable: float               # % of lifecycles net positive
    mean_revenue: float
    mean_cost: float
    mean_combine_attempts: float
    combine_pass_rate: float
    mean_xfa_payouts: float
    mean_xfa_days: float
    mean_b2f_uses: float
    percentiles: Dict[int, float]     # 10, 25, 50, 75, 90
    distribution: np.ndarray          # Full distribution of net profits

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"===========================================================\n"
            f" LIFECYCLE BATCH RESULTS ({self.n_sims:,} simulations)\n"
            f"===========================================================\n"
            f"\n"
            f"Combine Phase:\n"
            f"  Pass Rate:         {self.combine_pass_rate:.1%}\n"
            f"  Avg Attempts:      {self.mean_combine_attempts:.2f}\n"
            f"\n"
            f"XFA Phase:\n"
            f"  Avg Payouts:       ${self.mean_xfa_payouts:,.2f}\n"
            f"  Avg Days Active:   {self.mean_xfa_days:.1f}\n"
            f"  Avg B2F Uses:      {self.mean_b2f_uses:.2f}\n"
            f"\n"
            f"Financials:\n"
            f"  Mean Revenue:      ${self.mean_revenue:,.2f}\n"
            f"  Mean Cost:         ${self.mean_cost:,.2f}\n"
            f"\n"
            f"Full Lifecycle:\n"
            f"  -------------------------------\n"
            f"  LIFECYCLE EV:      ${self.mean_net_profit:,.2f}\n"
            f"  -------------------------------\n"
            f"  Median Profit:     ${self.median_net_profit:,.2f}\n"
            f"  Std Dev:           ${self.std_net_profit:,.2f}\n"
            f"  P(Profitable):     {self.p_profitable:.1%}\n"
            f"\n"
            f"Percentiles:\n"
            f"  10th:              ${self.percentiles[10]:,.2f}\n"
            f"  25th:              ${self.percentiles[25]:,.2f}\n"
            f"  50th:              ${self.percentiles[50]:,.2f}\n"
            f"  75th:              ${self.percentiles[75]:,.2f}\n"
            f"  90th:              ${self.percentiles[90]:,.2f}\n"
            f"==========================================================="
        )


def simulate_lifecycle(
    params: StrategyParams,
    rules: Optional[TopStepRules] = None,
    payout_strategy: Optional[PayoutStrategy] = None,
    use_b2f: bool = True,
    b2f_ev_threshold: Optional[float] = None,
    max_combine_attempts: int = DEFAULT_MAX_COMBINE_ATTEMPTS,
    max_days: int = DEFAULT_MAX_TRADING_DAYS,
    rng: Optional[np.random.Generator] = None
) -> LifecycleResult:
    """
    Simulate complete lifecycle: Combine → XFA → Payouts → Blow → B2F → repeat.

    Args:
        params: Strategy parameters
        rules: TopStep rules (uses defaults if None)
        payout_strategy: Payout decision strategy
        use_b2f: Whether to use Back2Funded when available
        b2f_ev_threshold: If set, only use B2F if XFA EV > threshold
                         If None, always use B2F when eligible and use_b2f=True
        max_combine_attempts: Give up after this many failed combines
        max_days: Maximum days per combine/XFA simulation
        rng: Random number generator

    Returns:
        LifecycleResult with complete outcome
    """
    if rules is None:
        rules = get_default_rules()
    if payout_strategy is None:
        payout_strategy = PayoutStrategy.immediate()
    if rng is None:
        rng = np.random.default_rng()

    # Initialize tracking
    total_cost = 0.0
    total_revenue = 0.0
    timeline: List[LifecycleEvent] = []
    current_day = 0

    # =================================================================
    # Phase 1: Pass the Combine
    # =================================================================
    combine_attempts = 0
    passed = False

    while not passed and combine_attempts < max_combine_attempts:
        combine_attempts += 1

        # Pay for this attempt (simplified: each attempt costs reset/monthly fee)
        attempt_cost = rules.costs.reset_fee
        total_cost += attempt_cost

        timeline.append(LifecycleEvent(
            event_type=EventType.COMBINE_START,
            day=current_day,
            balance=rules.combine.starting_balance,
            cost=attempt_cost
        ))

        # Run combine simulation
        generator = TradeGenerator(params, rng=rng)
        combine_result = simulate_combine(generator, rules.combine, max_days)

        current_day += combine_result.days_to_complete

        if combine_result.passed:
            passed = True
            # Pay activation fee
            total_cost += rules.costs.activation_fee

            timeline.append(LifecycleEvent(
                event_type=EventType.COMBINE_PASS,
                day=current_day,
                balance=combine_result.final_balance,
                cost=rules.costs.activation_fee,
                details={'days': combine_result.days_to_complete}
            ))
        else:
            timeline.append(LifecycleEvent(
                event_type=EventType.COMBINE_FAIL,
                day=current_day,
                balance=combine_result.final_balance,
                details={
                    'reason': combine_result.reason_failed,
                    'days': combine_result.days_to_complete
                }
            ))

    if not passed:
        # Failed to pass combine after max attempts
        timeline.append(LifecycleEvent(
            event_type=EventType.LIFECYCLE_END,
            day=current_day,
            balance=0,
            details={'reason': 'max_combine_attempts'}
        ))

        return LifecycleResult(
            total_revenue=0.0,
            total_cost=total_cost,
            net_profit=-total_cost,
            combine_attempts=combine_attempts,
            combine_passed=False,
            xfa_payouts=0.0,
            xfa_b2f_uses=0,
            xfa_days_active=0,
            total_days=current_day,
            timeline=timeline
        )

    # =================================================================
    # Phase 2: XFA Trading
    # =================================================================
    b2f_uses = 0
    xfa_active = True
    total_xfa_payouts = 0.0
    total_xfa_days = 0
    any_payout_taken = False

    timeline.append(LifecycleEvent(
        event_type=EventType.XFA_START,
        day=current_day,
        balance=rules.xfa.starting_balance
    ))

    while xfa_active:
        # Run XFA simulation
        generator = TradeGenerator(params, rng=rng)
        xfa_result = simulate_xfa(
            generator, rules.xfa, payout_strategy,
            base_contracts=params.base_contracts,
            max_days=max_days
        )

        current_day += xfa_result.days_active
        total_xfa_days += xfa_result.days_active
        total_xfa_payouts += xfa_result.total_payouts
        total_revenue += xfa_result.total_payouts

        if xfa_result.took_payout:
            any_payout_taken = True

        # Log payouts
        for payout in xfa_result.payout_history:
            timeline.append(LifecycleEvent(
                event_type=EventType.XFA_PAYOUT,
                day=current_day,  # Approximate
                balance=xfa_result.final_balance,
                revenue=payout
            ))

        if xfa_result.reason_closed == 'mll_breach':
            timeline.append(LifecycleEvent(
                event_type=EventType.XFA_BLOW,
                day=current_day,
                balance=xfa_result.final_balance,
                details={'took_payout': xfa_result.took_payout}
            ))

            # B2F decision
            # B2F only available if NO payout was EVER taken in this lifecycle
            can_use_b2f = (
                use_b2f
                and b2f_uses < rules.costs.b2f_max_uses
                and not any_payout_taken  # Changed from xfa_result.took_payout
            )

            if can_use_b2f:
                # Check EV threshold if set
                should_b2f = True
                if b2f_ev_threshold is not None:
                    # We'd need to estimate XFA EV - for speed, use the threshold directly
                    should_b2f = b2f_ev_threshold > rules.costs.b2f_fee

                if should_b2f:
                    total_cost += rules.costs.b2f_fee
                    b2f_uses += 1

                    timeline.append(LifecycleEvent(
                        event_type=EventType.B2F_USED,
                        day=current_day,
                        balance=rules.xfa.starting_balance,
                        cost=rules.costs.b2f_fee
                    ))

                    timeline.append(LifecycleEvent(
                        event_type=EventType.XFA_START,
                        day=current_day,
                        balance=rules.xfa.starting_balance
                    ))
                    # Continue loop - will simulate another XFA
                else:
                    xfa_active = False
            else:
                xfa_active = False
        else:
            # Timeout or stopped
            xfa_active = False

    timeline.append(LifecycleEvent(
        event_type=EventType.LIFECYCLE_END,
        day=current_day,
        balance=0,
        details={'reason': 'xfa_closed'}
    ))

    net_profit = total_revenue - total_cost

    return LifecycleResult(
        total_revenue=total_revenue,
        total_cost=total_cost,
        net_profit=net_profit,
        combine_attempts=combine_attempts,
        combine_passed=True,
        xfa_payouts=total_xfa_payouts,
        xfa_b2f_uses=b2f_uses,
        xfa_days_active=total_xfa_days,
        total_days=current_day,
        timeline=timeline
    )


def simulate_lifecycle_batch(
    params: StrategyParams,
    rules: Optional[TopStepRules] = None,
    payout_strategy: Optional[PayoutStrategy] = None,
    use_b2f: bool = True,
    n_sims: int = 100_000,
    max_combine_attempts: int = DEFAULT_MAX_COMBINE_ATTEMPTS,
    max_days: int = DEFAULT_MAX_TRADING_DAYS,
    rng: Optional[np.random.Generator] = None,
    show_progress: bool = False
) -> LifecycleBatchResults:
    """
    Run multiple full lifecycle simulations and compute aggregate statistics.

    Args:
        params: Strategy parameters
        rules: TopStep rules
        payout_strategy: Payout strategy
        use_b2f: Whether to use B2F
        n_sims: Number of simulations
        max_combine_attempts: Max combine attempts per lifecycle
        max_days: Max days per simulation phase
        rng: Random number generator
        show_progress: Print progress updates

    Returns:
        LifecycleBatchResults with aggregate statistics
    """
    if rules is None:
        rules = get_default_rules()
    if payout_strategy is None:
        payout_strategy = PayoutStrategy.immediate()
    if rng is None:
        rng = np.random.default_rng()

    # Pre-compute XFA EV for B2F decisions if using B2F
    b2f_ev_threshold = None
    if use_b2f:
        # Estimate XFA EV with smaller sample for speed
        xfa_ev = estimate_xfa_ev(params, rules.xfa, payout_strategy, n_sims=5000, rng=rng)
        b2f_ev_threshold = xfa_ev

    # Results arrays
    net_profits = np.zeros(n_sims)
    revenues = np.zeros(n_sims)
    costs = np.zeros(n_sims)
    combine_attempts = np.zeros(n_sims)
    combine_passed = np.zeros(n_sims, dtype=bool)
    xfa_payouts = np.zeros(n_sims)
    xfa_days = np.zeros(n_sims)
    b2f_uses = np.zeros(n_sims)

    progress_interval = max(1, n_sims // 10)

    for i in range(n_sims):
        if show_progress and i % progress_interval == 0:
            print(f"  Progress: {i:,}/{n_sims:,} ({100*i/n_sims:.0f}%)")

        result = simulate_lifecycle(
            params=params,
            rules=rules,
            payout_strategy=payout_strategy,
            use_b2f=use_b2f,
            b2f_ev_threshold=b2f_ev_threshold,
            max_combine_attempts=max_combine_attempts,
            max_days=max_days,
            rng=rng
        )

        net_profits[i] = result.net_profit
        revenues[i] = result.total_revenue
        costs[i] = result.total_cost
        combine_attempts[i] = result.combine_attempts
        combine_passed[i] = result.combine_passed
        xfa_payouts[i] = result.xfa_payouts
        xfa_days[i] = result.xfa_days_active
        b2f_uses[i] = result.xfa_b2f_uses

    # Compute statistics
    percentiles = {
        p: float(np.percentile(net_profits, p))
        for p in [10, 25, 50, 75, 90]
    }

    return LifecycleBatchResults(
        n_sims=n_sims,
        mean_net_profit=float(net_profits.mean()),
        median_net_profit=float(np.median(net_profits)),
        std_net_profit=float(net_profits.std()),
        p_profitable=float((net_profits > 0).mean()),
        mean_revenue=float(revenues.mean()),
        mean_cost=float(costs.mean()),
        mean_combine_attempts=float(combine_attempts.mean()),
        combine_pass_rate=float(combine_passed.mean()),
        mean_xfa_payouts=float(xfa_payouts.mean()),
        mean_xfa_days=float(xfa_days.mean()),
        mean_b2f_uses=float(b2f_uses.mean()),
        percentiles=percentiles,
        distribution=net_profits
    )


def quick_lifecycle_ev(
    params: StrategyParams,
    rules: Optional[TopStepRules] = None,
    payout_strategy: Optional[PayoutStrategy] = None,
    use_b2f: bool = True,
    n_sims: int = 10_000,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Quick estimate of lifecycle EV for optimization.

    Uses fewer simulations for faster iteration during parameter sweeps.

    Args:
        params: Strategy parameters
        rules: TopStep rules
        payout_strategy: Payout strategy
        use_b2f: Whether to use B2F
        n_sims: Number of simulations
        rng: Random number generator

    Returns:
        Estimated lifecycle EV
    """
    results = simulate_lifecycle_batch(
        params=params,
        rules=rules,
        payout_strategy=payout_strategy,
        use_b2f=use_b2f,
        n_sims=n_sims,
        rng=rng
    )
    return results.mean_net_profit
