"""
TopStep Strategy Tester

Monte Carlo simulation framework for TopStep 50K prop firm lifecycle optimization.

This package models the complete lifecycle of a TopStep 50K prop firm account:
- Trading Combine evaluation
- Express Funded Account (XFA) trading
- Payout mechanics
- Back2Funded (B2F) decisions

The primary optimization target is lifecycle EV (expected value), not traditional
metrics like Sharpe ratio or win rate.
"""

from .config import (
    TopStepRules,
    CombineRules,
    XFARules,
    CostStructure,
    get_default_rules,
    calculate_combine_mll,
    calculate_xfa_mll,
    calculate_payout_amount,
    calculate_net_payout,
    check_consistency_rule
)

from .trade_generator import (
    StrategyParams,
    TradeGenerator,
    BootstrapTradeGenerator,
    create_generator
)

from .combine_sim import (
    CombineResult,
    CombineBatchResults,
    simulate_combine,
    simulate_combine_batch,
    estimate_combine_pass_probability,
    calculate_combine_ev
)

from .xfa_sim import (
    PayoutStrategy,
    PayoutStrategyType,
    XFAResult,
    XFABatchResults,
    simulate_xfa,
    simulate_xfa_batch,
    estimate_xfa_ev,
    should_use_b2f
)

from .lifecycle_sim import (
    LifecycleResult,
    LifecycleBatchResults,
    simulate_lifecycle,
    simulate_lifecycle_batch,
    quick_lifecycle_ev
)

from .optimizer import (
    OptimizationConfig,
    OptimizationResults,
    optimize_lifecycle_ev,
    get_default_param_grid,
    get_default_payout_strategies,
    quick_optimize,
    find_breakeven_params
)

from .backtest_loader import (
    BacktestStats,
    load_backtest_csv,
    compute_backtest_stats,
    backtest_to_strategy_params,
    create_bootstrap_generator,
    load_and_analyze,
    validate_backtest_data
)

__version__ = "1.0.0"

__all__ = [
    # Config
    'TopStepRules',
    'CombineRules',
    'XFARules',
    'CostStructure',
    'get_default_rules',
    'calculate_combine_mll',
    'calculate_xfa_mll',
    'calculate_payout_amount',
    'calculate_net_payout',
    'check_consistency_rule',

    # Trade Generator
    'StrategyParams',
    'TradeGenerator',
    'BootstrapTradeGenerator',
    'create_generator',

    # Combine Sim
    'CombineResult',
    'CombineBatchResults',
    'simulate_combine',
    'simulate_combine_batch',
    'estimate_combine_pass_probability',
    'calculate_combine_ev',

    # XFA Sim
    'PayoutStrategy',
    'PayoutStrategyType',
    'XFAResult',
    'XFABatchResults',
    'simulate_xfa',
    'simulate_xfa_batch',
    'estimate_xfa_ev',
    'should_use_b2f',

    # Lifecycle Sim
    'LifecycleResult',
    'LifecycleBatchResults',
    'simulate_lifecycle',
    'simulate_lifecycle_batch',
    'quick_lifecycle_ev',

    # Optimizer
    'OptimizationConfig',
    'OptimizationResults',
    'optimize_lifecycle_ev',
    'get_default_param_grid',
    'get_default_payout_strategies',
    'quick_optimize',
    'find_breakeven_params',

    # Backtest Loader
    'BacktestStats',
    'load_backtest_csv',
    'compute_backtest_stats',
    'backtest_to_strategy_params',
    'create_bootstrap_generator',
    'load_and_analyze',
    'validate_backtest_data',
]
