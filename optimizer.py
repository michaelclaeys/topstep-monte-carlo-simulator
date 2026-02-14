"""
Parameter Optimizer

Sweeps over strategy parameters and payout strategies to find
configurations that maximize lifecycle EV.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import pandas as pd

from .config import TopStepRules, get_default_rules
from .trade_generator import StrategyParams
from .xfa_sim import PayoutStrategy, PayoutStrategyType
from .lifecycle_sim import (
    simulate_lifecycle_batch,
    quick_lifecycle_ev,
    LifecycleBatchResults
)


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    param_grid: Dict[str, List[Any]]
    payout_strategies: List[PayoutStrategy]
    n_sims_per_config: int = 10_000
    n_sims_final: int = 50_000
    use_b2f: bool = True
    n_workers: int = None  # None = use all CPUs
    seed: int = None

    def __post_init__(self):
        if self.n_workers is None:
            self.n_workers = max(1, multiprocessing.cpu_count() - 1)


@dataclass
class OptimizationResult:
    """Result for a single parameter configuration."""
    params: StrategyParams
    payout_strategy: PayoutStrategy
    lifecycle_ev: float
    pass_rate: float
    mean_cost: float
    mean_revenue: float
    p_profitable: float
    percentile_10: float
    percentile_90: float
    per_trade_ev: float
    daily_ev: float


@dataclass
class OptimizationResults:
    """Complete results from parameter optimization."""
    best_params: StrategyParams
    best_payout_strategy: PayoutStrategy
    best_lifecycle_ev: float
    all_results: pd.DataFrame
    sensitivity: Dict[str, pd.DataFrame]
    config: OptimizationConfig

    def summary(self) -> str:
        """Return summary of optimization results."""
        return (
            f"═══════════════════════════════════════════════════════\n"
            f" OPTIMIZATION RESULTS\n"
            f"═══════════════════════════════════════════════════════\n"
            f"\n"
            f"Best Configuration:\n"
            f"  Trades/Day:      {self.best_params.trades_per_day}\n"
            f"  Win Rate:        {self.best_params.win_rate:.1%}\n"
            f"  Avg Win:         ${self.best_params.avg_win:.2f}\n"
            f"  Avg Loss:        ${self.best_params.avg_loss:.2f}\n"
            f"  Per-Trade EV:    ${self.best_params.per_trade_ev:.2f}\n"
            f"  Daily EV:        ${self.best_params.daily_ev:.2f}\n"
            f"\n"
            f"Payout Strategy:   {self.best_payout_strategy.strategy_type.value}\n"
            f"\n"
            f"  ═══════════════════════════════\n"
            f"  BEST LIFECYCLE EV: ${self.best_lifecycle_ev:,.2f}\n"
            f"  ═══════════════════════════════\n"
            f"\n"
            f"Configurations Tested: {len(self.all_results)}\n"
            f"═══════════════════════════════════════════════════════"
        )


def _evaluate_config(args: Tuple) -> OptimizationResult:
    """
    Evaluate a single parameter configuration.

    This function is designed to be called in a separate process.
    """
    params, payout_strategy, rules, n_sims, use_b2f, seed = args

    rng = np.random.default_rng(seed)

    results = simulate_lifecycle_batch(
        params=params,
        rules=rules,
        payout_strategy=payout_strategy,
        use_b2f=use_b2f,
        n_sims=n_sims,
        rng=rng
    )

    return OptimizationResult(
        params=params,
        payout_strategy=payout_strategy,
        lifecycle_ev=results.mean_net_profit,
        pass_rate=results.combine_pass_rate,
        mean_cost=results.mean_cost,
        mean_revenue=results.mean_revenue,
        p_profitable=results.p_profitable,
        percentile_10=results.percentiles[10],
        percentile_90=results.percentiles[90],
        per_trade_ev=params.per_trade_ev,
        daily_ev=params.daily_ev
    )


def generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def optimize_lifecycle_ev(
    config: OptimizationConfig,
    rules: Optional[TopStepRules] = None,
    show_progress: bool = True
) -> OptimizationResults:
    """
    Sweep over strategy parameters and payout strategies to find
    the configuration that maximizes lifecycle EV.

    Args:
        config: Optimization configuration
        rules: TopStep rules
        show_progress: Print progress updates

    Returns:
        OptimizationResults with best configuration and all results
    """
    if rules is None:
        rules = get_default_rules()

    # Generate all parameter combinations
    param_combos = generate_param_combinations(config.param_grid)

    # Generate all configurations (params × payout strategies)
    all_configs = []
    base_seed = config.seed if config.seed else 42

    for i, params_dict in enumerate(param_combos):
        params = StrategyParams(**params_dict)
        for j, payout_strategy in enumerate(config.payout_strategies):
            seed = base_seed + i * len(config.payout_strategies) + j
            all_configs.append((
                params, payout_strategy, rules,
                config.n_sims_per_config, config.use_b2f, seed
            ))

    total_configs = len(all_configs)
    if show_progress:
        print(f"Evaluating {total_configs} configurations...")

    # Evaluate all configurations
    results: List[OptimizationResult] = []

    # Use single-process for simplicity (multiprocessing has issues with dataclasses)
    for i, args in enumerate(all_configs):
        if show_progress and i % 10 == 0:
            print(f"  Progress: {i}/{total_configs} ({100*i/total_configs:.0f}%)")
        result = _evaluate_config(args)
        results.append(result)

    if show_progress:
        print(f"  Progress: {total_configs}/{total_configs} (100%)")

    # Convert to DataFrame
    df_data = []
    for r in results:
        df_data.append({
            'trades_per_day': r.params.trades_per_day,
            'win_rate': r.params.win_rate,
            'avg_win': r.params.avg_win,
            'avg_loss': r.params.avg_loss,
            'win_std': r.params.win_std,
            'loss_std': r.params.loss_std,
            'payout_strategy': r.payout_strategy.strategy_type.value,
            'payout_target': r.payout_strategy.target_balance,
            'lifecycle_ev': r.lifecycle_ev,
            'pass_rate': r.pass_rate,
            'mean_cost': r.mean_cost,
            'mean_revenue': r.mean_revenue,
            'p_profitable': r.p_profitable,
            'percentile_10': r.percentile_10,
            'percentile_90': r.percentile_90,
            'per_trade_ev': r.per_trade_ev,
            'daily_ev': r.daily_ev
        })

    df = pd.DataFrame(df_data)
    df = df.sort_values('lifecycle_ev', ascending=False)

    # Find best configuration
    best_idx = df['lifecycle_ev'].idxmax()
    best_result = results[best_idx]

    # Compute sensitivity analysis
    sensitivity = compute_sensitivity(df, config.param_grid)

    if show_progress:
        print(f"\nBest Lifecycle EV: ${best_result.lifecycle_ev:,.2f}")
        print(f"Best Config: win_rate={best_result.params.win_rate:.1%}, "
              f"avg_win=${best_result.params.avg_win:.0f}, "
              f"avg_loss=${best_result.params.avg_loss:.0f}")

    return OptimizationResults(
        best_params=best_result.params,
        best_payout_strategy=best_result.payout_strategy,
        best_lifecycle_ev=best_result.lifecycle_ev,
        all_results=df,
        sensitivity=sensitivity,
        config=config
    )


def compute_sensitivity(
    df: pd.DataFrame,
    param_grid: Dict[str, List[Any]]
) -> Dict[str, pd.DataFrame]:
    """
    Compute sensitivity analysis for each parameter.

    For each parameter, compute how lifecycle EV changes as that parameter
    varies while averaging over other parameter values.
    """
    sensitivity = {}

    for param in param_grid.keys():
        if param not in df.columns:
            continue

        # Group by this parameter and compute mean lifecycle EV
        grouped = df.groupby(param)['lifecycle_ev'].agg(['mean', 'std', 'min', 'max'])
        grouped = grouped.reset_index()
        grouped.columns = [param, 'mean_ev', 'std_ev', 'min_ev', 'max_ev']
        sensitivity[param] = grouped

    return sensitivity


def find_breakeven_params(
    base_params: StrategyParams,
    param_to_vary: str,
    param_range: List[Any],
    rules: Optional[TopStepRules] = None,
    payout_strategy: Optional[PayoutStrategy] = None,
    n_sims: int = 10_000,
    rng: Optional[np.random.Generator] = None
) -> Optional[Any]:
    """
    Find the parameter value where lifecycle EV crosses zero.

    Args:
        base_params: Base strategy parameters
        param_to_vary: Name of parameter to vary
        param_range: Range of values to test (should be sorted)
        rules: TopStep rules
        payout_strategy: Payout strategy
        n_sims: Simulations per test
        rng: Random number generator

    Returns:
        Parameter value at breakeven, or None if not found
    """
    if rules is None:
        rules = get_default_rules()
    if payout_strategy is None:
        payout_strategy = PayoutStrategy.immediate()
    if rng is None:
        rng = np.random.default_rng()

    evs = []
    for value in param_range:
        # Create params with this value
        params_dict = {
            'trades_per_day': base_params.trades_per_day,
            'win_rate': base_params.win_rate,
            'avg_win': base_params.avg_win,
            'avg_loss': base_params.avg_loss,
            'win_std': base_params.win_std,
            'loss_std': base_params.loss_std,
            'base_contracts': base_params.base_contracts
        }
        params_dict[param_to_vary] = value
        params = StrategyParams(**params_dict)

        ev = quick_lifecycle_ev(params, rules, payout_strategy, n_sims=n_sims, rng=rng)
        evs.append(ev)

    # Find zero crossing
    evs = np.array(evs)
    param_values = np.array(param_range)

    # Look for sign change
    for i in range(len(evs) - 1):
        if evs[i] * evs[i + 1] < 0:
            # Linear interpolation
            x1, x2 = param_values[i], param_values[i + 1]
            y1, y2 = evs[i], evs[i + 1]
            breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
            return breakeven

    return None


def get_default_param_grid() -> Dict[str, List[Any]]:
    """Get default parameter grid for optimization."""
    return {
        'trades_per_day': [20, 30, 40, 50],
        'win_rate': [0.55, 0.60, 0.65, 0.68, 0.70, 0.75],
        'avg_win': [20, 25, 30, 35, 40],
        'avg_loss': [40, 50, 55, 60, 70],
        'win_std': [10],
        'loss_std': [15],
    }


def get_default_payout_strategies() -> List[PayoutStrategy]:
    """Get default payout strategies to test."""
    return [
        PayoutStrategy.immediate(),
        PayoutStrategy.target_balance(3_000),
        PayoutStrategy.target_balance(5_000),
        PayoutStrategy.target_balance(8_000),
        PayoutStrategy.max_payout(),
    ]


def quick_optimize(
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_sims: int = 5_000,
    show_progress: bool = True
) -> OptimizationResults:
    """
    Quick optimization with reduced simulation count.

    Good for initial exploration before running full optimization.
    """
    if param_grid is None:
        param_grid = get_default_param_grid()

    config = OptimizationConfig(
        param_grid=param_grid,
        payout_strategies=get_default_payout_strategies(),
        n_sims_per_config=n_sims,
        use_b2f=True
    )

    return optimize_lifecycle_ev(config, show_progress=show_progress)
