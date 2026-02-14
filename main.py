"""
TopStep Strategy Tester - CLI Interface

Monte Carlo simulation framework for TopStep 50K prop firm lifecycle optimization.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import numpy as np

from .config import get_default_rules, TopStepRules
from .trade_generator import StrategyParams
from .xfa_sim import PayoutStrategy, PayoutStrategyType
from .lifecycle_sim import simulate_lifecycle_batch, LifecycleBatchResults
from .combine_sim import simulate_combine_batch
from .optimizer import (
    OptimizationConfig, optimize_lifecycle_ev,
    get_default_param_grid, get_default_payout_strategies, quick_optimize
)
from .backtest_loader import load_and_analyze, backtest_to_strategy_params
from .analytics import generate_report, plot_lifecycle_distribution, HAS_MATPLOTLIB


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='topstep_sim',
        description='TopStep 50K Prop Firm Lifecycle Simulator & Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default parameters
  python -m topstep_sim simulate

  # Run simulation with custom parameters
  python -m topstep_sim simulate --win-rate 0.68 --avg-win 30 --avg-loss 55

  # Run parameter optimization
  python -m topstep_sim optimize --n-sims 10000

  # Evaluate backtest data
  python -m topstep_sim backtest --input trades.csv

  # Generate analysis report
  python -m topstep_sim analyze --output results/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run lifecycle simulation')
    add_strategy_params(sim_parser)
    sim_parser.add_argument('--n-sims', type=int, default=100000,
                           help='Number of simulations (default: 100000)')
    sim_parser.add_argument('--payout-strategy', choices=['immediate', 'target', 'max'],
                           default='immediate', help='Payout strategy (default: immediate)')
    sim_parser.add_argument('--payout-target', type=float, default=5000,
                           help='Target balance for payout (default: 5000)')
    sim_parser.add_argument('--no-b2f', action='store_true',
                           help='Disable Back2Funded')
    sim_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    sim_parser.add_argument('--output', type=str, help='Output directory for results')

    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    opt_parser.add_argument('--n-sims', type=int, default=10000,
                           help='Simulations per configuration (default: 10000)')
    opt_parser.add_argument('--quick', action='store_true',
                           help='Quick optimization with fewer simulations')
    opt_parser.add_argument('--output', type=str, default='optimization_results',
                           help='Output directory (default: optimization_results)')
    opt_parser.add_argument('--seed', type=int, help='Random seed')

    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Evaluate backtest data')
    bt_parser.add_argument('--input', '-i', type=str, required=True,
                          help='Path to CSV file with backtest trades')
    bt_parser.add_argument('--date-column', type=str, default='date',
                          help='Name of date column (default: date)')
    bt_parser.add_argument('--pnl-column', type=str, default='trade_pnl',
                          help='Name of P&L column (default: trade_pnl)')
    bt_parser.add_argument('--n-sims', type=int, default=50000,
                          help='Number of simulations (default: 50000)')
    bt_parser.add_argument('--output', type=str, help='Output directory')
    bt_parser.add_argument('--seed', type=int, help='Random seed')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Generate analysis report')
    add_strategy_params(analyze_parser)
    analyze_parser.add_argument('--params-file', type=str,
                               help='JSON file with strategy parameters')
    analyze_parser.add_argument('--n-sims', type=int, default=100000,
                               help='Number of simulations (default: 100000)')
    analyze_parser.add_argument('--output', type=str, default='analysis',
                               help='Output directory (default: analysis)')
    analyze_parser.add_argument('--seed', type=int, help='Random seed')

    # Combine-only command (for quick combine testing)
    combine_parser = subparsers.add_parser('combine', help='Simulate combine phase only')
    add_strategy_params(combine_parser)
    combine_parser.add_argument('--n-sims', type=int, default=100000,
                               help='Number of simulations (default: 100000)')
    combine_parser.add_argument('--seed', type=int, help='Random seed')

    return parser


def add_strategy_params(parser: argparse.ArgumentParser) -> None:
    """Add strategy parameter arguments to parser."""
    parser.add_argument('--trades-per-day', type=int, default=40,
                       help='Trades per day (default: 40)')
    parser.add_argument('--win-rate', type=float, default=0.68,
                       help='Win rate (default: 0.68)')
    parser.add_argument('--avg-win', type=float, default=30.0,
                       help='Average win in dollars (default: 30)')
    parser.add_argument('--avg-loss', type=float, default=55.0,
                       help='Average loss in dollars (default: 55)')
    parser.add_argument('--win-std', type=float, default=10.0,
                       help='Std dev of wins (default: 10)')
    parser.add_argument('--loss-std', type=float, default=15.0,
                       help='Std dev of losses (default: 15)')


def get_strategy_params(args) -> StrategyParams:
    """Extract strategy parameters from parsed args."""
    return StrategyParams(
        trades_per_day=args.trades_per_day,
        win_rate=args.win_rate,
        avg_win=args.avg_win,
        avg_loss=args.avg_loss,
        win_std=args.win_std,
        loss_std=args.loss_std
    )


def get_payout_strategy(args) -> PayoutStrategy:
    """Get payout strategy from args."""
    if args.payout_strategy == 'immediate':
        return PayoutStrategy.immediate()
    elif args.payout_strategy == 'target':
        return PayoutStrategy.target_balance(args.payout_target)
    elif args.payout_strategy == 'max':
        return PayoutStrategy.max_payout()
    return PayoutStrategy.immediate()


def cmd_simulate(args) -> None:
    """Run lifecycle simulation."""
    params = get_strategy_params(args)
    payout_strategy = get_payout_strategy(args)
    rng = np.random.default_rng(args.seed) if args.seed else None

    print("=" * 60)
    print(" TOPSTEP 50K LIFECYCLE SIMULATION")
    print("=" * 60)
    print()
    print(params.summary())
    print()
    print(f"Payout Strategy: {payout_strategy.strategy_type.value}")
    print(f"Simulations: {args.n_sims:,}")
    print(f"Back2Funded: {'Disabled' if args.no_b2f else 'Enabled'}")
    print()
    print("Running simulation...")

    results = simulate_lifecycle_batch(
        params=params,
        payout_strategy=payout_strategy,
        use_b2f=not args.no_b2f,
        n_sims=args.n_sims,
        rng=rng,
        show_progress=True
    )

    print()
    print(results.summary())

    if args.output:
        generate_report(results, params, payout_strategy, args.output)


def cmd_optimize(args) -> None:
    """Run parameter optimization."""
    print("=" * 60)
    print(" TOPSTEP 50K PARAMETER OPTIMIZATION")
    print("=" * 60)
    print()

    if args.quick:
        print("Running quick optimization...")
        results = quick_optimize(n_sims=args.n_sims, show_progress=True)
    else:
        config = OptimizationConfig(
            param_grid=get_default_param_grid(),
            payout_strategies=get_default_payout_strategies(),
            n_sims_per_config=args.n_sims,
            seed=args.seed
        )

        print(f"Simulations per config: {args.n_sims:,}")
        print()
        results = optimize_lifecycle_ev(config, show_progress=True)

    print()
    print(results.summary())

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save DataFrame
    csv_path = output_path / "optimization_results.csv"
    results.all_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save best params
    best_params_path = output_path / "best_params.json"
    best_params_dict = {
        'trades_per_day': results.best_params.trades_per_day,
        'win_rate': results.best_params.win_rate,
        'avg_win': results.best_params.avg_win,
        'avg_loss': results.best_params.avg_loss,
        'win_std': results.best_params.win_std,
        'loss_std': results.best_params.loss_std,
        'payout_strategy': results.best_payout_strategy.strategy_type.value,
        'lifecycle_ev': results.best_lifecycle_ev
    }
    with open(best_params_path, 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Best params saved to: {best_params_path}")


def cmd_backtest(args) -> None:
    """Evaluate backtest data."""
    print("=" * 60)
    print(" TOPSTEP 50K BACKTEST EVALUATION")
    print("=" * 60)
    print()

    print(f"Loading: {args.input}")
    trades, stats, validation = load_and_analyze(
        args.input,
        date_column=args.date_column,
        pnl_column=args.pnl_column
    )

    print()
    print(stats.summary())

    if validation['warnings']:
        print("\nWarnings:")
        for w in validation['warnings']:
            print(f"  - {w}")

    if validation['issues']:
        print("\nIssues:")
        for i in validation['issues']:
            print(f"  - {i}")
        return

    # Convert to strategy params
    params = backtest_to_strategy_params(stats)

    print()
    print("Running lifecycle simulation with backtest parameters...")
    print()

    rng = np.random.default_rng(args.seed) if args.seed else None
    payout_strategy = PayoutStrategy.immediate()

    results = simulate_lifecycle_batch(
        params=params,
        payout_strategy=payout_strategy,
        n_sims=args.n_sims,
        rng=rng,
        show_progress=True
    )

    print()
    print(results.summary())

    if args.output:
        generate_report(results, params, payout_strategy, args.output)


def cmd_analyze(args) -> None:
    """Generate analysis report."""
    print("=" * 60)
    print(" TOPSTEP 50K ANALYSIS REPORT")
    print("=" * 60)
    print()

    # Load params from file or use args
    if args.params_file:
        with open(args.params_file) as f:
            params_dict = json.load(f)
        params = StrategyParams(
            trades_per_day=params_dict.get('trades_per_day', 40),
            win_rate=params_dict.get('win_rate', 0.68),
            avg_win=params_dict.get('avg_win', 30.0),
            avg_loss=params_dict.get('avg_loss', 55.0),
            win_std=params_dict.get('win_std', 10.0),
            loss_std=params_dict.get('loss_std', 15.0)
        )
        payout_type = params_dict.get('payout_strategy', 'immediate')
        if payout_type == 'immediate':
            payout_strategy = PayoutStrategy.immediate()
        elif payout_type == 'max_payout':
            payout_strategy = PayoutStrategy.max_payout()
        else:
            payout_strategy = PayoutStrategy.target_balance(
                params_dict.get('payout_target', 5000)
            )
    else:
        params = get_strategy_params(args)
        payout_strategy = PayoutStrategy.immediate()

    print(params.summary())
    print()

    rng = np.random.default_rng(args.seed) if args.seed else None

    print(f"Running {args.n_sims:,} simulations...")
    results = simulate_lifecycle_batch(
        params=params,
        payout_strategy=payout_strategy,
        n_sims=args.n_sims,
        rng=rng,
        show_progress=True
    )

    print()
    generate_report(results, params, payout_strategy, args.output)


def cmd_combine(args) -> None:
    """Simulate combine phase only."""
    params = get_strategy_params(args)
    rng = np.random.default_rng(args.seed) if args.seed else None

    print("=" * 60)
    print(" TOPSTEP 50K COMBINE SIMULATION")
    print("=" * 60)
    print()
    print(params.summary())
    print()
    print(f"Simulations: {args.n_sims:,}")
    print()
    print("Running simulation...")

    results = simulate_combine_batch(
        params=params,
        n_sims=args.n_sims,
        rng=rng
    )

    print()
    print(results.summary())


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == 'simulate':
            cmd_simulate(args)
        elif args.command == 'optimize':
            cmd_optimize(args)
        elif args.command == 'backtest':
            cmd_backtest(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'combine':
            cmd_combine(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
