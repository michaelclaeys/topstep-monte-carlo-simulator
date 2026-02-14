"""
Analytics and Visualization Module

Generate plots and analysis for simulation results.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .config import TopStepRules, get_default_rules
from .trade_generator import StrategyParams, TradeGenerator
from .combine_sim import simulate_combine, CombineResult
from .xfa_sim import simulate_xfa, XFAResult, PayoutStrategy
from .lifecycle_sim import LifecycleBatchResults
from .optimizer import OptimizationResults


def check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def plot_lifecycle_distribution(
    results: LifecycleBatchResults,
    save_path: Optional[str] = None,
    title: str = "Lifecycle Profit Distribution"
) -> None:
    """
    Plot histogram of lifecycle net profits.

    Args:
        results: Lifecycle batch results
        save_path: Optional path to save figure
        title: Plot title
    """
    check_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))

    profits = results.distribution

    # Create histogram
    n, bins, patches = ax.hist(
        profits, bins=50, edgecolor='black', alpha=0.7
    )

    # Color bins by positive/negative
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i + 1]) / 2
        if bin_center >= 0:
            patch.set_facecolor('#2ecc71')  # Green
        else:
            patch.set_facecolor('#e74c3c')  # Red

    # Add vertical lines for key statistics
    ax.axvline(results.mean_net_profit, color='blue', linestyle='--',
               linewidth=2, label=f'Mean: ${results.mean_net_profit:,.0f}')
    ax.axvline(results.median_net_profit, color='orange', linestyle='--',
               linewidth=2, label=f'Median: ${results.median_net_profit:,.0f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)

    # Add percentile regions
    p10, p90 = results.percentiles[10], results.percentiles[90]
    ax.axvline(p10, color='gray', linestyle=':', alpha=0.7,
               label=f'10th: ${p10:,.0f}')
    ax.axvline(p90, color='gray', linestyle=':', alpha=0.7,
               label=f'90th: ${p90:,.0f}')

    ax.set_xlabel('Net Profit ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')

    # Format x-axis
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add text box with key stats
    textstr = (
        f"N = {results.n_sims:,}\n"
        f"P(Profitable) = {results.p_profitable:.1%}\n"
        f"Std Dev = ${results.std_net_profit:,.0f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_lifecycle_ev_heatmap(
    results_df: pd.DataFrame,
    x_param: str = 'win_rate',
    y_param: str = 'avg_loss',
    save_path: Optional[str] = None,
    title: str = "Lifecycle EV Heatmap"
) -> None:
    """
    Plot 2D heatmap of lifecycle EV as function of two parameters.

    Args:
        results_df: DataFrame from optimization with lifecycle_ev column
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        save_path: Optional path to save figure
        title: Plot title
    """
    check_matplotlib()

    # Pivot the data
    pivot = results_df.pivot_table(
        values='lifecycle_ev',
        index=y_param,
        columns=x_param,
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

    # Set axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))

    # Format tick labels based on parameter type
    if x_param == 'win_rate':
        x_labels = [f'{v:.0%}' for v in pivot.columns]
    else:
        x_labels = [f'${v:.0f}' if v >= 1 else f'{v}' for v in pivot.columns]

    if y_param == 'win_rate':
        y_labels = [f'{v:.0%}' for v in pivot.index]
    else:
        y_labels = [f'${v:.0f}' if v >= 1 else f'{v}' for v in pivot.index]

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Lifecycle EV ($)', fontsize=10)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            color = 'white' if abs(value) > (pivot.values.max() - pivot.values.min()) / 2 else 'black'
            ax.text(j, i, f'${value:,.0f}', ha='center', va='center',
                    color=color, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_equity_curves(
    params: StrategyParams,
    rules: Optional[TopStepRules] = None,
    n_samples: int = 20,
    phase: str = 'combine',
    save_path: Optional[str] = None,
    rng: Optional[np.random.Generator] = None
) -> None:
    """
    Plot sample equity curves for combine or XFA phase.

    Args:
        params: Strategy parameters
        rules: TopStep rules
        n_samples: Number of sample curves to plot
        phase: 'combine' or 'xfa'
        save_path: Optional path to save figure
        rng: Random number generator
    """
    check_matplotlib()

    if rules is None:
        rules = get_default_rules()
    if rng is None:
        rng = np.random.default_rng()

    fig, ax = plt.subplots(figsize=(14, 8))

    colors_pass = plt.cm.Greens(np.linspace(0.4, 0.8, n_samples))
    colors_fail = plt.cm.Reds(np.linspace(0.4, 0.8, n_samples))

    pass_count = 0
    fail_count = 0

    for i in range(n_samples):
        generator = TradeGenerator(params, rng=rng)

        if phase == 'combine':
            result = simulate_combine(generator, rules.combine)
            starting = rules.combine.starting_balance
            daily_pnls = result.daily_pnls
            passed = result.passed
        else:
            payout_strategy = PayoutStrategy.immediate()
            result = simulate_xfa(generator, rules.xfa, payout_strategy)
            starting = rules.xfa.starting_balance
            daily_pnls = result.daily_pnls
            passed = result.num_payouts > 0

        # Build equity curve
        equity = [starting]
        for pnl in daily_pnls:
            equity.append(equity[-1] + pnl)

        days = range(len(equity))

        if passed:
            ax.plot(days, equity, color=colors_pass[pass_count % n_samples],
                    alpha=0.6, linewidth=1)
            pass_count += 1
        else:
            ax.plot(days, equity, color=colors_fail[fail_count % n_samples],
                    alpha=0.6, linewidth=1)
            fail_count += 1

    # Add reference lines for combine
    if phase == 'combine':
        ax.axhline(rules.combine.pass_balance, color='green', linestyle='--',
                   linewidth=2, label=f'Target: ${rules.combine.pass_balance:,}')
        ax.axhline(rules.combine.starting_balance - rules.combine.max_loss_limit,
                   color='red', linestyle='--', linewidth=2,
                   label=f'Initial MLL: ${rules.combine.starting_balance - rules.combine.max_loss_limit:,}')

    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Account Balance ($)', fontsize=12)
    ax.set_title(f'{phase.upper()} Equity Curves ({n_samples} samples)', fontsize=14)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box
    textstr = f"Green: {'Passed' if phase == 'combine' else 'Had Payout'} ({pass_count})\nRed: {'Failed' if phase == 'combine' else 'No Payout'} ({fail_count})"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_payout_strategy_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot bar chart comparing lifecycle EV across payout strategies.

    Args:
        results_df: DataFrame from optimization with payout_strategy column
        save_path: Optional path to save figure
    """
    check_matplotlib()

    # Group by payout strategy
    grouped = results_df.groupby('payout_strategy')['lifecycle_ev'].agg(['mean', 'std'])
    grouped = grouped.sort_values('mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = grouped.index.tolist()
    means = grouped['mean'].values
    stds = grouped['std'].values

    y_pos = range(len(strategies))

    # Color bars by positive/negative
    colors = ['#2ecc71' if m >= 0 else '#e74c3c' for m in means]

    bars = ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.7,
                   edgecolor='black', capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies)
    ax.set_xlabel('Lifecycle EV ($)', fontsize=12)
    ax.set_title('Payout Strategy Comparison', fontsize=14)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 50, i, f'${mean:,.0f}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_sensitivity(
    sensitivity: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None
) -> None:
    """
    Plot sensitivity analysis for each parameter.

    Args:
        sensitivity: Dictionary from OptimizationResults.sensitivity
        save_path: Optional base path for saving figures
    """
    check_matplotlib()

    n_params = len(sensitivity)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (param, df) in enumerate(sensitivity.items()):
        ax = axes[idx]

        x = df[param].values
        y = df['mean_ev'].values
        err = df['std_ev'].values

        ax.errorbar(x, y, yerr=err, marker='o', capsize=5, linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Format x-axis based on parameter
        if param == 'win_rate':
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0%}'))
        elif param in ['avg_win', 'avg_loss']:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:.0f}'))

        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Lifecycle EV ($)')
        ax.set_title(f'Sensitivity: {param}')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(sensitivity), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_ev_vs_expectancy(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Scatter plot of per-trade expectancy vs lifecycle EV.

    Shows how near-zero-expectancy strategies can still be +EV.

    Args:
        results_df: DataFrame from optimization
        save_path: Optional path to save figure
    """
    check_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 8))

    x = results_df['per_trade_ev'].values
    y = results_df['lifecycle_ev'].values

    # Color by whether lifecycle is positive
    colors = ['#2ecc71' if ev >= 0 else '#e74c3c' for ev in y]

    ax.scatter(x, y, c=colors, alpha=0.5, s=20)

    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'b--', alpha=0.5, label=f'Trend')

    # Add reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('Per-Trade Expectancy ($)', fontsize=12)
    ax.set_ylabel('Lifecycle EV ($)', fontsize=12)
    ax.set_title('Per-Trade Expectancy vs Lifecycle EV', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Highlight the "prop firm advantage" region
    # Where per_trade_ev is near zero but lifecycle_ev is positive
    ax.axvspan(-2, 2, alpha=0.1, color='blue', label='Near-zero expectancy')

    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_report(
    results: LifecycleBatchResults,
    params: StrategyParams,
    payout_strategy: PayoutStrategy,
    output_dir: str = "results"
) -> str:
    """
    Generate a complete report with plots and text summary.

    Args:
        results: Lifecycle batch results
        params: Strategy parameters
        payout_strategy: Payout strategy used
        output_dir: Directory to save outputs

    Returns:
        Path to the summary text file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plots
    if HAS_MATPLOTLIB:
        plot_lifecycle_distribution(
            results,
            save_path=str(output_path / "lifecycle_distribution.png")
        )

        plot_equity_curves(
            params,
            phase='combine',
            save_path=str(output_path / "combine_equity_curves.png")
        )

        plot_equity_curves(
            params,
            phase='xfa',
            save_path=str(output_path / "xfa_equity_curves.png")
        )

    # Generate text report
    report = []
    report.append("=" * 60)
    report.append(" TOPSTEP 50K LIFECYCLE SIMULATION RESULTS")
    report.append("=" * 60)
    report.append("")
    report.append(params.summary())
    report.append("")
    report.append(f"Payout Strategy: {payout_strategy.strategy_type.value}")
    if payout_strategy.strategy_type.value == 'target_balance':
        report.append(f"  Target Balance: ${payout_strategy.target_balance:,.0f}")
    report.append("")
    report.append(results.summary())
    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)

    # Save text report
    report_path = output_path / "simulation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {output_path}")

    return str(report_path)
