"""Debug script to trace simulation behavior."""
import sys
sys.path.insert(0, 'C:/Users/micha')
import numpy as np
from topstep_sim.trade_generator import StrategyParams, TradeGenerator
from topstep_sim.config import CombineRules, calculate_combine_mll

params = StrategyParams(
    trades_per_day=30,
    win_rate=0.50,
    avg_win=750,
    avg_loss=750,
    win_std=10,
    loss_std=15
)

print(f"Per-trade EV: ${params.per_trade_ev:.2f}")
print(f"Daily EV: ${params.daily_ev:.2f}")
print(f"Daily Std: ${params.daily_std:.2f}")

# Simulate a few days
rng = np.random.default_rng(42)
gen = TradeGenerator(params, rng=rng)
rules = CombineRules()

print(f"\nStarting balance: ${rules.starting_balance:,}")
print(f"Target: ${rules.pass_balance:,}")
print(f"Initial MLL: ${calculate_combine_mll(rules.starting_balance, rules):,}")

balance = rules.starting_balance
high_watermark = balance
print("\nSimulating 10 days:")
for day in range(10):
    daily_pnl, trades = gen.generate_day()
    balance += daily_pnl
    high_watermark = max(high_watermark, balance)
    mll = calculate_combine_mll(high_watermark, rules)
    print(f"Day {day+1}: PnL=${daily_pnl:,.0f}, Balance=${balance:,.0f}, HWM=${high_watermark:,.0f}, MLL=${mll:,.0f}")
    if balance <= mll:
        print("  -> MLL BREACH!")
        break
    if balance >= rules.pass_balance:
        print("  -> TARGET REACHED!")
        break

# Also show sample trades from one day
print("\n\nSample trades from one day:")
rng2 = np.random.default_rng(123)
gen2 = TradeGenerator(params, rng=rng2)
daily_pnl, trades = gen2.generate_day()
print(f"Number of trades: {len(trades)}")
print(f"Trade samples: {trades[:10]}")
print(f"Wins: {(trades > 0).sum()}, Losses: {(trades < 0).sum()}")
print(f"Daily PnL: ${daily_pnl:,.2f}")
