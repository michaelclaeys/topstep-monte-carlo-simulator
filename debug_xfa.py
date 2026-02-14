"""Debug script to trace XFA simulation."""
import sys
sys.path.insert(0, 'C:/Users/micha')
import numpy as np
from topstep_sim.trade_generator import StrategyParams, TradeGenerator
from topstep_sim.config import XFARules, calculate_xfa_mll
from topstep_sim.xfa_sim import simulate_xfa, PayoutStrategy

params = StrategyParams(
    trades_per_day=30,
    win_rate=0.50,
    avg_win=750,
    avg_loss=750,
    win_std=10,
    loss_std=15
)

print(f"Daily EV: ${params.daily_ev:.2f}")
print(f"Daily Std: ${params.daily_std:.2f}")

rules = XFARules()
print(f"\nXFA Starting balance: ${rules.starting_balance}")
print(f"XFA MLL floor: ${-rules.max_loss_limit}")
print(f"Benchmark day threshold: ${rules.benchmark_day_threshold}")
print(f"Benchmark days required: {rules.benchmark_days_required}")

# Run 100 XFA simulations and see outcomes
rng = np.random.default_rng(42)
payouts = []
days_active = []
num_payouts_list = []
blowup_count = 0
timeout_count = 0

for i in range(100):
    gen = TradeGenerator(params, rng=rng)
    result = simulate_xfa(gen, rules, PayoutStrategy.immediate(), max_days=500)
    payouts.append(result.total_payouts)
    days_active.append(result.days_active)
    num_payouts_list.append(result.num_payouts)
    if result.reason_closed == 'mll_breach':
        blowup_count += 1
    elif result.reason_closed == 'timeout':
        timeout_count += 1

print(f"\n100 XFA simulations:")
print(f"  Avg payouts: ${np.mean(payouts):,.2f}")
print(f"  Avg num payouts: {np.mean(num_payouts_list):.2f}")
print(f"  Avg days: {np.mean(days_active):.1f}")
print(f"  Blowups: {blowup_count}")
print(f"  Timeouts: {timeout_count}")

# Now trace ONE simulation in detail WITH profit-since-last-payout rule
print("\n" + "="*60)
print("DETAILED TRACE OF ONE XFA (with profit-since-last-payout rule)")
print("="*60)

rng2 = np.random.default_rng(123)
gen = TradeGenerator(params, rng=rng2)

balance = 0.0
high_watermark = 0.0
benchmark_days = 0
total_payouts = 0.0
num_payouts = 0
balance_after_last_payout = 0.0

for day in range(50):  # Just trace 50 days
    daily_pnl, _ = gen.generate_day()
    balance += daily_pnl

    if daily_pnl >= 150:
        benchmark_days += 1

    high_watermark = max(high_watermark, balance)
    mll = calculate_xfa_mll(high_watermark, rules)

    status = ""
    if balance <= mll:
        status = " -> MLL BREACH!"
    elif benchmark_days >= 5 and balance > 0:
        # Check profit-since-last-payout rule
        if num_payouts == 0 or balance > balance_after_last_payout:
            gross = min(5000, balance * 0.5)
            net = gross * 0.9 - 30
            status = f" -> PAYOUT ${net:.0f} (bal_after={balance - gross:.0f})"
            total_payouts += net
            num_payouts += 1
            balance -= gross
            balance_after_last_payout = balance
            high_watermark = balance
            benchmark_days = 0
        else:
            status = f" -> PAYOUT BLOCKED (need bal>{balance_after_last_payout:.0f})"

    print(f"Day {day+1:2d}: PnL=${daily_pnl:+8,.0f} Bal=${balance:8,.0f} HWM=${high_watermark:8,.0f} MLL=${mll:6,.0f} BM={benchmark_days}{status}")

    if balance <= mll:
        break

print(f"\nTotal payouts extracted: ${total_payouts:,.2f}")
print(f"Num payouts: {num_payouts}")
