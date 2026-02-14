"""
Tests for XFA payout mechanics.

Payout amount is min($5,000, balance * 50%).
Net payout = gross * 90% - $30 processing fee.
"""

import pytest
from topstep_sim.config import (
    XFARules,
    calculate_payout_amount,
    calculate_net_payout
)
from topstep_sim.xfa_sim import PayoutStrategy, PayoutStrategyType


class TestPayoutCalculation:
    """Test payout amount calculations."""

    def test_payout_50_percent_of_balance(self):
        """Payout is 50% of balance when under cap."""
        rules = XFARules()

        # $4,000 balance -> $2,000 payout (50%)
        payout = calculate_payout_amount(4_000, rules)
        assert payout == 2_000

        # $6,000 balance -> $3,000 payout (50%)
        payout = calculate_payout_amount(6_000, rules)
        assert payout == 3_000

    def test_payout_capped_at_5000(self):
        """Payout is capped at $5,000."""
        rules = XFARules()

        # $10,000 balance -> $5,000 payout (50% = $5,000 = cap)
        payout = calculate_payout_amount(10_000, rules)
        assert payout == 5_000

        # $12,000 balance -> $5,000 payout (50% = $6,000 but capped)
        payout = calculate_payout_amount(12_000, rules)
        assert payout == 5_000

        # $20,000 balance -> still $5,000 (capped)
        payout = calculate_payout_amount(20_000, rules)
        assert payout == 5_000

    def test_payout_zero_or_negative_balance(self):
        """Zero or negative balance yields no payout."""
        rules = XFARules()

        payout = calculate_payout_amount(0, rules)
        assert payout == 0

        payout = calculate_payout_amount(-500, rules)
        assert payout == 0

    def test_payout_edge_case_at_cap_threshold(self):
        """Test balance exactly at cap threshold."""
        rules = XFARules()

        # At exactly $10,000, 50% = $5,000 = cap
        payout = calculate_payout_amount(10_000, rules)
        assert payout == 5_000

        # Just below: $9,999 -> $4,999.50
        payout = calculate_payout_amount(9_999, rules)
        assert payout == pytest.approx(4_999.50)


class TestNetPayout:
    """Test net payout after split and fees."""

    def test_net_payout_calculation(self):
        """Net payout = gross * 90% - $30."""
        rules = XFARules()

        # $2,000 gross -> $1,800 - $30 = $1,770
        net = calculate_net_payout(2_000, rules)
        assert net == 1_770

        # $5,000 gross -> $4,500 - $30 = $4,470
        net = calculate_net_payout(5_000, rules)
        assert net == 4_470

    def test_net_payout_small_amounts(self):
        """Small payouts still have $30 fee."""
        rules = XFARules()

        # $100 gross -> $90 - $30 = $60
        net = calculate_net_payout(100, rules)
        assert net == 60

        # $33.33 gross -> $30 - $30 = $0
        net = calculate_net_payout(33.33, rules)
        assert net == pytest.approx(0, abs=0.01)

    def test_net_payout_below_fee(self):
        """Very small payouts can result in negative net."""
        rules = XFARules()

        # $20 gross -> $18 - $30 = -$12
        net = calculate_net_payout(20, rules)
        assert net == -12


class TestPayoutStrategy:
    """Test payout strategy decision logic."""

    def test_immediate_strategy(self):
        """Immediate strategy takes payout as soon as eligible."""
        strategy = PayoutStrategy.immediate()
        rules = XFARules()

        # Eligible: 5 benchmark days, positive balance
        should_take = strategy.should_take_payout(
            balance=1_000,
            benchmark_days=5,
            rules=rules,
            num_payouts_taken=0
        )
        assert should_take is True

        # Not enough benchmark days
        should_take = strategy.should_take_payout(
            balance=1_000,
            benchmark_days=4,
            rules=rules,
            num_payouts_taken=0
        )
        assert should_take is False

    def test_target_balance_strategy(self):
        """Target balance strategy waits for target."""
        strategy = PayoutStrategy.target_balance(5_000)
        rules = XFARules()

        # Below target
        should_take = strategy.should_take_payout(
            balance=3_000,
            benchmark_days=5,
            rules=rules,
            num_payouts_taken=0
        )
        assert should_take is False

        # At target
        should_take = strategy.should_take_payout(
            balance=5_000,
            benchmark_days=5,
            rules=rules,
            num_payouts_taken=0
        )
        assert should_take is True

        # Above target
        should_take = strategy.should_take_payout(
            balance=7_000,
            benchmark_days=5,
            rules=rules,
            num_payouts_taken=0
        )
        assert should_take is True

    def test_max_payout_strategy(self):
        """Max payout strategy waits for $10K balance."""
        strategy = PayoutStrategy.max_payout()
        rules = XFARules()

        # Below threshold for max payout
        should_take = strategy.should_take_payout(
            balance=8_000,
            benchmark_days=5,
            rules=rules,
            num_payouts_taken=0
        )
        assert should_take is False

        # At threshold
        should_take = strategy.should_take_payout(
            balance=10_000,
            benchmark_days=5,
            rules=rules,
            num_payouts_taken=0
        )
        assert should_take is True

    def test_one_and_done_strategy(self):
        """One and done only takes first payout."""
        strategy = PayoutStrategy.one_and_done()
        rules = XFARules()

        # First payout
        should_take = strategy.should_take_payout(
            balance=5_000,
            benchmark_days=5,
            rules=rules,
            num_payouts_taken=0
        )
        assert should_take is True

        # Second payout - should not take
        should_take = strategy.should_take_payout(
            balance=5_000,
            benchmark_days=5,
            rules=rules,
            num_payouts_taken=1
        )
        assert should_take is False

    def test_benchmark_days_required(self):
        """All strategies require 5 benchmark days."""
        rules = XFARules()

        for strategy in [
            PayoutStrategy.immediate(),
            PayoutStrategy.target_balance(1_000),
            PayoutStrategy.max_payout(),
            PayoutStrategy.one_and_done()
        ]:
            # 4 days - not enough
            should_take = strategy.should_take_payout(
                balance=15_000,  # High enough for any strategy
                benchmark_days=4,
                rules=rules,
                num_payouts_taken=0
            )
            assert should_take is False, f"Strategy {strategy.strategy_type} should require 5 days"

    def test_positive_balance_required(self):
        """All strategies require positive balance."""
        rules = XFARules()

        for strategy in [
            PayoutStrategy.immediate(),
            PayoutStrategy.target_balance(0),  # Even with 0 target
        ]:
            should_take = strategy.should_take_payout(
                balance=0,
                benchmark_days=5,
                rules=rules,
                num_payouts_taken=0
            )
            assert should_take is False

            should_take = strategy.should_take_payout(
                balance=-100,
                benchmark_days=5,
                rules=rules,
                num_payouts_taken=0
            )
            assert should_take is False
