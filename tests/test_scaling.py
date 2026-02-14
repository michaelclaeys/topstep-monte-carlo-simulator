"""
Tests for XFA scaling plan and B2F eligibility.

Scaling plan limits contracts based on balance.
B2F is only available if no payout has been taken.
"""

import pytest
from topstep_sim.config import XFARules
from topstep_sim.xfa_sim import get_contract_scale, should_use_b2f


class TestScalingPlan:
    """Test XFA scaling plan contract limits."""

    def test_default_scaling_thresholds(self):
        """Test contract limits at various balance levels."""
        rules = XFARules()

        # Below $1,500: 2 minis / 20 micros
        max_mini, max_micro = rules.get_max_contracts(0)
        assert max_mini == 2
        assert max_micro == 20

        max_mini, max_micro = rules.get_max_contracts(1_499)
        assert max_mini == 2
        assert max_micro == 20

        # $1,500+: 3 minis / 30 micros
        max_mini, max_micro = rules.get_max_contracts(1_500)
        assert max_mini == 3
        assert max_micro == 30

        max_mini, max_micro = rules.get_max_contracts(1_999)
        assert max_mini == 3
        assert max_micro == 30

        # $2,000+: 5 minis / 50 micros
        max_mini, max_micro = rules.get_max_contracts(2_000)
        assert max_mini == 5
        assert max_micro == 50

        max_mini, max_micro = rules.get_max_contracts(10_000)
        assert max_mini == 5
        assert max_micro == 50

    def test_negative_balance(self):
        """Negative balance still uses lowest tier."""
        rules = XFARules()

        max_mini, max_micro = rules.get_max_contracts(-500)
        assert max_mini == 2
        assert max_micro == 20

        max_mini, max_micro = rules.get_max_contracts(-1_999)
        assert max_mini == 2
        assert max_micro == 20

    def test_contract_scaling_factor(self):
        """Test contract scale calculation for trade sizing."""
        rules = XFARules()
        base_contracts = 20  # Strategy designed for 20 micros

        # At base level: 20 allowed, 20 base -> scale = 1.0
        scale = get_contract_scale(0, base_contracts, rules)
        assert scale == 1.0

        # At $1,500+: 30 allowed, 20 base -> scale = 1.0 (capped)
        scale = get_contract_scale(1_500, base_contracts, rules)
        assert scale == 1.0

        # If base was 30: at $1,500+, 30 allowed, 30 base -> scale = 1.0
        scale = get_contract_scale(1_500, 30, rules)
        assert scale == 1.0

        # If base was 50: at $1,500, only 30 allowed -> scale = 30/50 = 0.6
        scale = get_contract_scale(1_500, 50, rules)
        assert scale == pytest.approx(0.6)

        # At $2,000+: 50 allowed, 50 base -> scale = 1.0
        scale = get_contract_scale(2_000, 50, rules)
        assert scale == 1.0

    def test_scaling_with_reduced_balance(self):
        """Test scaling when balance drops and contracts are limited."""
        rules = XFARules()

        # Strategy designed for 50 micros
        base = 50

        # At $500 balance, only 20 allowed
        scale = get_contract_scale(500, base, rules)
        assert scale == pytest.approx(20 / 50)  # 0.4

        # This means trades are 40% of normal size
        # A $30 trade becomes $12


class TestB2FEligibility:
    """Test Back2Funded eligibility rules."""

    def test_b2f_positive_ev(self):
        """B2F is worth using when XFA EV > B2F cost."""
        # XFA EV of $1,000 > $499 B2F cost
        assert should_use_b2f(1_000, b2f_cost=499) is True

        # XFA EV of $600 > $499 B2F cost
        assert should_use_b2f(600, b2f_cost=499) is True

    def test_b2f_negative_ev(self):
        """B2F is not worth using when XFA EV < B2F cost."""
        # XFA EV of $400 < $499 B2F cost
        assert should_use_b2f(400, b2f_cost=499) is False

        # XFA EV of $0 < $499
        assert should_use_b2f(0, b2f_cost=499) is False

    def test_b2f_breakeven(self):
        """Test B2F at breakeven point."""
        # XFA EV exactly equal to cost
        assert should_use_b2f(499, b2f_cost=499) is False

        # Just above
        assert should_use_b2f(500, b2f_cost=499) is True

    def test_b2f_with_margin(self):
        """B2F decision with required margin."""
        # Require 1.5x margin (EV must be 1.5x the cost)
        # $499 * 1.5 = $748.50

        assert should_use_b2f(750, b2f_cost=499, margin=1.5) is True
        assert should_use_b2f(700, b2f_cost=499, margin=1.5) is False

    def test_b2f_max_uses(self):
        """
        B2F is limited to 2 uses per XFA.
        This is enforced in lifecycle_sim, not in should_use_b2f.
        """
        # This function only checks EV, not usage count
        # Usage count is checked in simulate_lifecycle
        pass

    def test_b2f_blocked_after_payout(self):
        """
        B2F is blocked if any payout was taken.
        This is enforced in lifecycle_sim based on took_payout flag.
        """
        # This logic is in lifecycle_sim.py
        # XFAResult.took_payout is checked before allowing B2F
        pass
