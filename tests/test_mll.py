"""
Tests for Maximum Loss Limit (MLL) trailing logic.

The MLL is based on end-of-day high watermark, NOT intraday.
This is critical to model correctly.
"""

import pytest
from topstep_sim.config import (
    CombineRules,
    XFARules,
    calculate_combine_mll,
    calculate_xfa_mll
)


class TestCombineMLL:
    """Test combine MLL calculations."""

    def test_initial_mll(self):
        """MLL starts at starting_balance - max_loss_limit."""
        rules = CombineRules()
        mll = calculate_combine_mll(50_000, rules)
        assert mll == 48_000  # $50,000 - $2,000

    def test_mll_trails_up(self):
        """MLL trails upward as high watermark increases."""
        rules = CombineRules()

        # High watermark at $51,000
        mll = calculate_combine_mll(51_000, rules)
        assert mll == 49_000  # $51,000 - $2,000

        # High watermark at $52,000
        mll = calculate_combine_mll(52_000, rules)
        assert mll == 50_000  # $52,000 - $2,000

    def test_mll_stops_at_starting_balance(self):
        """MLL stops trailing once it reaches starting balance."""
        rules = CombineRules()

        # High watermark at $52,000 -> MLL would be $50,000
        mll = calculate_combine_mll(52_000, rules)
        assert mll == 50_000

        # High watermark at $53,000 -> MLL stays at $50,000
        mll = calculate_combine_mll(53_000, rules)
        assert mll == 50_000

        # High watermark at $55,000 -> MLL still at $50,000
        mll = calculate_combine_mll(55_000, rules)
        assert mll == 50_000

    def test_mll_edge_cases(self):
        """Test edge cases for MLL calculation."""
        rules = CombineRules()

        # Just at starting balance
        mll = calculate_combine_mll(50_000, rules)
        assert mll == 48_000

        # Slightly above threshold where MLL would cap
        mll = calculate_combine_mll(52_001, rules)
        assert mll == 50_000

    def test_combine_mll_sequence(self):
        """
        Test MLL through a sequence of days.

        Day 1: balance goes to $51,000 -> high_watermark = $51,000, MLL = $49,000
        Day 2: balance drops to $49,500 -> above MLL ($49,000) -> survives
        Day 3: balance goes to $52,500 -> high_watermark = $52,500, MLL = $50,000 (capped)
        Day 4: balance drops to $50,400 -> above MLL ($50,000) -> survives
        Day 5: balance drops to $49,900 -> BELOW MLL ($50,000) -> FAIL
        """
        rules = CombineRules()

        # Day 1
        high_watermark = 51_000
        mll = calculate_combine_mll(high_watermark, rules)
        assert mll == 49_000
        balance = 51_000
        assert balance > mll  # Survives

        # Day 2
        balance = 49_500
        # High watermark unchanged
        assert balance > mll  # Survives

        # Day 3
        balance = 52_500
        high_watermark = max(high_watermark, balance)
        mll = calculate_combine_mll(high_watermark, rules)
        assert mll == 50_000  # Capped at starting balance
        assert balance > mll  # Survives

        # Day 4
        balance = 50_400
        assert balance > mll  # Survives

        # Day 5
        balance = 49_900
        assert balance <= mll  # FAILS - below or at MLL


class TestXFAMLL:
    """Test XFA MLL calculations."""

    def test_xfa_initial_mll(self):
        """XFA MLL starts at -$2,000 (the floor)."""
        rules = XFARules()

        # High watermark at $0 (starting)
        mll = calculate_xfa_mll(0, rules)
        assert mll == -2_000

    def test_xfa_mll_trails_from_zero(self):
        """XFA MLL trails up from the floor."""
        rules = XFARules()

        # High watermark at $1,000
        mll = calculate_xfa_mll(1_000, rules)
        assert mll == -1_000  # $1,000 - $2,000

        # High watermark at $2,000
        mll = calculate_xfa_mll(2_000, rules)
        assert mll == 0  # $2,000 - $2,000

    def test_xfa_mll_stops_at_zero(self):
        """XFA MLL stops trailing at $0."""
        rules = XFARules()

        # High watermark at $3,000 -> MLL would be $1,000, but caps at $0
        mll = calculate_xfa_mll(3_000, rules)
        assert mll == 0

        # High watermark at $5,000 -> MLL stays at $0
        mll = calculate_xfa_mll(5_000, rules)
        assert mll == 0

    def test_xfa_post_payout_mll(self):
        """
        After payout, MLL resets to $0.

        If balance is $2,500 after withdrawing $2,500:
        - High watermark resets to $2,500
        - MLL = max(-2000, 2500 - 2000) = 500
        - But MLL is capped at 0
        - So MLL = 0

        Actually, the reset makes MLL_level = 0, meaning the balance
        must stay > 0. The trailing then resumes from the new balance.
        """
        rules = XFARules()

        # After payout, high_watermark resets to current balance
        # Let's say balance is $2,500
        new_high_watermark = 2_500
        mll = calculate_xfa_mll(new_high_watermark, rules)
        assert mll == 0  # Capped at 0

        # If balance later grows to $3,000
        mll = calculate_xfa_mll(3_000, rules)
        assert mll == 0  # Still capped

    def test_xfa_negative_balance(self):
        """Test MLL with negative balance scenarios."""
        rules = XFARules()

        # High watermark at $500
        mll = calculate_xfa_mll(500, rules)
        assert mll == -1_500  # $500 - $2,000

        # Balance can go to -$1,400 and still survive
        balance = -1_400
        assert balance > mll  # Survives

        # Balance at -$1,500 exactly on MLL
        balance = -1_500
        assert balance <= mll  # Fails (at or below)
