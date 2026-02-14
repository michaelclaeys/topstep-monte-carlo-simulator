"""
Tests for the Trading Combine consistency rule.

The consistency rule states that the best day's profit cannot exceed
50% of total cumulative profits.
"""

import pytest
from topstep_sim.config import CombineRules, check_consistency_rule


class TestConsistencyRule:
    """Test consistency rule calculations."""

    def test_equal_days_pass(self):
        """Equal profit days should pass consistency."""
        rules = CombineRules()

        # Two equal days: 50% each
        daily_pnls = [1500.0, 1500.0]
        assert check_consistency_rule(daily_pnls, rules) is True

    def test_barely_passing(self):
        """Best day at exactly 50% should pass."""
        rules = CombineRules()

        # Best day is exactly 50%
        daily_pnls = [1500.0, 1500.0]  # Each is 50%
        assert check_consistency_rule(daily_pnls, rules) is True

        # Three days: 1000, 1000, 1000 = 33% each
        daily_pnls = [1000.0, 1000.0, 1000.0]
        assert check_consistency_rule(daily_pnls, rules) is True

    def test_barely_failing(self):
        """Best day just over 50% should fail."""
        rules = CombineRules()

        # Best day is 51% (just over)
        # Total = 3000, best = 1530
        # 1530 / 3000 = 51%
        daily_pnls = [1530.0, 1470.0]
        assert check_consistency_rule(daily_pnls, rules) is False

    def test_big_day_fails(self):
        """One big day that dominates should fail."""
        rules = CombineRules()

        # One big day of $2500, rest small
        daily_pnls = [2500.0, 250.0, 250.0]
        # Total = 3000, best = 2500
        # 2500 / 3000 = 83.3% - FAIL
        assert check_consistency_rule(daily_pnls, rules) is False

    def test_dilution_through_trading(self):
        """
        If you have a big day, you can dilute it by continuing to trade.

        Start: $2500 day -> need total profits >= $5000 for consistency
        After more trading: $2500 is now <= 50% of total
        """
        rules = CombineRules()

        # Day 1: Big $2500 day
        daily_pnls = [2500.0]
        # Can't even check yet - need positive total and at least some days
        # Total = 2500, best = 2500 -> 100% - FAIL
        assert check_consistency_rule(daily_pnls, rules) is False

        # Day 2: Add $500
        daily_pnls = [2500.0, 500.0]
        # Total = 3000, best = 2500 -> 83% - FAIL
        assert check_consistency_rule(daily_pnls, rules) is False

        # Day 3: Add $1000
        daily_pnls = [2500.0, 500.0, 1000.0]
        # Total = 4000, best = 2500 -> 62.5% - FAIL
        assert check_consistency_rule(daily_pnls, rules) is False

        # Day 4: Add $1000
        daily_pnls = [2500.0, 500.0, 1000.0, 1000.0]
        # Total = 5000, best = 2500 -> 50% - PASS
        assert check_consistency_rule(daily_pnls, rules) is True

    def test_negative_days_dont_count(self):
        """Negative days don't count as 'best day' but affect total."""
        rules = CombineRules()

        # Mix of positive and negative days
        daily_pnls = [2000.0, -500.0, 1500.0]
        # Total = 3000, best profitable day = 2000
        # 2000 / 3000 = 66.7% - FAIL
        assert check_consistency_rule(daily_pnls, rules) is False

        # But if we had more positive days
        daily_pnls = [2000.0, -500.0, 1500.0, 1000.0]
        # Total = 4000, best = 2000
        # 2000 / 4000 = 50% - PASS
        assert check_consistency_rule(daily_pnls, rules) is True

    def test_all_negative_fails(self):
        """All negative days should fail consistency."""
        rules = CombineRules()

        daily_pnls = [-100.0, -200.0, -50.0]
        # Total is negative, no profitable days
        assert check_consistency_rule(daily_pnls, rules) is False

    def test_zero_total_fails(self):
        """Zero or negative total profit fails consistency."""
        rules = CombineRules()

        # Break even
        daily_pnls = [500.0, -500.0]
        # Total = 0
        assert check_consistency_rule(daily_pnls, rules) is False

        # Net negative
        daily_pnls = [500.0, -600.0]
        # Total = -100
        assert check_consistency_rule(daily_pnls, rules) is False

    def test_single_day_fails(self):
        """A single profitable day is always 100% and fails."""
        rules = CombineRules()

        daily_pnls = [3000.0]
        # 3000 / 3000 = 100% - FAIL
        assert check_consistency_rule(daily_pnls, rules) is False

    def test_many_small_days_pass(self):
        """Many small consistent days easily pass."""
        rules = CombineRules()

        # 10 days of $300 each
        daily_pnls = [300.0] * 10
        # Total = 3000, best = 300
        # 300 / 3000 = 10% - PASS easily
        assert check_consistency_rule(daily_pnls, rules) is True

    def test_edge_case_two_days(self):
        """
        Minimum 2 days required to pass combine.
        Equal profits on both days = exactly 50% each.
        """
        rules = CombineRules()

        # Exactly $1500 each day to hit $53K from $50K
        daily_pnls = [1500.0, 1500.0]
        # Total = 3000, best = 1500
        # 1500 / 3000 = 50% - PASS
        assert check_consistency_rule(daily_pnls, rules) is True

        # Slightly uneven
        daily_pnls = [1600.0, 1400.0]
        # Total = 3000, best = 1600
        # 1600 / 3000 = 53.3% - FAIL
        assert check_consistency_rule(daily_pnls, rules) is False
