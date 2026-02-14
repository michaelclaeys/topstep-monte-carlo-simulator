"""
TopStep 50K Prop Firm Rules Configuration

All rules and constants for the TopStep 50K Trading Combine and Express Funded Account.
These values are based on TopstepX rules as of 2024.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


# =============================================================================
# TRADING COMBINE RULES
# =============================================================================

COMBINE_STARTING_BALANCE = 50_000
COMBINE_PROFIT_TARGET = 3_000           # Must reach $53,000
COMBINE_MAX_LOSS_LIMIT = 2_000          # Trailing drawdown from EOD high watermark
COMBINE_DAILY_LOSS_LIMIT = None         # NOT enforced on TopstepX
COMBINE_MIN_TRADING_DAYS = 2            # Minimum 2 days with executed trades
COMBINE_CONSISTENCY_RULE = 0.50         # Best day's profit cannot exceed 50% of total profits
COMBINE_MAX_CONTRACTS_MINI = 5          # 5 minis or 50 micros on TopstepX
COMBINE_MAX_CONTRACTS_MICRO = 50
COMBINE_PASS_BALANCE = COMBINE_STARTING_BALANCE + COMBINE_PROFIT_TARGET  # $53,000


# =============================================================================
# EXPRESS FUNDED ACCOUNT (XFA) RULES
# =============================================================================

XFA_STARTING_BALANCE = 0                # Starts at $0
XFA_MAX_LOSS_LIMIT = 2_000              # Can go to -$2,000 before closure
XFA_DAILY_LOSS_LIMIT = None             # NOT enforced on TopstepX
XFA_CONSISTENCY_RULE = None             # No consistency rule in XFA
XFA_PROFIT_TARGET = None                # No profit target — just trade and extract payouts

# Scaling plan (contract limits based on balance)
# Format: (min_balance, max_contracts_mini, max_contracts_micro)
XFA_SCALING_PLAN: List[Tuple[float, int, int]] = [
    (float('-inf'), 2, 20),     # Below $1,500: 2 minis / 20 micros
    (1_500, 3, 30),             # $1,500+: 3 minis / 30 micros
    (2_000, 5, 50),             # $2,000+: 5 minis / 50 micros
]


# =============================================================================
# PAYOUT RULES
# =============================================================================

XFA_BENCHMARK_DAY_THRESHOLD = 150       # A "winning day" requires $150+ net profit
XFA_BENCHMARK_DAYS_REQUIRED = 5         # Need 5 winning days before payout eligible
XFA_MAX_PAYOUT_AMOUNT = 5_000           # Max $5,000 per payout
XFA_MAX_PAYOUT_PERCENT = 0.50           # Or 50% of balance, whichever is lower
XFA_PROFIT_SPLIT = 0.90                 # Trader keeps 90%
XFA_PROCESSING_FEE = 30                 # $30 per payout (ACH/wire)


# =============================================================================
# COST STRUCTURE
# =============================================================================

# Standard Path
COMBINE_MONTHLY_FEE = 49                # Monthly subscription
COMBINE_RESET_FEE = 49                  # Reset after failing
COMBINE_ACTIVATION_FEE = 149            # One-time after passing

# No Activation Fee Path (alternative)
COMBINE_MONTHLY_FEE_NAF = 89            # Higher monthly
COMBINE_ACTIVATION_FEE_NAF = 0          # No activation fee

# Back2Funded
B2F_FEE_50K = 599                       # Reactivation fee
B2F_MAX_USES = 2                        # Max 2 reactivations per XFA
B2F_WINDOW_DAYS = 7                     # Must decide within 7 days
# B2F is only available if NO payout has been taken from the XFA


# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================

DEFAULT_MAX_TRADING_DAYS = 500          # Safety limit for simulation
DEFAULT_MAX_COMBINE_ATTEMPTS = 20       # Give up after this many failed combines


# =============================================================================
# RULE CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class CombineRules:
    """Configuration for Trading Combine rules."""
    starting_balance: float = COMBINE_STARTING_BALANCE
    profit_target: float = COMBINE_PROFIT_TARGET
    max_loss_limit: float = COMBINE_MAX_LOSS_LIMIT
    daily_loss_limit: Optional[float] = COMBINE_DAILY_LOSS_LIMIT
    min_trading_days: int = COMBINE_MIN_TRADING_DAYS
    consistency_rule: float = COMBINE_CONSISTENCY_RULE
    max_contracts_mini: int = COMBINE_MAX_CONTRACTS_MINI
    max_contracts_micro: int = COMBINE_MAX_CONTRACTS_MICRO

    @property
    def pass_balance(self) -> float:
        """Balance required to pass the combine."""
        return self.starting_balance + self.profit_target

    @property
    def initial_mll_level(self) -> float:
        """Initial Maximum Loss Limit level."""
        return self.starting_balance - self.max_loss_limit


@dataclass
class XFARules:
    """Configuration for Express Funded Account rules."""
    starting_balance: float = XFA_STARTING_BALANCE
    max_loss_limit: float = XFA_MAX_LOSS_LIMIT
    daily_loss_limit: Optional[float] = XFA_DAILY_LOSS_LIMIT
    scaling_plan: List[Tuple[float, int, int]] = None
    benchmark_day_threshold: float = XFA_BENCHMARK_DAY_THRESHOLD
    benchmark_days_required: int = XFA_BENCHMARK_DAYS_REQUIRED
    max_payout_amount: float = XFA_MAX_PAYOUT_AMOUNT
    max_payout_percent: float = XFA_MAX_PAYOUT_PERCENT
    profit_split: float = XFA_PROFIT_SPLIT
    processing_fee: float = XFA_PROCESSING_FEE

    def __post_init__(self):
        if self.scaling_plan is None:
            self.scaling_plan = XFA_SCALING_PLAN.copy()

    @property
    def initial_mll_level(self) -> float:
        """Initial Maximum Loss Limit level (floor)."""
        return -self.max_loss_limit

    def get_max_contracts(self, balance: float) -> Tuple[int, int]:
        """Get max contracts (mini, micro) allowed at given balance."""
        max_mini, max_micro = 2, 20  # Default
        for min_bal, mini, micro in self.scaling_plan:
            if balance >= min_bal:
                max_mini, max_micro = mini, micro
        return max_mini, max_micro


@dataclass
class CostStructure:
    """Cost structure for the account lifecycle."""
    monthly_fee: float = COMBINE_MONTHLY_FEE
    reset_fee: float = COMBINE_RESET_FEE
    activation_fee: float = COMBINE_ACTIVATION_FEE
    b2f_fee: float = B2F_FEE_50K
    b2f_max_uses: int = B2F_MAX_USES

    @classmethod
    def standard(cls) -> 'CostStructure':
        """Standard cost path with activation fee."""
        return cls(
            monthly_fee=COMBINE_MONTHLY_FEE,
            reset_fee=COMBINE_RESET_FEE,
            activation_fee=COMBINE_ACTIVATION_FEE,
            b2f_fee=B2F_FEE_50K,
            b2f_max_uses=B2F_MAX_USES
        )

    @classmethod
    def no_activation_fee(cls) -> 'CostStructure':
        """No activation fee path with higher monthly."""
        return cls(
            monthly_fee=COMBINE_MONTHLY_FEE_NAF,
            reset_fee=COMBINE_MONTHLY_FEE_NAF,  # Same as monthly for NAF path
            activation_fee=COMBINE_ACTIVATION_FEE_NAF,
            b2f_fee=B2F_FEE_50K,
            b2f_max_uses=B2F_MAX_USES
        )


@dataclass
class TopStepRules:
    """Complete rule set for TopStep 50K account."""
    combine: CombineRules = None
    xfa: XFARules = None
    costs: CostStructure = None

    def __post_init__(self):
        if self.combine is None:
            self.combine = CombineRules()
        if self.xfa is None:
            self.xfa = XFARules()
        if self.costs is None:
            self.costs = CostStructure.standard()


def get_default_rules() -> TopStepRules:
    """Get default TopStep 50K rules."""
    return TopStepRules()


# =============================================================================
# MLL CALCULATION HELPERS
# =============================================================================

def calculate_combine_mll(high_watermark: float, rules: CombineRules = None) -> float:
    """
    Calculate the Maximum Loss Limit level for the Trading Combine.

    The MLL is based on end-of-day high watermark and trails upward.
    It stops trailing once it reaches the starting balance.

    Args:
        high_watermark: Highest end-of-day balance achieved
        rules: Combine rules (uses defaults if None)

    Returns:
        The MLL level - account closes if balance drops to or below this
    """
    if rules is None:
        rules = CombineRules()

    floor = rules.starting_balance - rules.max_loss_limit  # $48,000
    trailing = high_watermark - rules.max_loss_limit
    cap = rules.starting_balance  # $50,000 - MLL stops here

    return min(cap, max(floor, trailing))


def calculate_xfa_mll(high_watermark: float, rules: XFARules = None) -> float:
    """
    Calculate the Maximum Loss Limit level for the XFA.

    The MLL trails from EOD high watermark, with floor at -$2,000 and cap at $0.
    After payout, high_watermark resets and MLL effectively becomes $0.

    Args:
        high_watermark: Highest end-of-day balance achieved since last payout
        rules: XFA rules (uses defaults if None)

    Returns:
        The MLL level - account closes if balance drops to or below this
    """
    if rules is None:
        rules = XFARules()

    floor = -rules.max_loss_limit  # -$2,000
    trailing = high_watermark - rules.max_loss_limit
    cap = 0  # MLL stops at $0

    return min(cap, max(floor, trailing))


def calculate_payout_amount(balance: float, rules: XFARules = None) -> float:
    """
    Calculate the payout amount based on current balance.

    Payout is min(max_amount, balance * max_percent).

    Args:
        balance: Current account balance
        rules: XFA rules (uses defaults if None)

    Returns:
        Gross payout amount before split and fees
    """
    if rules is None:
        rules = XFARules()

    if balance <= 0:
        return 0.0

    return min(rules.max_payout_amount, balance * rules.max_payout_percent)


def calculate_net_payout(gross_payout: float, rules: XFARules = None) -> float:
    """
    Calculate net payout after profit split and processing fee.

    Args:
        gross_payout: Gross payout amount
        rules: XFA rules (uses defaults if None)

    Returns:
        Net payout received by trader
    """
    if rules is None:
        rules = XFARules()

    return (gross_payout * rules.profit_split) - rules.processing_fee


def check_consistency_rule(daily_pnls: List[float], rules: CombineRules = None) -> bool:
    """
    Check if the consistency rule is satisfied.

    The best day's profit cannot exceed 50% of total profits.

    Args:
        daily_pnls: List of daily P&L values
        rules: Combine rules (uses defaults if None)

    Returns:
        True if consistency rule is satisfied
    """
    if rules is None:
        rules = CombineRules()

    total_profit = sum(daily_pnls)
    if total_profit <= 0:
        return False  # Need positive total profit to pass

    # Only consider profitable days for finding max
    profitable_days = [pnl for pnl in daily_pnls if pnl > 0]
    if not profitable_days:
        return False

    best_day_profit = max(profitable_days)

    return best_day_profit / total_profit <= rules.consistency_rule
