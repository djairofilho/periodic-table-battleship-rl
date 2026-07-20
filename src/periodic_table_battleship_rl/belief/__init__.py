"""Public-history belief models and reproducible Battleship planners.

The package never reads ``AttackEnv`` private fleet fields.  A belief is
constructed solely from the public observation channels and action mask.
"""

from .model import (
    BeliefPopulation,
    CompatibleFleetLimitError,
    MonteCarloDiagnostics,
    PublicAttackState,
    enumerate_compatible_fleets,
    exact_belief,
    sample_compatible_fleets,
)
from .planners import (
    BeliefPlanner,
    information_action,
    information_gain,
    probability_action,
    short_horizon_action,
)

__all__ = [
    "BeliefPlanner",
    "BeliefPopulation",
    "CompatibleFleetLimitError",
    "MonteCarloDiagnostics",
    "PublicAttackState",
    "enumerate_compatible_fleets",
    "exact_belief",
    "information_action",
    "information_gain",
    "probability_action",
    "sample_compatible_fleets",
    "short_horizon_action",
]
