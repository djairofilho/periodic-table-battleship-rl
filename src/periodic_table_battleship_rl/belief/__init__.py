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
from .features import (
    BELIEF_FEATURE_SCHEMA_VERSION,
    BeliefAugmentedAttackEnv,
    BeliefFeatureConfig,
)
from .calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationCase,
    CalibrationCaseResult,
    CalibrationMetrics,
    SamplerCalibration,
    calibrate_constrained_sampler,
    default_micro_calibration_cases,
    rectangular_micro_topology,
)

__all__ = [
    "BeliefPlanner",
    "BeliefAugmentedAttackEnv",
    "BeliefFeatureConfig",
    "BELIEF_FEATURE_SCHEMA_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationCase",
    "CalibrationCaseResult",
    "CalibrationMetrics",
    "BeliefPopulation",
    "CompatibleFleetLimitError",
    "MonteCarloDiagnostics",
    "PublicAttackState",
    "SamplerCalibration",
    "calibrate_constrained_sampler",
    "default_micro_calibration_cases",
    "enumerate_compatible_fleets",
    "exact_belief",
    "information_action",
    "information_gain",
    "probability_action",
    "sample_compatible_fleets",
    "short_horizon_action",
    "rectangular_micro_topology",
]
