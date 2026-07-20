"""Exact belief-state solvers for deliberately small Battleship boards."""

from .micro import (
    BaselineResult,
    BeliefState,
    ExactBattleshipOracle,
    MicroBoardConfig,
    MicroFleet,
    OracleComparison,
    OracleSolution,
    enumerate_fleets,
    evaluate_baselines,
)

__all__ = [
    "BaselineResult",
    "BeliefState",
    "ExactBattleshipOracle",
    "MicroBoardConfig",
    "MicroFleet",
    "OracleComparison",
    "OracleSolution",
    "enumerate_fleets",
    "evaluate_baselines",
]
