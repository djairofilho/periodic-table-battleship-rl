"""Explainable probability, information, and short-horizon policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from periodic_table_battleship_rl.belief.model import (
    BeliefPopulation,
    MonteCarloDiagnostics,
    PublicAttackState,
    sample_compatible_fleets,
)


def probability_action(belief: BeliefPopulation, action_mask: np.ndarray) -> int:
    """Choose the legal shot with maximal estimated occupancy probability."""
    probabilities = belief.action_probabilities(action_mask)
    return _argmax_stable(probabilities, action_mask)


def information_gain(probability: float) -> float:
    """Return binary outcome information in nats for one prospective shot."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if probability in (0.0, 1.0):
        return 0.0
    return float(-probability * np.log(probability) - (1 - probability) * np.log(1 - probability))


def information_action(belief: BeliefPopulation, action_mask: np.ndarray) -> int:
    """Choose the shot that most evenly partitions the finite belief."""
    probabilities = belief.action_probabilities(action_mask)
    gains = np.array([information_gain(float(value)) for value in probabilities])
    return _argmax_stable(gains, action_mask)


def short_horizon_action(
    belief: BeliefPopulation,
    action_mask: np.ndarray,
    *,
    horizon: int = 2,
) -> int:
    """Maximize expected hits over a finite belief for horizon one or two.

    This intentionally optimizes an auditable surrogate (expected discovered
    segments), rather than claiming to solve the complete POMDP.
    """
    if horizon not in {1, 2}:
        raise ValueError("short-horizon planner supports only horizon 1 or 2")
    probabilities = belief.action_probabilities(action_mask)
    if horizon == 1:
        return _argmax_stable(probabilities, action_mask)
    occupancy = np.zeros((belief.size, action_mask.size), dtype=np.bool_)
    for index, fleet in enumerate(belief.fleets):
        occupancy[index, list(fleet.occupied_cells)] = True
    values = np.full(probabilities.shape, -np.inf, dtype=np.float64)
    for action in np.flatnonzero(action_mask):
        probability = float(probabilities[action])
        next_mask = action_mask.copy()
        next_mask[action] = False
        value = probability
        if not np.any(next_mask):
            values[action] = value
            continue
        if probability > 0.0:
            hit_probabilities = occupancy[occupancy[:, action]].mean(axis=0)
            value += probability * float(np.max(hit_probabilities[next_mask]))
        if probability < 1.0:
            miss_probabilities = occupancy[~occupancy[:, action]].mean(axis=0)
            value += (1.0 - probability) * float(np.max(miss_probabilities[next_mask]))
        values[action] = value
    return _argmax_stable(values, action_mask)


@dataclass(frozen=True, slots=True)
class BeliefPlanner:
    """Stateless public-observation adapter suitable for attack evaluation."""

    strategy: str
    sample_count: int = 128
    max_restarts_per_sample: int = 128
    max_nodes_per_sample: int = 8_192

    def __post_init__(self) -> None:
        if self.strategy not in {"probability", "information", "horizon-2"}:
            raise ValueError("strategy must be probability, information, or horizon-2")
        if min(
            self.sample_count,
            self.max_restarts_per_sample,
            self.max_nodes_per_sample,
        ) <= 0:
            raise ValueError("planner sampling limits must be positive")

    def select_action(
        self,
        state: PublicAttackState,
        action_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, MonteCarloDiagnostics]:
        """Sample only from the supplied public state and choose one action."""
        belief, diagnostics = sample_compatible_fleets(
            state,
            sample_count=self.sample_count,
            rng=rng,
            max_restarts_per_sample=self.max_restarts_per_sample,
            max_nodes_per_sample=self.max_nodes_per_sample,
        )
        if self.strategy == "probability":
            return probability_action(belief, action_mask), diagnostics
        if self.strategy == "information":
            return information_action(belief, action_mask), diagnostics
        return short_horizon_action(belief, action_mask, horizon=2), diagnostics


def _argmax_stable(values: np.ndarray, action_mask: np.ndarray) -> int:
    if values.shape != action_mask.shape:
        raise ValueError("values and action_mask must have the same shape")
    candidates = np.flatnonzero(action_mask)
    if not len(candidates):
        raise ValueError("action_mask contains no valid actions")
    maximum = float(np.max(values[candidates]))
    return int(candidates[np.flatnonzero(values[candidates] == maximum)[0]])
