"""Bayesian attacker adapters and validation-only frozen self-play scores.

The planner is used through the same public observation and legal-mask API as
learned attack policies.  It never receives a fleet, fleet IDs, or private
``AttackEnv`` fields.  The frozen suite intentionally accepts validation seeds
only: a self-play pilot must not spend the blind attack test inventory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Literal

import numpy as np

from periodic_table_battleship_rl.belief import BeliefPlanner, PublicAttackState
from periodic_table_battleship_rl.placement.defensive import DefensiveEvaluator
from periodic_table_battleship_rl.selfplay.coupled import (
    FleetSampler,
    PublicAttackPolicyEvaluator,
)
from periodic_table_battleship_rl.topology import Topology


_POLICY_ID_BY_STRATEGY = {
    "probability": "belief_probability_mc-v1",
    "information": "belief_information_mc-v1",
    "horizon-2": "belief_horizon2_mc-v1",
}


@dataclass(slots=True)
class BayesianAttackPolicy:
    """Stateful RNG wrapper around :class:`BeliefPlanner` for public attacks."""

    topology: Topology
    strategy: Literal["probability", "information", "horizon-2"] = "probability"
    sample_count: int = 16
    max_restarts_per_sample: int = 128
    max_nodes_per_sample: int = 8_192
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _planner: BeliefPlanner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._planner = BeliefPlanner(
            self.strategy,
            sample_count=self.sample_count,
            max_restarts_per_sample=self.max_restarts_per_sample,
            max_nodes_per_sample=self.max_nodes_per_sample,
        )
        self.reset(seed=self.seed)

    @property
    def policy_id(self) -> str:
        """Return the stable identity of the planner strategy."""

        return _POLICY_ID_BY_STRATEGY[self.strategy]

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the sole stochastic source used by Monte Carlo sampling."""

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer or None")
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def select_action(
        self, observation: Any, action_mask: Any, *, deterministic: bool = True
    ) -> int:
        """Choose from public history and the legal action mask only."""

        del deterministic
        state = PublicAttackState.from_observation(self.topology, np.asarray(observation))
        mask = np.asarray(action_mask)
        action, _ = self._planner.select_action(state, mask, self._rng)
        return action


@dataclass(frozen=True, slots=True)
class BayesianAttackEvaluator:
    """Freeze a Bayesian planner as a defensive evaluator for placement."""

    topology: Topology
    strategy: Literal["probability", "information", "horizon-2"] = "probability"
    sample_count: int = 16
    max_restarts_per_sample: int = 128
    max_nodes_per_sample: int = 8_192

    @property
    def evaluator_id(self) -> str:
        """Use the same stable identifier in attack and placement reports."""

        return _POLICY_ID_BY_STRATEGY[self.strategy]

    def evaluate(self, fleet, *, rng: np.random.Generator) -> int:
        """Attack a private fleet through a public-observation environment."""

        policy = BayesianAttackPolicy(
            self.topology,
            strategy=self.strategy,
            sample_count=self.sample_count,
            max_restarts_per_sample=self.max_restarts_per_sample,
            max_nodes_per_sample=self.max_nodes_per_sample,
            seed=int(rng.integers(2**32, dtype=np.uint32)),
        )
        return PublicAttackPolicyEvaluator(
            policy=policy,
            topology=self.topology,
            evaluator_id=self.evaluator_id,
        ).evaluate(fleet, rng=rng)


@dataclass(frozen=True, slots=True)
class ValidationFrozenSuiteEvaluator:
    """Score coupled snapshots on immutable targets using validation seeds only."""

    topology: Topology
    validation_seeds: tuple[int, ...]
    attacker_evaluators: Mapping[str, DefensiveEvaluator]
    placement_samplers: Mapping[str, FleetSampler]
    split: Literal["validation"] = "validation"

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("the self-play frozen suite may use validation only")
        if not self.validation_seeds or len(set(self.validation_seeds)) != len(
            self.validation_seeds
        ):
            raise ValueError("validation_seeds must be non-empty and unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.validation_seeds
        ):
            raise ValueError("validation_seeds must be non-negative integers")
        if not self.attacker_evaluators or not self.placement_samplers:
            raise ValueError("both frozen target registries must be non-empty")

    def public_dict(self) -> dict[str, object]:
        """Return the fixed evaluation inputs suitable for a pilot report."""

        return {
            "split": self.split,
            "validation_seeds": list(self.validation_seeds),
            "attacker_evaluator_ids": sorted(self.attacker_evaluators),
            "placement_sampler_ids": sorted(self.placement_samplers),
        }

    def evaluate(
        self,
        *,
        role: str,
        runtime_opponent: FleetSampler | DefensiveEvaluator,
        target_ids: tuple[str, ...],
    ) -> Mapping[str, float]:
        """Return mean valid shots for every requested fixed validation target."""

        if role == "attacker":
            if not isinstance(runtime_opponent, DefensiveEvaluator):
                raise TypeError("attacker snapshots must expose a DefensiveEvaluator")
            registry = self.placement_samplers
            scores = {
                target_id: self._score_attacker(runtime_opponent, registry[target_id], index)
                for index, target_id in enumerate(target_ids)
            }
        elif role == "placer":
            if not isinstance(runtime_opponent, FleetSampler):
                raise TypeError("placer snapshots must expose a FleetSampler")
            registry = self.attacker_evaluators
            scores = {
                target_id: self._score_placer(runtime_opponent, registry[target_id], index)
                for index, target_id in enumerate(target_ids)
            }
        else:
            raise ValueError("role must be 'attacker' or 'placer'")
        if set(scores) != set(target_ids):
            raise KeyError("a requested frozen target is not registered")
        return scores

    def _score_attacker(
        self,
        attacker: DefensiveEvaluator,
        placer: FleetSampler,
        target_index: int,
    ) -> float:
        shots = []
        for seed in self.validation_seeds:
            fleet_rng, attack_rng = _role_rngs(seed, target_index)
            fleet = placer.sample_fleet(self.topology, rng=fleet_rng)
            shots.append(attacker.evaluate(fleet, rng=attack_rng))
        return float(fmean(shots))

    def _score_placer(
        self,
        placer: FleetSampler,
        attacker: DefensiveEvaluator,
        target_index: int,
    ) -> float:
        shots = []
        for seed in self.validation_seeds:
            fleet_rng, attack_rng = _role_rngs(seed, target_index)
            fleet = placer.sample_fleet(self.topology, rng=fleet_rng)
            shots.append(attacker.evaluate(fleet, rng=attack_rng))
        return float(fmean(shots))


def _role_rngs(seed: int, target_index: int) -> tuple[np.random.Generator, np.random.Generator]:
    generated = np.random.SeedSequence((seed, target_index)).generate_state(2)
    return np.random.default_rng(int(generated[0])), np.random.default_rng(int(generated[1]))
