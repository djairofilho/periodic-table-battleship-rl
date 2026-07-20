"""Public-history Bayesian adapters used by the validation-only self-play pilot."""

from __future__ import annotations

import numpy as np
import pytest

from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.placement.baselines import RandomLegalPlacementPolicy
from periodic_table_battleship_rl.placement.defensive import HuntTargetEvaluator
from periodic_table_battleship_rl.selfplay import (
    BayesianAttackEvaluator,
    BayesianAttackPolicy,
    PlacementPolicyFleetSampler,
    ValidationFrozenSuiteEvaluator,
)
from periodic_table_battleship_rl.topology import BATTLESHIP


def _placer() -> PlacementPolicyFleetSampler:
    return PlacementPolicyFleetSampler(
        policy=RandomLegalPlacementPolicy(BATTLESHIP, seed=10),
        sampler_id="random-placement-validation-v1",
    )


def test_bayesian_policy_uses_public_observation_and_returns_a_legal_action() -> None:
    environment = AttackEnv(BATTLESHIP)
    observation, _ = environment.reset(seed=901)
    policy = BayesianAttackPolicy(BATTLESHIP, sample_count=2, seed=902)

    action = policy.select_action(observation, environment.action_masks())

    assert policy.policy_id == "belief_probability_mc-v1"
    assert environment.action_masks()[action]


def test_bayesian_evaluator_scores_a_fleet_through_public_attack_state() -> None:
    fleet = _placer().sample_fleet(BATTLESHIP, rng=np.random.default_rng(903))

    shots = BayesianAttackEvaluator(BATTLESHIP, sample_count=2).evaluate(
        fleet, rng=np.random.default_rng(904)
    )

    assert 17 <= shots <= BATTLESHIP.valid_cell_count


def test_validation_suite_is_deterministic_and_rejects_blind_test_split() -> None:
    suite = ValidationFrozenSuiteEvaluator(
        topology=BATTLESHIP,
        validation_seeds=(910, 911),
        attacker_evaluators={
            "belief_probability_mc-v1": BayesianAttackEvaluator(
                BATTLESHIP, sample_count=2
            ),
            "hunt-target-v1": HuntTargetEvaluator(BATTLESHIP),
        },
        placement_samplers={"random-placement-validation-v1": _placer()},
    )

    first = suite.evaluate(
        role="placer",
        runtime_opponent=_placer(),
        target_ids=("belief_probability_mc-v1", "hunt-target-v1"),
    )
    second = suite.evaluate(
        role="placer",
        runtime_opponent=_placer(),
        target_ids=("belief_probability_mc-v1", "hunt-target-v1"),
    )

    assert first == second
    assert suite.public_dict()["split"] == "validation"
    with pytest.raises(ValueError, match="validation only"):
        ValidationFrozenSuiteEvaluator(
            topology=BATTLESHIP,
            validation_seeds=(910,),
            attacker_evaluators={"hunt-target-v1": HuntTargetEvaluator(BATTLESHIP)},
            placement_samplers={"random-placement-validation-v1": _placer()},
            split="test",  # type: ignore[arg-type]
        )
