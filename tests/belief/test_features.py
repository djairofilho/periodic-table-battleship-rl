"""Tests for public-only neural belief-map features."""

from __future__ import annotations

import numpy as np

from periodic_table_battleship_rl.belief.features import (
    BeliefAugmentedAttackEnv,
    BeliefFeatureConfig,
)
from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.topology import BATTLESHIP


def _environment() -> BeliefAugmentedAttackEnv:
    return BeliefAugmentedAttackEnv(
        AttackEnv(BATTLESHIP),
        BeliefFeatureConfig(sample_count=2, max_nodes_per_sample=4_096),
    )


def test_belief_maps_are_deterministic_for_a_seed_and_preserve_public_channels() -> None:
    first = _environment()
    second = _environment()
    first_observation, _ = first.reset(seed=77)
    second_observation, _ = second.reset(seed=77)

    assert first_observation.shape == (6, 10, 18)
    assert first_observation.dtype == np.float32
    assert np.array_equal(first_observation, second_observation)
    assert np.array_equal(first_observation[:4], first.attack_environment.reset(seed=77)[0])
    assert first.last_diagnostics is not None
    assert not first.last_diagnostics.posterior_exact


def test_belief_maps_zero_every_masked_action_after_a_public_shot() -> None:
    environment = _environment()
    observation, _ = environment.reset(seed=17)
    action = int(np.flatnonzero(environment.action_masks())[0])
    observation, _, _, _, _ = environment.step(action)
    unavailable = ~environment.action_masks()
    probabilities = observation[-2].reshape(-1)
    entropy = observation[-1].reshape(-1)

    assert np.all(probabilities[unavailable] == 0.0)
    assert np.all(entropy[unavailable] == 0.0)
