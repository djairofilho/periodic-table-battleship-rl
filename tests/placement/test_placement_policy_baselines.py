"""Tests for independent public-state fleet-placement baselines."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest

from periodic_table_battleship_rl.envs.placement import PlacementEnv
from periodic_table_battleship_rl.placement import (
    DispersionPlacementPolicy,
    HuntTargetResistantPlacementPolicy,
    PlacementBaseline,
    RandomLegalPlacementPolicy,
    default_defensive_mixture,
)
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP, Topology


TOPOLOGIES = (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP)
POLICY_TYPES = (
    RandomLegalPlacementPolicy,
    DispersionPlacementPolicy,
    HuntTargetResistantPlacementPolicy,
)


class _PlacementPolicyFactory(Protocol):
    def __call__(self, topology: Topology, *, seed: int = 0) -> PlacementBaseline:
        """Construct one public placement policy."""


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=lambda topology: topology.name)
@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_placement_baselines_complete_only_legal_actions(
    topology: Topology, policy_type: _PlacementPolicyFactory
) -> None:
    """Every baseline completes a fleet without accessing an invalid action."""

    environment = PlacementEnv(topology, default_defensive_mixture(topology))
    policy = policy_type(topology, seed=2026)
    policy.reset(seed=99)
    observation, _ = environment.reset(seed=7)
    terminated = False
    actions: list[int] = []
    while not terminated:
        mask = environment.action_masks()
        action = policy.select_action(observation, mask)
        assert mask[action]
        actions.append(action)
        observation, _, terminated, _, _ = environment.step(action)

    assert len(actions) == 5
    assert environment.fleet is not None


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=lambda topology: topology.name)
@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_placement_baselines_are_reproducible_after_seeded_reset(
    topology: Topology, policy_type: _PlacementPolicyFactory
) -> None:
    """Equivalent episode resets reproduce all public placement decisions."""

    first = _rollout(topology, policy_type(topology, seed=1), policy_seed=42)
    second = _rollout(topology, policy_type(topology, seed=999), policy_seed=42)

    assert first == second


def test_hunt_target_baseline_avoids_direct_cross_ship_adjacency_when_available() -> None:
    """The anti-hunt rule prioritizes no direct target path between ships."""

    environment = PlacementEnv(BATTLESHIP, default_defensive_mixture(BATTLESHIP))
    observation, _ = environment.reset(seed=5)
    first_action = 0
    observation, _, terminated, _, _ = environment.step(first_action)
    assert not terminated

    policy = HuntTargetResistantPlacementPolicy(BATTLESHIP, seed=8)
    action = policy.select_action(observation, environment.action_masks())
    cells = BATTLESHIP.segment_from(action % 180, "horizontal" if action < 180 else "vertical", 4)

    assert cells is not None
    assert all(neighbor not in set(range(5)) for cell in cells for neighbor in BATTLESHIP.neighbors(cell))


def test_baseline_rejects_non_boolean_or_empty_masks() -> None:
    """Baseline callers receive a clear contract error before a bad action leaks."""

    policy = RandomLegalPlacementPolicy(BATTLESHIP)
    observation = np.zeros((3, 10, 18), dtype=np.float32)

    with pytest.raises(TypeError, match="boolean"):
        policy.select_action(observation, np.zeros(360, dtype=np.int8))
    with pytest.raises(ValueError, match="no legal"):
        policy.select_action(observation, np.zeros(360, dtype=np.bool_))


def _rollout(
    topology: Topology, policy: PlacementBaseline, *, policy_seed: int
) -> tuple[int, ...]:
    """Run one fixed public episode and record its accepted actions."""

    environment = PlacementEnv(topology, default_defensive_mixture(topology))
    policy.reset(seed=policy_seed)
    observation, _ = environment.reset(seed=2026)
    terminated = False
    actions: list[int] = []
    while not terminated:
        action = policy.select_action(observation, environment.action_masks())
        actions.append(action)
        observation, _, terminated, _, _ = environment.step(action)
    return tuple(actions)
