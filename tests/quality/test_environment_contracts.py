"""Black-box quality properties shared by the Gymnasium environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.envs.placement import PlacementEnv
from periodic_table_battleship_rl.game import Fleet
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP, Topology


TOPOLOGIES = (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP)


@dataclass(frozen=True)
class RandomizedEvaluator:
    """Evaluator that makes placement reproducibility observable through reward."""

    shots_low: int = 17
    shots_high: int = 90

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        del fleet
        return int(rng.integers(self.shots_low, self.shots_high + 1))


def _first_legal_action(env: PlacementEnv) -> int:
    return int(np.flatnonzero(env.action_masks())[0])


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_attack_seed_replays_public_trajectory_and_masks(topology: Topology) -> None:
    first = AttackEnv(topology)
    second = AttackEnv(topology)

    first_observation, first_info = first.reset(seed=20260720)
    second_observation, second_info = second.reset(seed=20260720)
    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info == second_info

    called: set[int] = set()
    for _ in range(min(12, topology.valid_cell_count)):
        first_mask = first.action_masks()
        second_mask = second.action_masks()
        np.testing.assert_array_equal(first_mask, second_mask)
        assert first_mask.dtype == np.bool_
        assert set(np.flatnonzero(first_mask)) == set(topology.valid_actions) - called

        action = int(np.flatnonzero(first_mask)[len(called) % first_mask.sum()])
        first_result = first.step(action)
        second_result = second.step(action)
        np.testing.assert_array_equal(first_result[0], second_result[0])
        assert first_result[1:] == second_result[1:]
        called.add(action)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_attack_masks_never_offer_gaps_or_previously_called_cells(topology: Topology) -> None:
    env = AttackEnv(topology)
    observation, _ = env.reset(seed=81)
    called: set[int] = set()

    for _ in range(min(20, topology.valid_cell_count)):
        mask = env.action_masks()
        assert env.action_space.contains(int(np.flatnonzero(mask)[0]))
        assert not mask[list(called)].any() if called else True
        assert not mask[[action for action in range(180) if action not in topology.valid_actions]].any()
        np.testing.assert_array_equal(
            observation[0].astype(bool).reshape(-1),
            np.isin(np.arange(180), tuple(topology.valid_actions)),
        )

        action = int(np.flatnonzero(mask)[0])
        observation, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
        called.add(action)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_placement_seed_replays_masks_observations_and_evaluator_reward(
    topology: Topology,
) -> None:
    first = PlacementEnv(topology, RandomizedEvaluator())
    second = PlacementEnv(topology, RandomizedEvaluator())

    first_observation, first_info = first.reset(seed=20260720)
    second_observation, second_info = second.reset(seed=20260720)
    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info == second_info

    for _ in range(5):
        first_mask = first.action_masks()
        second_mask = second.action_masks()
        np.testing.assert_array_equal(first_mask, second_mask)
        action = _first_legal_action(first)
        assert first_mask[action]

        first_result = first.step(action)
        second_result = second.step(action)
        np.testing.assert_array_equal(first_result[0], second_result[0])
        assert first_result[1:] == second_result[1:]

    assert first_result[2] is True
    assert first_result[3] is False
    assert not first.action_masks().any()


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_placement_masks_only_offer_actions_the_environment_accepts(topology: Topology) -> None:
    env = PlacementEnv(topology, RandomizedEvaluator())
    _, _ = env.reset(seed=43)
    accepted_prefix: list[int] = []
    occupied_segments = 0

    for _ in range(5):
        mask = env.action_masks()
        legal_actions = np.flatnonzero(mask)
        assert mask.dtype == np.bool_
        assert legal_actions.size > 0
        assert all(env.action_space.contains(int(action)) for action in legal_actions)

        for action in legal_actions:
            candidate = PlacementEnv(topology, RandomizedEvaluator())
            candidate.reset(seed=43)
            for accepted_action in accepted_prefix:
                candidate.step(accepted_action)
            _, _, _, truncated, info = candidate.step(int(action))
            assert not truncated
            assert info["invalid_action"] is False

        action = int(legal_actions[len(legal_actions) // 2])
        next_observation, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0 or terminated
        assert not truncated
        assert info["invalid_action"] is False
        assert next_observation[1].sum() > occupied_segments
        occupied_segments = int(next_observation[1].sum())
        accepted_prefix.append(action)

    assert terminated
    assert next_observation[1].sum() == 17
    assert not env.action_masks().any()


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_gymnasium_checker_accepts_each_public_environment(topology: Topology) -> None:
    check_env(AttackEnv(topology), skip_render_check=True)
    check_env(PlacementEnv(topology, RandomizedEvaluator()), skip_render_check=True)
