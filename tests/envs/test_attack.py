from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from periodic_table_battleship_rl.envs import AttackEnvironmentConfig, AttackEnv
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP


@pytest.mark.parametrize("topology", [BATTLESHIP, PERIODIC_TABLE_BATTLESHIP])
def test_attack_environment_passes_gymnasium_checker(topology) -> None:
    check_env(AttackEnv(topology), skip_render_check=True)


@pytest.mark.parametrize("topology", [BATTLESHIP, PERIODIC_TABLE_BATTLESHIP])
def test_reset_is_reproducible_and_observation_has_no_hidden_fleet(topology) -> None:
    env = AttackEnv(topology)

    first_observation, first_info = env.reset(seed=843)
    first_fleet = env._fleet
    second_observation, second_info = env.reset(seed=843)

    assert first_fleet == env._fleet
    assert np.array_equal(first_observation, second_observation)
    assert first_observation.dtype == np.uint8
    assert first_observation.shape == (4, 10, 18)
    assert first_observation[1:].sum() == 0
    assert set(first_info) == set(second_info) == {
        "is_hit",
        "sunk_ship_length",
        "valid_shots",
        "invalid_attempts",
        "episode_id",
    }
    assert first_info["is_hit"] is False
    assert "fleet" not in first_info
    assert "occupied_cells" not in first_info


@pytest.mark.parametrize("topology", [BATTLESHIP, PERIODIC_TABLE_BATTLESHIP])
def test_action_mask_has_exactly_the_available_valid_cells(topology) -> None:
    env = AttackEnv(topology)
    env.reset(seed=2)

    initial_mask = env.action_masks()
    assert initial_mask.dtype == np.bool_
    assert initial_mask.shape == (180,)
    assert set(np.flatnonzero(initial_mask)) == set(topology.valid_actions)

    action = next(iter(topology.valid_actions))
    env.step(action)
    assert not env.action_masks()[action]
    assert env.action_masks().sum() == topology.valid_cell_count - 1


def test_invalid_actions_are_noops_and_truncate_at_separate_attempt_limit() -> None:
    env = AttackEnv(PERIODIC_TABLE_BATTLESHIP)
    original_observation, _ = env.reset(seed=3)
    gap_action = next(action for action in range(180) if action not in env.topology.valid_actions)

    observation, reward, terminated, truncated, info = env.step(gap_action)

    assert np.array_equal(observation, original_observation)
    assert reward == -1.0
    assert not terminated
    assert not truncated
    assert info["valid_shots"] == 0
    assert info["invalid_attempts"] == 1
    for _ in range(env.max_total_attempts - 1):
        _, _, terminated, truncated, _ = env.step(gap_action)
    assert not terminated
    assert truncated


@pytest.mark.parametrize("topology", [BATTLESHIP, PERIODIC_TABLE_BATTLESHIP])
def test_masked_policy_always_finishes_within_valid_cell_bound(topology) -> None:
    env = AttackEnv(topology)
    env.reset(seed=5)

    terminated = truncated = False
    while not (terminated or truncated):
        action = int(np.flatnonzero(env.action_masks())[0])
        _, _, terminated, truncated, info = env.step(action)

    assert terminated
    assert not truncated
    assert info["valid_shots"] <= topology.valid_cell_count
    assert info["invalid_attempts"] == 0


def test_sunk_ship_moves_all_of_its_hit_segments_to_sunk_channel() -> None:
    env = AttackEnv(BATTLESHIP)
    env.reset(seed=4)
    assert env._fleet is not None
    placement = env._fleet.placements[0]

    for action in placement.cells[:-1]:
        observation, _, _, _, info = env.step(action)
        assert info["sunk_ship_length"] == 0
        row, column = env.topology.coordinate_for(action)
        assert observation[1, row, column] == 1

    observation, reward, _, _, info = env.step(placement.cells[-1])
    assert reward == 1.0
    assert info["is_hit"] is True
    assert info["sunk_ship_length"] == placement.length
    for action in placement.cells:
        row, column = env.topology.coordinate_for(action)
        assert observation[1, row, column] == 0
        assert observation[2, row, column] == 1


def test_available_observation_profile_tracks_public_action_mask() -> None:
    env = AttackEnv(
        BATTLESHIP,
        config=AttackEnvironmentConfig(observation_profile="outcomes-plus-available-v1"),
    )
    observation, _ = env.reset(seed=4)

    assert observation.shape == (5, 10, 18)
    assert np.array_equal(observation[4].reshape(-1), env.action_masks())
    env.step(next(iter(BATTLESHIP.valid_actions)))
    observation = env._observation()
    assert np.array_equal(observation[4].reshape(-1), env.action_masks())


def test_exploration_reward_only_changes_miss_penalty() -> None:
    control = AttackEnv(BATTLESHIP)
    exploration = AttackEnv(
        BATTLESHIP,
        config=AttackEnvironmentConfig(reward_profile="exploration-v1"),
    )
    control.reset(seed=4)
    exploration.reset(seed=4)
    assert control._fleet is not None
    miss = next(action for action in BATTLESHIP.valid_actions if action not in control._fleet.occupied_cells)

    _, control_reward, _, _, _ = control.step(miss)
    _, exploration_reward, _, _, _ = exploration.step(miss)

    assert control_reward == -1.0
    assert exploration_reward == -0.2
