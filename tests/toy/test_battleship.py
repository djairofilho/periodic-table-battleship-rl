"""Tests for the isolated finite environment used by tabular RL."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from periodic_table_battleship_rl.toy import TinyBattleshipEnv


def test_tiny_environment_passes_gymnasium_checker() -> None:
    check_env(TinyBattleshipEnv(), skip_render_check=True)


def test_reset_is_seed_reproducible_without_exposing_hidden_target() -> None:
    env = TinyBattleshipEnv()

    observation_a, info_a = env.reset(seed=93)
    target_a = env._target_action
    observation_b, info_b = env.reset(seed=93)

    assert target_a == env._target_action
    assert observation_a == observation_b == 0
    assert info_a == info_b == {"is_hit": 0, "total_attempts": 0, "episode_id": 93}
    assert "target" not in info_a


def test_action_mask_starts_full_then_excludes_shot_actions() -> None:
    env = TinyBattleshipEnv()
    env.reset(seed=7)

    initial_mask = env.action_masks()
    assert initial_mask.dtype == np.bool_
    assert initial_mask.shape == (TinyBattleshipEnv.CELL_COUNT,)
    assert initial_mask.all()

    env.step(0)
    assert not env.action_masks()[0]
    assert env.action_masks().sum() == TinyBattleshipEnv.CELL_COUNT - 1


def test_rewards_and_terminal_resolution_for_miss_then_target_hit() -> None:
    env = TinyBattleshipEnv()
    env.reset(seed=11)
    assert env._target_action is not None
    miss = next(action for action in range(env.CELL_COUNT) if action != env._target_action)

    state, reward, terminated, truncated, info = env.step(miss)
    assert reward == TinyBattleshipEnv.MISS_REWARD
    assert not terminated
    assert not truncated
    assert info["is_hit"] == 0
    assert TinyBattleshipEnv.decode_state_index(state).flat[miss] == 1

    state, reward, terminated, truncated, info = env.step(env._target_action)
    assert reward == TinyBattleshipEnv.HIT_REWARD
    assert terminated
    assert not truncated
    assert info["is_hit"] == 1
    assert TinyBattleshipEnv.decode_state_index(state).flat[env._target_action] == 2


def test_repeated_or_out_of_space_actions_are_penalised_noops() -> None:
    env = TinyBattleshipEnv()
    initial_state, _ = env.reset(seed=17)
    assert env._target_action is not None
    miss = next(action for action in range(env.CELL_COUNT) if action != env._target_action)
    state_after_miss, *_ = env.step(miss)

    state, reward, terminated, truncated, _ = env.step(miss)
    assert state == state_after_miss
    assert reward == TinyBattleshipEnv.INVALID_ACTION_REWARD
    assert not terminated
    assert not truncated

    state, reward, terminated, truncated, _ = env.step(TinyBattleshipEnv.CELL_COUNT)
    assert state == state_after_miss
    assert reward == TinyBattleshipEnv.INVALID_ACTION_REWARD
    assert not terminated
    assert not truncated
    assert state != initial_state


def test_state_codec_and_coordinate_api_are_lossless() -> None:
    states = np.array([0, 1, 2, 0] * 4, dtype=np.uint8)

    state_index = TinyBattleshipEnv.encode_cell_states(states)

    assert TinyBattleshipEnv.decode_state_index(state_index).reshape(-1).tolist() == states.tolist()
    assert TinyBattleshipEnv.coordinate_for(6) == (1, 2)
    assert TinyBattleshipEnv.action_for(1, 2) == 6
