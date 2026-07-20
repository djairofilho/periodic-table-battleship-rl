"""Unit tests for masked Q-learning and SARSA on the finite toy board."""

from __future__ import annotations

import numpy as np
import pytest

from periodic_table_battleship_rl.algorithms import (
    SparseQTable,
    TabularTrainingConfig,
    epsilon_greedy_action,
    evaluate_greedy_policy,
    evaluate_random_policy,
    q_learning_update,
    sarsa_update,
    train_q_learning,
    train_sarsa,
)
from periodic_table_battleship_rl.toy import TinyBattleshipEnv


def test_epsilon_greedy_never_selects_a_masked_action() -> None:
    values = np.array([99.0, 1.0, 1.0, -2.0])
    mask = np.array([False, True, True, False])

    selected = {
        epsilon_greedy_action(values, mask, epsilon=epsilon, rng=np.random.default_rng(7))
        for epsilon in (0.0, 1.0)
    }

    assert selected <= {1, 2}
    assert epsilon_greedy_action(
        values, mask, epsilon=0.0, rng=np.random.default_rng(8)
    ) in {1, 2}


def test_q_learning_uses_masked_bootstrap_maximum() -> None:
    table = SparseQTable(action_count=4)
    table.values_for(9)[:] = np.array([100.0, 4.0, 2.0, 3.0])

    updated = q_learning_update(
        table,
        state=2,
        action=1,
        reward=1.0,
        next_state=9,
        next_action_mask=np.array([False, True, True, True]),
        terminated=False,
        truncated=False,
        alpha=0.5,
        gamma=0.5,
    )

    assert updated == pytest.approx(1.5)
    assert table.values_for(2)[1] == pytest.approx(1.5)


def test_q_learning_does_not_bootstrap_after_a_terminal_transition() -> None:
    table = SparseQTable(action_count=2)
    table.values_for(7)[:] = np.array([999.0, 999.0])

    updated = q_learning_update(
        table,
        state=0,
        action=1,
        reward=1.0,
        next_state=7,
        next_action_mask=None,
        terminated=True,
        truncated=False,
        alpha=1.0,
        gamma=1.0,
    )

    assert updated == pytest.approx(1.0)


def test_sarsa_uses_the_action_selected_by_its_behaviour_policy() -> None:
    table = SparseQTable(action_count=4)
    table.values_for(8)[:] = np.array([3.0, 8.0, -4.0, 1.0])

    updated = sarsa_update(
        table,
        state=1,
        action=2,
        reward=-0.5,
        next_state=8,
        next_action=2,
        terminated=False,
        truncated=False,
        alpha=0.5,
        gamma=1.0,
    )

    assert updated == pytest.approx(-2.25)
    assert table.values_for(1)[2] == pytest.approx(-2.25)


@pytest.mark.parametrize("trainer", (train_q_learning, train_sarsa))
def test_training_is_reproducible_with_an_explicit_seed(trainer: object) -> None:
    config = TabularTrainingConfig(episodes=80, alpha=0.2, epsilon_end=0.05)
    train = trainer

    first = train(TinyBattleshipEnv(), config, seed=73)
    second = train(TinyBattleshipEnv(), config, seed=73)

    assert first.algorithm == second.algorithm
    assert first.episode_returns == second.episode_returns
    assert first.episode_lengths == second.episode_lengths
    assert first.q_table.snapshot() == second.q_table.snapshot()


@pytest.mark.parametrize("trainer", (train_q_learning, train_sarsa))
def test_masked_learners_complete_all_tiny_episodes_without_invalid_actions(
    trainer: object,
) -> None:
    config = TabularTrainingConfig(episodes=100, epsilon_end=0.0)
    train = trainer
    result = train(TinyBattleshipEnv(), config, seed=17)

    assert len(result.episode_returns) == config.episodes
    assert all(1 <= length <= TinyBattleshipEnv.CELL_COUNT for length in result.episode_lengths)
    assert all(
        TinyBattleshipEnv.MISS_REWARD * (length - 1) + TinyBattleshipEnv.HIT_REWARD
        == pytest.approx(total_reward)
        for length, total_reward in zip(result.episode_lengths, result.episode_returns)
    )


def test_q_and_sarsa_results_are_evaluated_on_a_shared_seed_protocol() -> None:
    config = TabularTrainingConfig(episodes=120, epsilon_end=0.02)
    q_result = train_q_learning(TinyBattleshipEnv(), config, seed=111)
    sarsa_result = train_sarsa(TinyBattleshipEnv(), config, seed=111)

    q_evaluation = evaluate_greedy_policy(
        TinyBattleshipEnv(), q_result.q_table, episodes=64, seed=29
    )
    sarsa_evaluation = evaluate_greedy_policy(
        TinyBattleshipEnv(), sarsa_result.q_table, episodes=64, seed=29
    )
    random_evaluation = evaluate_random_policy(TinyBattleshipEnv(), episodes=64, seed=29)

    assert q_evaluation.win_rate == sarsa_evaluation.win_rate == random_evaluation.win_rate == 1.0
    assert q_evaluation.mean_episode_length >= 1.0
    assert sarsa_evaluation.mean_episode_length >= 1.0
    assert random_evaluation.mean_episode_length >= 1.0
