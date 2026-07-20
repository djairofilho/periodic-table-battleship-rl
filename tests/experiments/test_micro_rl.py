"""Tests for the tabular comparison under the exact microgame oracle."""

from __future__ import annotations

import pytest

from periodic_table_battleship_rl.algorithms import SparseQTable, TabularTrainingConfig
from periodic_table_battleship_rl.experiments.micro_rl import (
    evaluate_greedy_q_table_exact,
    run_micro_rl_comparison,
)
from periodic_table_battleship_rl.oracle import ExactBattleshipOracle
from periodic_table_battleship_rl.toy import MicroBattleshipEnv


def test_micro_environment_state_is_the_oracle_public_state() -> None:
    env = MicroBattleshipEnv()
    _, info = env.reset(seed=11)
    assert env.state_index_for_belief(info["belief"]) == env.state_index

    state = info["belief"]
    action = ExactBattleshipOracle(env.config).valid_actions(state)[0]
    _, _, _, _, next_info = env.step(action)
    assert env.state_index_for_belief(next_info["belief"]) == env.state_index


def test_empty_q_table_has_exact_masked_random_value() -> None:
    table = SparseQTable(action_count=MicroBattleshipEnv().action_space.n)
    assert evaluate_greedy_q_table_exact(table) == pytest.approx(20 / 3)


def test_q_learning_and_sarsa_comparison_is_reproducible_and_oracle_bounded() -> None:
    config = TabularTrainingConfig(episodes=300, alpha=0.2, epsilon_end=0.05)
    first = run_micro_rl_comparison(config, seeds=(21, 22))
    second = run_micro_rl_comparison(config, seeds=(21, 22))

    assert first == second
    assert {trial.algorithm for trial in first.trials} == {"q_learning", "sarsa"}
    assert all(trial.regret_vs_oracle >= -1e-12 for trial in first.trials)
