"""Exact-oracle evaluation for Q-learning and SARSA on the 3 by 3 microgame."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from periodic_table_battleship_rl.algorithms import (
    SparseQTable,
    TabularTrainingConfig,
    TabularTrainingResult,
    train_q_learning,
    train_sarsa,
)
from periodic_table_battleship_rl.oracle import BeliefState, ExactBattleshipOracle
from periodic_table_battleship_rl.toy import MicroBattleshipEnv


Trainer = Callable[[MicroBattleshipEnv, TabularTrainingConfig], TabularTrainingResult]


@dataclass(frozen=True, slots=True)
class MicroRLTrial:
    """One trained table and its exact public-belief evaluation."""

    algorithm: str
    seed: int
    expected_shots: float
    regret_vs_oracle: float
    visited_states: int


@dataclass(frozen=True, slots=True)
class MicroRLComparison:
    """Paired seed results for both tabular learners on one fixed microgame."""

    oracle_expected_shots: float
    config: TabularTrainingConfig
    trials: tuple[MicroRLTrial, ...]

    def rows(self) -> tuple[dict[str, float | int | str], ...]:
        """Return stable, serialization-ready records with aggregate metrics."""

        rows: list[dict[str, float | int | str]] = [
            {
                "algorithm": "dynamic-programming-oracle",
                "seed": "exact",
                "expected_shots": self.oracle_expected_shots,
                "regret_vs_oracle": 0.0,
                "visited_states": 0,
            }
        ]
        for trial in self.trials:
            rows.append(asdict(trial))
        for algorithm in ("q_learning", "sarsa"):
            values = [trial.expected_shots for trial in self.trials if trial.algorithm == algorithm]
            regrets = [trial.regret_vs_oracle for trial in self.trials if trial.algorithm == algorithm]
            visited = [trial.visited_states for trial in self.trials if trial.algorithm == algorithm]
            rows.append(
                {
                    "algorithm": f"{algorithm}-mean",
                    "seed": "aggregate",
                    "expected_shots": float(np.mean(values)),
                    "regret_vs_oracle": float(np.mean(regrets)),
                    "visited_states": int(round(float(np.mean(visited)))),
                }
            )
        return tuple(rows)


def evaluate_greedy_q_table_exact(q_table: SparseQTable) -> float:
    """Score a learned greedy table exactly under the oracle's uniform prior.

    The evaluator only maps the public :class:`BeliefState` to its ternary
    observation.  It never reads the selected hidden fleet of an episode.
    Ties are handled uniformly, matching the learner's greedy action rule.
    """

    env = MicroBattleshipEnv()
    oracle = ExactBattleshipOracle(env.config)
    if q_table.action_count != env.action_space.n:
        raise ValueError("q_table action count must match the microboard")

    def policy(state: BeliefState, planner: ExactBattleshipOracle) -> dict[int, float]:
        state_index = env.state_index_for_belief(state)
        values = q_table.values_or_zeros(state_index)
        valid_actions = planner.valid_actions(state)
        maximum = max(values[action] for action in valid_actions)
        actions = tuple(action for action in valid_actions if values[action] == maximum)
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}

    return oracle.evaluate_policy(policy)


def run_micro_rl_comparison(
    config: TabularTrainingConfig,
    *,
    seeds: tuple[int, ...],
) -> MicroRLComparison:
    """Train Q-learning and SARSA on shared seeds, then evaluate exactly."""

    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    oracle = ExactBattleshipOracle()
    oracle_expected_shots = oracle.solve().expected_shots
    trainers: tuple[Callable[..., TabularTrainingResult], ...] = (
        train_q_learning,
        train_sarsa,
    )
    trials: list[MicroRLTrial] = []
    for trainer in trainers:
        for seed in seeds:
            result = trainer(MicroBattleshipEnv(), config, seed=seed)
            exact_shots = evaluate_greedy_q_table_exact(result.q_table)
            trials.append(
                MicroRLTrial(
                    algorithm=result.algorithm,
                    seed=seed,
                    expected_shots=exact_shots,
                    regret_vs_oracle=exact_shots - oracle_expected_shots,
                    visited_states=len(result.q_table.snapshot()),
                )
            )
    return MicroRLComparison(oracle_expected_shots, config, tuple(trials))
