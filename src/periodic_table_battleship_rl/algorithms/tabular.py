"""Masked Q-learning and SARSA reference implementations.

These algorithms intentionally support only the finite ``TinyBattleshipEnv``.
They use a sparse state-to-action table because the full ternary state space is
large even for a 4 by 4 board, while a short training run visits few states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np



class MaskedTabularEnv(Protocol):
    """Finite environment contract consumed by the tabular learners."""

    action_space: Any

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]: ...

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]: ...

    def action_masks(self) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TabularTrainingConfig:
    """Hyperparameters shared by Q-learning and SARSA training."""

    episodes: int
    alpha: float = 0.15
    gamma: float = 1.0
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= self.epsilon_start <= 1:
            raise ValueError("epsilon_start must be in [0, 1]")
        if not 0 <= self.epsilon_end <= 1:
            raise ValueError("epsilon_end must be in [0, 1]")

    def epsilon_for_episode(self, episode_index: int) -> float:
        """Linearly anneal epsilon over the configured training episodes."""

        if episode_index not in range(self.episodes):
            raise ValueError("episode_index must refer to a configured episode")
        if self.episodes == 1:
            return self.epsilon_end
        progress = episode_index / (self.episodes - 1)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)


class SparseQTable:
    """Sparse action values indexed by finite integer state observations."""

    def __init__(self, action_count: int) -> None:
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        self.action_count = action_count
        self._values: dict[int, np.ndarray] = {}

    def values_for(self, state: int) -> np.ndarray:
        """Return mutable action values for ``state``, creating zeros if absent."""

        if state < 0:
            raise ValueError("state must be non-negative")
        return self._values.setdefault(state, np.zeros(self.action_count, dtype=float))

    def values_or_zeros(self, state: int) -> np.ndarray:
        """Return known values or transient zeros without expanding the table."""

        if state < 0:
            raise ValueError("state must be non-negative")
        return self._values.get(state, np.zeros(self.action_count, dtype=float))

    def snapshot(self) -> Mapping[int, tuple[float, ...]]:
        """Return a deterministic, immutable view for result comparisons."""

        return {
            state: tuple(float(value) for value in values)
            for state, values in sorted(self._values.items())
        }


@dataclass(frozen=True, slots=True)
class AlgorithmEvaluation:
    """Paired evaluation metrics for one deterministic policy rollout set."""

    returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    wins: tuple[bool, ...]

    @property
    def mean_return(self) -> float:
        """Return the arithmetic mean episode reward."""

        return float(np.mean(self.returns))

    @property
    def mean_episode_length(self) -> float:
        """Return the arithmetic mean number of valid masked shots."""

        return float(np.mean(self.episode_lengths))

    @property
    def win_rate(self) -> float:
        """Return the fraction of episodes that found the target."""

        return float(np.mean(self.wins))


@dataclass(frozen=True, slots=True)
class TabularTrainingResult:
    """Comparable learning curve and learned values for one tabular method."""

    algorithm: str
    config: TabularTrainingConfig
    seed: int
    q_table: SparseQTable
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]

    @property
    def mean_return(self) -> float:
        """Return the arithmetic mean reward over training episodes."""

        return float(np.mean(self.episode_returns))


def epsilon_greedy_action(
    action_values: np.ndarray,
    action_mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Sample an epsilon-greedy action using only entries admitted by a mask."""

    valid_actions = _validate_mask(action_mask, len(action_values))
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if rng.random() < epsilon:
        return int(valid_actions[int(rng.integers(len(valid_actions)))])

    valid_values = action_values[valid_actions]
    best_actions = valid_actions[valid_values == valid_values.max()]
    return int(best_actions[int(rng.integers(len(best_actions)))])


def q_learning_update(
    q_table: SparseQTable,
    *,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    next_action_mask: np.ndarray | None,
    terminated: bool,
    truncated: bool,
    alpha: float,
    gamma: float,
) -> float:
    """Apply one masked off-policy Q-learning update and return the new value."""

    _validate_update_parameters(q_table, action, alpha, gamma)
    done = terminated or truncated
    target = float(reward)
    if not done:
        if next_action_mask is None:
            raise ValueError("next_action_mask is required before episode completion")
        valid_actions = _validate_mask(next_action_mask, q_table.action_count)
        target += gamma * float(q_table.values_for(next_state)[valid_actions].max())
    return _apply_td_update(q_table.values_for(state), action, target, alpha)


def sarsa_update(
    q_table: SparseQTable,
    *,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    next_action: int | None,
    terminated: bool,
    truncated: bool,
    alpha: float,
    gamma: float,
) -> float:
    """Apply one on-policy SARSA update and return the new action value."""

    _validate_update_parameters(q_table, action, alpha, gamma)
    done = terminated or truncated
    target = float(reward)
    if not done:
        if next_action is None:
            raise ValueError("next_action is required before episode completion")
        if next_action not in range(q_table.action_count):
            raise ValueError("next_action must be a valid action index")
        target += gamma * float(q_table.values_for(next_state)[next_action])
    return _apply_td_update(q_table.values_for(state), action, target, alpha)


def train_q_learning(
    env: MaskedTabularEnv,
    config: TabularTrainingConfig,
    *,
    seed: int,
) -> TabularTrainingResult:
    """Train masked Q-learning with explicit, isolated environment and policy seeds."""

    q_table = SparseQTable(env.action_space.n)
    seed_sequence = np.random.SeedSequence(seed)
    environment_sequence, policy_sequence = seed_sequence.spawn(2)
    environment_rng = np.random.default_rng(environment_sequence)
    policy_rng = np.random.default_rng(policy_sequence)
    returns: list[float] = []
    lengths: list[int] = []

    for episode_index in range(config.episodes):
        state, _ = env.reset(seed=int(environment_rng.integers(2**32)))
        terminated = truncated = False
        total_reward = 0.0
        length = 0
        epsilon = config.epsilon_for_episode(episode_index)
        while not (terminated or truncated):
            action = epsilon_greedy_action(
                q_table.values_for(state), env.action_masks(), epsilon, policy_rng
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            q_learning_update(
                q_table,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action_mask=None if terminated or truncated else env.action_masks(),
                terminated=terminated,
                truncated=truncated,
                alpha=config.alpha,
                gamma=config.gamma,
            )
            state = next_state
            total_reward += reward
            length += 1
        returns.append(total_reward)
        lengths.append(length)

    return TabularTrainingResult(
        algorithm="q_learning",
        config=config,
        seed=seed,
        q_table=q_table,
        episode_returns=tuple(returns),
        episode_lengths=tuple(lengths),
    )


def train_sarsa(
    env: MaskedTabularEnv,
    config: TabularTrainingConfig,
    *,
    seed: int,
) -> TabularTrainingResult:
    """Train masked SARSA with explicit, isolated environment and policy seeds."""

    q_table = SparseQTable(env.action_space.n)
    seed_sequence = np.random.SeedSequence(seed)
    environment_sequence, policy_sequence = seed_sequence.spawn(2)
    environment_rng = np.random.default_rng(environment_sequence)
    policy_rng = np.random.default_rng(policy_sequence)
    returns: list[float] = []
    lengths: list[int] = []

    for episode_index in range(config.episodes):
        state, _ = env.reset(seed=int(environment_rng.integers(2**32)))
        epsilon = config.epsilon_for_episode(episode_index)
        action = epsilon_greedy_action(
            q_table.values_for(state), env.action_masks(), epsilon, policy_rng
        )
        terminated = truncated = False
        total_reward = 0.0
        length = 0
        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = None
            if not (terminated or truncated):
                next_action = epsilon_greedy_action(
                    q_table.values_for(next_state),
                    env.action_masks(),
                    epsilon,
                    policy_rng,
                )
            sarsa_update(
                q_table,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                terminated=terminated,
                truncated=truncated,
                alpha=config.alpha,
                gamma=config.gamma,
            )
            state = next_state
            total_reward += reward
            length += 1
            if next_action is not None:
                action = next_action
        returns.append(total_reward)
        lengths.append(length)

    return TabularTrainingResult(
        algorithm="sarsa",
        config=config,
        seed=seed,
        q_table=q_table,
        episode_returns=tuple(returns),
        episode_lengths=tuple(lengths),
    )


def evaluate_greedy_policy(
    env: MaskedTabularEnv,
    q_table: SparseQTable,
    *,
    episodes: int,
    seed: int,
) -> AlgorithmEvaluation:
    """Evaluate a learned table greedily on a deterministic held-out seed stream."""

    return _evaluate(env, episodes=episodes, seed=seed, q_table=q_table)


def evaluate_random_policy(
    env: MaskedTabularEnv, *, episodes: int, seed: int
) -> AlgorithmEvaluation:
    """Evaluate the legal masked-random reference on the same seed protocol."""

    return _evaluate(env, episodes=episodes, seed=seed, q_table=None)


def _evaluate(
    env: MaskedTabularEnv,
    *,
    episodes: int,
    seed: int,
    q_table: SparseQTable | None,
) -> AlgorithmEvaluation:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if q_table is not None and q_table.action_count != env.action_space.n:
        raise ValueError("q_table action count must match environment action count")

    seed_sequence = np.random.SeedSequence(seed)
    environment_sequence, policy_sequence = seed_sequence.spawn(2)
    environment_rng = np.random.default_rng(environment_sequence)
    policy_rng = np.random.default_rng(policy_sequence)
    returns: list[float] = []
    lengths: list[int] = []
    wins: list[bool] = []
    for _ in range(episodes):
        state, _ = env.reset(seed=int(environment_rng.integers(2**32)))
        terminated = truncated = False
        total_reward = 0.0
        length = 0
        while not (terminated or truncated):
            mask = env.action_masks()
            if q_table is None:
                valid_actions = _validate_mask(mask, env.action_space.n)
                action = int(valid_actions[int(policy_rng.integers(len(valid_actions)))])
            else:
                action = epsilon_greedy_action(
                    q_table.values_or_zeros(state), mask, 0.0, policy_rng
                )
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            length += 1
        returns.append(total_reward)
        lengths.append(length)
        wins.append(terminated)
    return AlgorithmEvaluation(tuple(returns), tuple(lengths), tuple(wins))


def _validate_mask(action_mask: np.ndarray, action_count: int) -> np.ndarray:
    if not isinstance(action_mask, np.ndarray):
        raise TypeError("action_mask must be a NumPy array")
    if action_mask.dtype != np.bool_ or action_mask.ndim != 1:
        raise TypeError("action_mask must be a one-dimensional boolean NumPy array")
    if action_mask.size != action_count:
        raise ValueError(f"action_mask must contain {action_count} entries")
    valid_actions = np.flatnonzero(action_mask)
    if not len(valid_actions):
        raise ValueError("action_mask contains no valid actions")
    return valid_actions


def _validate_update_parameters(
    q_table: SparseQTable, action: int, alpha: float, gamma: float
) -> None:
    if action not in range(q_table.action_count):
        raise ValueError("action must be a valid action index")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")


def _apply_td_update(
    action_values: np.ndarray, action: int, target: float, alpha: float
) -> float:
    current_value = float(action_values[action])
    updated_value = current_value + alpha * (target - current_value)
    action_values[action] = updated_value
    return float(updated_value)
