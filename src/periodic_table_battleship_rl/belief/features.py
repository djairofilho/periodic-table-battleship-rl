"""Public belief-map features for neural attack policies.

The wrapper is deliberately outside :mod:`envs`: it transforms an already
public observation and ``action_masks`` output.  In particular, it never
reads ``AttackEnv`` private fleet state.  The Monte Carlo proposal remains an
approximation, so metadata must name its sampler rather than claim an exact
posterior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from periodic_table_battleship_rl.belief.model import (
    MonteCarloDiagnostics,
    PublicAttackState,
    sample_compatible_fleets,
)
from periodic_table_battleship_rl.envs.attack import AttackEnv


BELIEF_FEATURE_SCHEMA_VERSION = "public-belief-maps-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class BeliefFeatureConfig:
    """Bounded, public configuration for one belief-map transform."""

    sample_count: int = 16
    max_restarts_per_sample: int = 128
    max_nodes_per_sample: int = 8_192
    sampler_seed: int = 0

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.max_restarts_per_sample <= 0:
            raise ValueError("max_restarts_per_sample must be positive")
        if self.max_nodes_per_sample <= 0:
            raise ValueError("max_nodes_per_sample must be positive")
        if self.sampler_seed < 0:
            raise ValueError("sampler_seed must be non-negative")

    def public_dict(self) -> dict[str, int | str | bool]:
        """Return JSON-native provenance, including approximation limits."""
        return {
            "schema_version": BELIEF_FEATURE_SCHEMA_VERSION,
            "sampler_id": "constrained-backtracking-v1",
            "posterior_exact": False,
            **asdict(self),
        }


class BeliefAugmentedAttackEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Append occupancy and uncertainty maps inferred from public history.

    The output has two extra float planes: estimated occupancy probability and
    binary outcome entropy.  Both are zeroed for unavailable actions.  The
    wrapped environment remains the sole owner of rewards and hidden fleets.
    """

    def __init__(self, environment: AttackEnv, config: BeliefFeatureConfig) -> None:
        if not isinstance(environment, AttackEnv):
            raise TypeError("BeliefAugmentedAttackEnv requires an AttackEnv")
        super().__init__(environment)
        self.feature_config = config
        channels, rows, columns = environment.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels + 2, rows, columns),
            dtype=np.float32,
        )
        self._policy_rng = np.random.default_rng(config.sampler_seed)
        self.last_diagnostics: MonteCarloDiagnostics | None = None

    @property
    def attack_environment(self) -> AttackEnv:
        """Return the wrapped public environment for masks and topology only."""
        return self.env

    def action_masks(self) -> np.ndarray:
        """Expose the unchanged public mask required by MaskablePPO."""
        return self.attack_environment.action_masks()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, int | bool]]:
        super().reset(seed=seed)
        observation, info = self.attack_environment.reset(seed=seed, options=options)
        episode_seed = (
            int(self.np_random.integers(0, np.iinfo(np.uint32).max))
            if seed is None
            else seed
        )
        self._policy_rng = np.random.default_rng(
            np.random.SeedSequence((episode_seed, self.feature_config.sampler_seed))
        )
        return self._augment(observation), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, int | bool]]:
        observation, reward, terminated, truncated, info = self.attack_environment.step(action)
        return self._augment(observation), reward, terminated, truncated, info

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        state = PublicAttackState.from_observation(
            self.attack_environment.topology, observation
        )
        belief, diagnostics = sample_compatible_fleets(
            state,
            sample_count=self.feature_config.sample_count,
            rng=self._policy_rng,
            max_restarts_per_sample=self.feature_config.max_restarts_per_sample,
            max_nodes_per_sample=self.feature_config.max_nodes_per_sample,
        )
        self.last_diagnostics = diagnostics
        action_mask = self.action_masks()
        probabilities = belief.action_probabilities(action_mask)
        entropy = _binary_entropy(probabilities)
        augmented = np.empty(self.observation_space.shape, dtype=np.float32)
        augmented[:-2] = observation.astype(np.float32, copy=False)
        augmented[-2] = probabilities.reshape(
            self.attack_environment.topology.rows,
            self.attack_environment.topology.columns,
        )
        augmented[-1] = entropy.reshape(
            self.attack_environment.topology.rows,
            self.attack_environment.topology.columns,
        )
        return augmented


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return normalized binary entropy without producing NaN at 0 or 1."""
    entropy = np.zeros_like(probabilities, dtype=np.float64)
    interior = (probabilities > 0.0) & (probabilities < 1.0)
    values = probabilities[interior]
    entropy[interior] = -(
        values * np.log2(values) + (1.0 - values) * np.log2(1.0 - values)
    )
    return entropy.astype(np.float32)
