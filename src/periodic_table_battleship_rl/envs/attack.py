"""Masked single-agent environment for the Battleship attack experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from periodic_table_battleship_rl.game import Fleet, is_legal_fleet, sample_random_legal_fleet
from periodic_table_battleship_rl.topology import Topology


FleetFactory = Callable[[Topology, np.random.Generator], Fleet]
"""Create a hidden legal fleet for one attack episode.

The callable is owned by the environment, never by an attack policy.  It is
used by the coupled self-play adapter while the default remains the benchmark
random-fleet sampler.
"""


@dataclass(frozen=True, slots=True)
class AttackEnvironmentConfig:
    """Public, versioned choices for the attack observation and reward.

    ``outcomes-v1`` and ``hit-miss-terminal-v1`` exactly preserve the v0.3
    environment.  Alternative values are deliberately opt-in so benchmark
    runs cannot silently change their learning objective or observation
    shape.
    """

    observation_profile: Literal[
        "outcomes-v1", "outcomes-plus-available-v1"
    ] = "outcomes-v1"
    reward_profile: Literal["hit-miss-terminal-v1", "exploration-v1"] = (
        "hit-miss-terminal-v1"
    )

    def __post_init__(self) -> None:
        if self.observation_profile not in {
            "outcomes-v1",
            "outcomes-plus-available-v1",
        }:
            raise ValueError("unsupported attack observation profile")
        if self.reward_profile not in {"hit-miss-terminal-v1", "exploration-v1"}:
            raise ValueError("unsupported attack reward profile")

    @property
    def observation_channels(self) -> int:
        """Return the public number of observation planes."""
        return 4 if self.observation_profile == "outcomes-v1" else 5

    def public_dict(self) -> dict[str, str]:
        """Return JSON-native provenance suitable for model metadata."""
        return {
            "observation_profile": self.observation_profile,
            "reward_profile": self.reward_profile,
        }

    @classmethod
    def from_public_dict(cls, values: dict[str, Any]) -> "AttackEnvironmentConfig":
        """Restore a configuration persisted in training metadata."""
        return cls(
            observation_profile=str(values.get("observation_profile", "outcomes-v1")),
            reward_profile=str(values.get("reward_profile", "hit-miss-terminal-v1")),
        )


class AttackEnv(gym.Env[np.ndarray, int]):
    """Learn which unknown cell to shoot against a random legal fleet.

    The public observation intentionally contains only scenario geometry and
    outcomes of prior shots.  Fleet placement remains internal state throughout
    an episode, including after a terminal transition.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: Topology,
        *,
        config: AttackEnvironmentConfig | None = None,
        fleet_factory: FleetFactory | None = None,
    ) -> None:
        """Create an attack environment for a fixed 10 by 18 topology."""
        super().__init__()
        if topology.action_count != 180:
            raise ValueError("attack environments require a 10x18 action canvas")

        self.topology = topology
        self.config = AttackEnvironmentConfig() if config is None else config
        self._fleet_factory = (
            sample_random_legal_fleet if fleet_factory is None else fleet_factory
        )
        self.action_space = spaces.Discrete(topology.action_count)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.config.observation_channels, topology.rows, topology.columns),
            dtype=np.uint8,
        )
        self.max_total_attempts = 2 * topology.valid_cell_count

        self._valid_cells = frozenset(topology.valid_cells)
        self._valid_mask = np.zeros(topology.action_count, dtype=np.bool_)
        self._valid_mask[list(self._valid_cells)] = True
        self._called_cells: set[int] = set()
        self._hit_cells: set[int] = set()
        self._missed_cells: set[int] = set()
        self._sunk_ship_ids: set[str] = set()
        self._fleet: Fleet | None = None
        self._ship_id_by_cell: dict[int, str] = {}
        self._episode_id = 0
        self._valid_shots = 0
        self._invalid_attempts = 0
        self._total_attempts = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, int | bool]]:
        """Sample a hidden ``random_legal-v1`` fleet and clear shot history."""
        del options
        super().reset(seed=seed)
        assert self.np_random is not None

        self._fleet = self._fleet_factory(self.topology, self.np_random)
        if not isinstance(self._fleet, Fleet):
            raise TypeError("fleet_factory must return a Fleet")
        if not is_legal_fleet(self.topology, self._fleet):
            raise ValueError("fleet_factory must return a legal fleet for the topology")
        self._ship_id_by_cell = {
            cell: placement.ship_id
            for placement in self._fleet.placements
            for cell in placement.cells
        }
        self._called_cells.clear()
        self._hit_cells.clear()
        self._missed_cells.clear()
        self._sunk_ship_ids.clear()
        self._valid_shots = 0
        self._invalid_attempts = 0
        self._total_attempts = 0
        # A supplied seed is a stable public episode identifier.  This keeps
        # all public reset/step output deterministic for Gymnasium's seeded
        # environment contract, while unseeded episodes still receive a new ID.
        self._episode_id = int(seed) if seed is not None else self._episode_id + 1
        return self._observation(), self._info(is_hit=False, sunk_ship_length=0)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, int | bool]]:
        """Resolve one shot, treating a masked-out action as a penalised no-op."""
        if self._fleet is None:
            raise RuntimeError("reset() must be called before step()")

        self._total_attempts += 1
        action_value = int(action) if self.action_space.contains(action) else -1
        if action_value not in self._valid_cells or action_value in self._called_cells:
            self._invalid_attempts += 1
            truncated = self._total_attempts >= self.max_total_attempts
            return (
                self._observation(),
                -1.0,
                False,
                truncated,
                self._info(is_hit=False, sunk_ship_length=0),
            )

        self._called_cells.add(action_value)
        self._valid_shots += 1
        is_hit = action_value in self._ship_id_by_cell
        sunk_ship_length = 0
        reward = self._shot_reward(is_hit)
        if is_hit:
            self._hit_cells.add(action_value)
            ship_id = self._ship_id_by_cell[action_value]
            placement = self._fleet.placement_for(ship_id)
            if set(placement.cells).issubset(self._hit_cells):
                self._sunk_ship_ids.add(ship_id)
                sunk_ship_length = placement.length
        else:
            self._missed_cells.add(action_value)

        terminated = self._hit_cells == self._fleet.occupied_cells
        if terminated:
            reward += float(self._fleet.segment_count)
        truncated = not terminated and self._total_attempts >= self.max_total_attempts
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(is_hit=is_hit, sunk_ship_length=sunk_ship_length),
        )

    def action_masks(self) -> np.ndarray:
        """Return the valid, not-yet-called shots as a boolean action mask."""
        mask = self._valid_mask.copy()
        if self._called_cells:
            mask[list(self._called_cells)] = False
        return mask

    def _observation(self) -> np.ndarray:
        observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        observation[0] = self._valid_mask.reshape(self.topology.rows, self.topology.columns)
        if self._fleet is None:
            return observation

        sunk_cells = {
            cell
            for placement in self._fleet.placements
            if placement.ship_id in self._sunk_ship_ids
            for cell in placement.cells
        }
        active_hits = self._hit_cells - sunk_cells
        for channel, cells in ((1, active_hits), (2, sunk_cells), (3, self._missed_cells)):
            if cells:
                rows, columns = zip(*(self.topology.coordinate_for(cell) for cell in cells))
                observation[channel, rows, columns] = 1
        if self.config.observation_profile == "outcomes-plus-available-v1":
            observation[4] = self.action_masks().reshape(
                self.topology.rows, self.topology.columns
            )
        return observation

    def _shot_reward(self, is_hit: bool) -> float:
        """Return the configured immediate reward without exposing fleet state."""
        if self.config.reward_profile == "hit-miss-terminal-v1":
            return 1.0 if is_hit else -1.0
        # The alternative keeps a hit's reward unchanged while reducing the
        # miss penalty.  Its explicit hypothesis is that PPO will explore
        # enough to exploit the local information revealed by a hit.
        return 1.0 if is_hit else -0.2

    def _info(self, *, is_hit: bool, sunk_ship_length: int) -> dict[str, int | bool]:
        return {
            "is_hit": is_hit,
            "sunk_ship_length": sunk_ship_length,
            "valid_shots": self._valid_shots,
            "invalid_attempts": self._invalid_attempts,
            "episode_id": self._episode_id,
        }
