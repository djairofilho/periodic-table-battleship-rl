"""Gymnasium microenvironment with exactly the oracle's 3 by 3 rules."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from periodic_table_battleship_rl.oracle import (
    BeliefState,
    MicroBoardConfig,
    MicroFleet,
    enumerate_fleets,
)


class MicroBattleshipEnv(gym.Env[int, int]):
    """A sampled episode of the oracle's public-belief microgame.

    The hidden fleet is uniform over the same legal fleet list as the exact
    oracle. The integer observation encodes only public outcomes: 0 unknown,
    1 miss and 2 hit. It is thus a lossless encoding of the oracle state.
    """

    metadata = {"render_modes": []}
    MISS_REWARD = -1.0
    WIN_REWARD = -1.0

    def __init__(self, config: MicroBoardConfig = MicroBoardConfig()) -> None:
        super().__init__()
        self.config = config
        self.fleets = enumerate_fleets(config)
        self.action_space = spaces.Discrete(config.cell_count)
        self.observation_space = spaces.Discrete(3**config.cell_count)
        self._cell_states = np.zeros(config.cell_count, dtype=np.uint8)
        self._fleet: MicroFleet | None = None
        self._shots = 0

    @property
    def state_index(self) -> int:
        """Return the finite public ternary state used by tabular methods."""

        return self.encode_cell_states(self._cell_states)

    @property
    def belief_state(self) -> BeliefState:
        """Return the oracle-compatible public state for this observation."""

        tried_mask = self._tried_mask
        hit_mask = sum(
            1 << action for action, state in enumerate(self._cell_states) if state == 2
        )
        candidate_ids = tuple(
            index
            for index, fleet in enumerate(self.fleets)
            if fleet.occupied_mask & tried_mask == hit_mask
        )
        if not candidate_ids:
            raise RuntimeError("public observation is inconsistent with the legal fleet set")
        return BeliefState(candidate_ids, tried_mask, hit_mask)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Sample one uniform hidden fleet and return the all-unknown state."""

        del options
        super().reset(seed=seed)
        assert self.np_random is not None
        self._cell_states.fill(0)
        self._fleet = self.fleets[int(self.np_random.integers(len(self.fleets)))]
        self._shots = 0
        return self.state_index, self._info(is_hit=False)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Fire one legal shot; repeated and invalid actions are rejected."""

        if self._fleet is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action) or not self.action_masks()[int(action)]:
            raise ValueError("action must be an untried microboard cell")

        action = int(action)
        self._shots += 1
        is_hit = bool(self._fleet.occupied_mask & (1 << action))
        self._cell_states[action] = 2 if is_hit else 1
        terminated = self._fleet.occupied_mask & ~self._tried_mask == 0
        return self.state_index, -1.0, terminated, False, self._info(is_hit=is_hit)

    def action_masks(self) -> np.ndarray:
        """Return the mask of still-untried public actions."""

        return self._cell_states == 0

    def encode_cell_states(self, cell_states: np.ndarray) -> int:
        """Encode a public 0/1/2 outcome vector into one integer observation."""

        states = np.asarray(cell_states, dtype=np.uint8).reshape(-1)
        if states.shape != (self.config.cell_count,):
            raise ValueError(f"expected {self.config.cell_count} cells")
        if np.any(states > 2):
            raise ValueError("cell states must use only digits 0, 1, and 2")
        powers = np.power(3, np.arange(self.config.cell_count, dtype=np.int64))
        return int(np.dot(states, powers))

    def state_index_for_belief(self, state: BeliefState) -> int:
        """Encode an oracle state with the environment observation code."""

        states = np.zeros(self.config.cell_count, dtype=np.uint8)
        for action in range(self.config.cell_count):
            bit = 1 << action
            if state.hit_mask & bit:
                states[action] = 2
            elif state.tried_mask & bit:
                states[action] = 1
        return self.encode_cell_states(states)

    @property
    def _tried_mask(self) -> int:
        return sum(1 << action for action, state in enumerate(self._cell_states) if state)

    def _info(self, *, is_hit: bool) -> dict[str, Any]:
        return {"is_hit": is_hit, "shots": self._shots, "belief": self.belief_state}
