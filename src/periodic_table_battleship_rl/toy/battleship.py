"""Finite masked Battleship environment for Q-learning and SARSA.

This module is deliberately separate from the 10 by 18 benchmark environments.
It models a 4 by 4 board with one hidden, randomly placed target.  Its complete
public state has a finite base-three encoding, which makes it useful for
checking tabular update implementations before training neural agents.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TinyBattleshipEnv(gym.Env[int, int]):
    """A seedable 4 by 4 masked target-search environment.

    Each episode samples one hidden target uniformly from the 16 legal cells.
    A state digit is ``0`` for an untried cell, ``1`` for a miss, and ``2`` for
    the target hit.  The digit for action ``a`` has weight ``3**a`` in the
    integer observation.  This gives Q-learning and SARSA a stable finite
    state index without exposing the hidden target before it is hit.
    """

    metadata = {"render_modes": []}

    BOARD_SIZE: ClassVar[int] = 4
    CELL_COUNT: ClassVar[int] = BOARD_SIZE * BOARD_SIZE
    STATE_COUNT: ClassVar[int] = 3**CELL_COUNT
    HIT_REWARD: ClassVar[float] = 1.0
    MISS_REWARD: ClassVar[float] = -0.1
    INVALID_ACTION_REWARD: ClassVar[float] = -1.0

    def __init__(self) -> None:
        """Create the fixed, finite environment used by tabular algorithms."""
        super().__init__()
        self.action_space = spaces.Discrete(self.CELL_COUNT)
        self.observation_space = spaces.Discrete(self.STATE_COUNT)
        self.max_total_attempts = self.CELL_COUNT

        self._cell_states = np.zeros(self.CELL_COUNT, dtype=np.uint8)
        self._target_action: int | None = None
        self._total_attempts = 0
        self._episode_id = 0

    @property
    def state_index(self) -> int:
        """Return the current finite tabular-state index."""
        return self.encode_cell_states(self._cell_states)

    @classmethod
    def coordinate_for(cls, action: int) -> tuple[int, int]:
        """Map a row-major action index to its board coordinate."""
        if action not in range(cls.CELL_COUNT):
            raise ValueError(f"action must be in [0, {cls.CELL_COUNT}), got {action}")
        return divmod(action, cls.BOARD_SIZE)

    @classmethod
    def action_for(cls, row: int, column: int) -> int:
        """Map a board coordinate to its row-major action index."""
        if not 0 <= row < cls.BOARD_SIZE or not 0 <= column < cls.BOARD_SIZE:
            raise ValueError("row and column must be valid tiny-board coordinates")
        return row * cls.BOARD_SIZE + column

    @classmethod
    def encode_cell_states(cls, cell_states: np.ndarray) -> int:
        """Encode ``CELL_COUNT`` ternary cell states into a discrete observation."""
        states = np.asarray(cell_states, dtype=np.uint8).reshape(-1)
        if states.shape != (cls.CELL_COUNT,):
            raise ValueError(f"expected {cls.CELL_COUNT} cell states")
        if np.any(states > 2):
            raise ValueError("cell states must use only digits 0, 1, and 2")
        powers = np.power(3, np.arange(cls.CELL_COUNT, dtype=np.int64))
        return int(np.dot(states, powers))

    @classmethod
    def decode_state_index(cls, state_index: int) -> np.ndarray:
        """Decode a tabular observation into row-major ternary cell states."""
        if state_index not in range(cls.STATE_COUNT):
            raise ValueError(f"state index must be in [0, {cls.STATE_COUNT})")
        remaining = state_index
        states = np.zeros(cls.CELL_COUNT, dtype=np.uint8)
        for action in range(cls.CELL_COUNT):
            states[action] = remaining % 3
            remaining //= 3
        return states.reshape(cls.BOARD_SIZE, cls.BOARD_SIZE)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, int]]:
        """Sample a hidden target uniformly and return the all-unknown state."""
        del options
        super().reset(seed=seed)
        assert self.np_random is not None

        self._cell_states.fill(0)
        self._target_action = int(self.np_random.integers(self.CELL_COUNT))
        self._total_attempts = 0
        self._episode_id = int(seed) if seed is not None else self._episode_id + 1
        return self.state_index, self._info(is_hit=False)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, int]]:
        """Resolve one shot; masked-out actions are penalised no-ops."""
        if self._target_action is None:
            raise RuntimeError("reset() must be called before step()")

        self._total_attempts += 1
        action_value = int(action) if self.action_space.contains(action) else -1
        if action_value < 0 or self._cell_states[action_value] != 0:
            return (
                self.state_index,
                self.INVALID_ACTION_REWARD,
                False,
                self._total_attempts >= self.max_total_attempts,
                self._info(is_hit=False),
            )

        is_hit = action_value == self._target_action
        self._cell_states[action_value] = 2 if is_hit else 1
        terminated = is_hit
        truncated = not terminated and self._total_attempts >= self.max_total_attempts
        return (
            self.state_index,
            self.HIT_REWARD if is_hit else self.MISS_REWARD,
            terminated,
            truncated,
            self._info(is_hit=is_hit),
        )

    def action_masks(self) -> np.ndarray:
        """Return the currently untried actions as a boolean mask."""
        return self._cell_states == 0

    def _info(self, *, is_hit: bool) -> dict[str, int]:
        return {
            "is_hit": int(is_hit),
            "total_attempts": self._total_attempts,
            "episode_id": self._episode_id,
        }
