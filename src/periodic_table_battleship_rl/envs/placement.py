"""Gymnasium environment for the defensive fleet-placement experiment.

An episode contains exactly one decision for every ship in the canonical
fleet.  Once the fleet is complete, an injected :class:`PlacementEvaluator`
measures how many valid shots an attacker needs to sink it.  Keeping the
attacker behind this small protocol makes the environment usable before the
attack policies are implemented and keeps policy selection outside the state.
"""

from __future__ import annotations

from numbers import Integral
from typing import Protocol, runtime_checkable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from periodic_table_battleship_rl.game import (
    CANONICAL_FLEET,
    Fleet,
    Orientation,
    ShipPlacement,
    is_legal_fleet,
)
from periodic_table_battleship_rl.topology import Topology, get_topology


PLACEMENT_ACTION_COUNT = 360
"""Two orientations for every location in the fixed 10 by 18 canvas."""


@runtime_checkable
class PlacementEvaluator(Protocol):
    """Evaluate a completed fleet against one already-selected attacker.

    Implementations must return the number of *valid* shots needed to sink
    every segment in ``fleet``.  They may use ``rng`` for attacker tie breaks
    or deterministic episode-level randomness.  The environment validates the
    returned count against the number of playable cells in its topology.
    """

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        """Return valid shots required to sink ``fleet``."""


class PlacementEnv(gym.Env[np.ndarray, int]):
    """Place the canonical fleet, then receive a normalized survival reward.

    The observation is a ``float32`` array with shape ``(3, rows, columns)``:

    * channel 0 is one for valid topology cells and zero for gaps;
    * channel 1 is one for already occupied cells and zero elsewhere;
    * channel 2 contains the next ship length divided by the largest ship
      length at every valid cell (and zero in gaps).  It becomes zero once all
      ships have been placed.

    Actions ``0..179`` use a horizontal placement and ``180..359`` use a
    vertical placement.  The anchor is always ``action % 180``.  Invalid
    actions leave the partial fleet unchanged and receive reward ``-1``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: str | Topology,
        evaluator: PlacementEvaluator,
    ) -> None:
        super().__init__()
        self.topology = get_topology(topology) if isinstance(topology, str) else topology
        self.evaluator = evaluator
        if self.topology.action_count != PLACEMENT_ACTION_COUNT // 2:
            raise ValueError("placement actions require a 180-cell logical canvas")

        self.action_space = spaces.Discrete(PLACEMENT_ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.topology.rows, self.topology.columns),
            dtype=np.float32,
        )
        self._topology_channel = self._make_topology_channel()
        self._occupied: set[int] = set()
        self._placements: list[ShipPlacement] = []
        self._placement_actions: list[int] = []
        self._terminated = False
        self._last_valid_shots_to_sink: int | None = None

    @property
    def fleet(self) -> Fleet | None:
        """Return the completed fleet, or ``None`` while placement continues."""

        if len(self._placements) != len(CANONICAL_FLEET):
            return None
        return Fleet(tuple(self._placements))

    @property
    def placement_actions(self) -> tuple[int, ...]:
        """Actions accepted so far, in canonical ship order."""

        return tuple(self._placement_actions)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Start a new placement episode, resetting evaluator randomness too."""

        del options
        super().reset(seed=seed)
        self._occupied.clear()
        self._placements.clear()
        self._placement_actions.clear()
        self._terminated = False
        self._last_valid_shots_to_sink = None
        return self._observation(), self._info(invalid_action=False)

    def action_masks(self) -> np.ndarray:
        """Return exactly the actions that can place the current next ship."""

        mask = np.zeros(PLACEMENT_ACTION_COUNT, dtype=np.bool_)
        if self._terminated or len(self._placements) == len(CANONICAL_FLEET):
            return mask

        length = CANONICAL_FLEET[len(self._placements)].length
        for action in range(PLACEMENT_ACTION_COUNT):
            if self._cells_for_action(action, length) is not None:
                mask[action] = True
        return mask

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Accept one legal ship placement or penalize an invalid candidate."""

        if self._terminated:
            raise RuntimeError("episode has terminated; call reset() before step()")

        action_value = int(action) if isinstance(action, Integral) else -1
        if not 0 <= action_value < PLACEMENT_ACTION_COUNT:
            return self._observation(), -1.0, False, False, self._info(invalid_action=True)

        next_spec = CANONICAL_FLEET[len(self._placements)]
        cells = self._cells_for_action(action_value, next_spec.length)
        if cells is None:
            return self._observation(), -1.0, False, False, self._info(invalid_action=True)

        anchor, orientation = self._decode_action(action_value)
        placement = ShipPlacement(
            ship_id=next_spec.ship_id,
            length=next_spec.length,
            anchor=anchor,
            orientation=orientation,
            cells=cells,
        )
        self._placements.append(placement)
        self._placement_actions.append(action_value)
        self._occupied.update(cells)

        if len(self._placements) != len(CANONICAL_FLEET):
            return self._observation(), 0.0, False, False, self._info(invalid_action=False)

        fleet = Fleet(tuple(self._placements))
        if not is_legal_fleet(self.topology, fleet):
            raise RuntimeError("internal error: action mask produced an illegal fleet")
        valid_shots_to_sink = self.evaluator.evaluate(fleet, rng=self.np_random)
        self._last_valid_shots_to_sink = self._validate_evaluation(valid_shots_to_sink)
        self._terminated = True
        normalized_reward = self._last_valid_shots_to_sink / self.topology.valid_cell_count
        return self._observation(), normalized_reward, True, False, self._info(
            invalid_action=False
        )

    def _make_topology_channel(self) -> np.ndarray:
        channel = np.zeros((self.topology.rows, self.topology.columns), dtype=np.float32)
        for action in self.topology.valid_cells:
            row, column = self.topology.coordinate_for(action)
            channel[row, column] = 1.0
        return channel

    def _observation(self) -> np.ndarray:
        occupied_channel = np.zeros_like(self._topology_channel)
        for action in self._occupied:
            row, column = self.topology.coordinate_for(action)
            occupied_channel[row, column] = 1.0

        next_ship_channel = np.zeros_like(self._topology_channel)
        if not self._terminated and len(self._placements) < len(CANONICAL_FLEET):
            next_length = CANONICAL_FLEET[len(self._placements)].length
            next_ship_channel = self._topology_channel * (
                next_length / CANONICAL_FLEET[0].length
            )
        return np.stack(
            (self._topology_channel, occupied_channel, next_ship_channel), dtype=np.float32
        )

    def _cells_for_action(self, action: int, length: int) -> tuple[int, ...] | None:
        anchor, orientation = self._decode_action(action)
        cells = self.topology.segment_from(anchor, orientation, length)
        if cells is None or not self._occupied.isdisjoint(cells):
            return None
        return cells

    @staticmethod
    def _decode_action(action: int) -> tuple[int, Orientation]:
        if not 0 <= action < PLACEMENT_ACTION_COUNT:
            raise ValueError(f"placement action {action} outside [0, 360)")
        orientation = (
            Orientation.HORIZONTAL if action < PLACEMENT_ACTION_COUNT // 2 else Orientation.VERTICAL
        )
        return action % (PLACEMENT_ACTION_COUNT // 2), orientation

    def _validate_evaluation(self, valid_shots_to_sink: int) -> int:
        if isinstance(valid_shots_to_sink, bool) or not isinstance(
            valid_shots_to_sink, Integral
        ):
            raise TypeError("PlacementEvaluator.evaluate() must return an integer")
        shots = int(valid_shots_to_sink)
        if not 1 <= shots <= self.topology.valid_cell_count:
            raise ValueError(
                "PlacementEvaluator.evaluate() must return a valid-shot count "
                f"within [1, {self.topology.valid_cell_count}]"
            )
        return shots

    def _info(self, *, invalid_action: bool) -> dict[str, object]:
        return {
            "invalid_action": invalid_action,
            "placements_completed": len(self._placements),
            "placement_actions": self.placement_actions,
            "valid_shots_to_sink": self._last_valid_shots_to_sink,
        }
