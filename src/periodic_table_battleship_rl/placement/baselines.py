"""Seeded public-state baselines for defensive fleet placement.

The policies in this module receive only the public ``PlacementEnv``
observation and its legal-action mask.  In particular, they do not inspect the
environment's partial fleet or the attacker's hidden state.  An evaluator owns
the episode boundary and must call :meth:`PlacementBaseline.reset` once per
episode to make stochastic tie breaks reproducible.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from periodic_table_battleship_rl.game import CANONICAL_FLEET, Orientation
from periodic_table_battleship_rl.topology import Topology


@runtime_checkable
class PlacementBaseline(Protocol):
    """Small public API shared by non-learned placement policies."""

    policy_id: str

    def reset(self, *, seed: int | None = None) -> None:
        """Start a reproducible policy episode, optionally with a new seed."""

    def select_action(
        self,
        observation: Any,
        action_mask: Any,
        *,
        deterministic: bool = True,
    ) -> int:
        """Choose one action that is true in the supplied legal-action mask."""


class _SeededPlacementBaseline:
    """Common explicit-RNG and tie-break behavior for placement baselines."""

    policy_id: str

    def __init__(self, topology: Topology, *, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.topology = topology
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset tie breaks to a portable NumPy generator state."""

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer or None")
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def _break_tie(self, actions: np.ndarray) -> int:
        if actions.size == 0:
            raise ValueError("action_mask contains no legal placement action")
        return int(actions[int(self._rng.integers(actions.size))])


class RandomLegalPlacementPolicy(_SeededPlacementBaseline):
    """Sample uniformly from the legal placement actions at every ship step."""

    policy_id = "random-legal-placement-v1"

    def select_action(
        self,
        observation: Any,
        action_mask: Any,
        *,
        deterministic: bool = True,
    ) -> int:
        """Return one uniformly sampled legal action.

        ``observation`` and ``deterministic`` are accepted to share the future
        evaluator interface with learned placement policies.  Randomness is
        controlled exclusively by :meth:`reset` and the constructor seed.
        """

        del observation, deterministic
        return self._break_tie(_legal_actions(action_mask))


class DispersionPlacementPolicy(_SeededPlacementBaseline):
    """Place each next ship as far as possible from public occupied cells.

    Candidate scores sum every candidate segment's Manhattan distance to its
    nearest already occupied segment.  The first ship has no prior segments,
    so it uses the seeded tie break.  This deliberately simple rule is a
    reproducible, non-learned baseline rather than an estimate of attack value.
    """

    policy_id = "dispersion-placement-v1"

    def select_action(
        self,
        observation: Any,
        action_mask: Any,
        *,
        deterministic: bool = True,
    ) -> int:
        """Choose a legal action with maximum public geometric dispersion."""

        del deterministic
        legal_actions = _legal_actions(action_mask)
        occupied = _occupied_cells_from_observation(observation, self.topology)
        if not occupied:
            return self._break_tie(legal_actions)

        length = _next_ship_length(occupied)
        scores = {
            action: _dispersion_score(self.topology, action, length, occupied)
            for action in legal_actions
        }
        return self._break_tie(_best_actions(scores))


class HuntTargetResistantPlacementPolicy(_SeededPlacementBaseline):
    """Reduce cross-ship targets and retain empty neighbour decoys.

    Hunt-target attackers inspect neighbours of unsunk hits.  This policy first
    minimizes direct adjacency to previously placed ships, then maximizes
    unoccupied neighbouring cells around the candidate, then applies the same
    distance score as :class:`DispersionPlacementPolicy`.  All inputs are
    derived from the public observation, legal mask, and fixed topology.
    """

    policy_id = "hunt-target-resistant-placement-v1"

    def select_action(
        self,
        observation: Any,
        action_mask: Any,
        *,
        deterministic: bool = True,
    ) -> int:
        """Choose a legal placement tailored to the hunt-target baseline."""

        del deterministic
        legal_actions = _legal_actions(action_mask)
        occupied = _occupied_cells_from_observation(observation, self.topology)
        length = _next_ship_length(occupied)
        scores = {
            action: _hunt_target_score(self.topology, action, length, occupied)
            for action in legal_actions
        }
        return self._break_tie(_best_actions(scores))


def _legal_actions(action_mask: Any) -> np.ndarray:
    """Validate and return the current legal placement actions in index order."""

    mask = np.asarray(action_mask)
    if mask.shape != (360,):
        raise ValueError("action_mask must have shape (360,)")
    if mask.dtype != np.bool_:
        raise TypeError("action_mask must have boolean dtype")
    actions = np.flatnonzero(mask)
    if actions.size == 0:
        raise ValueError("action_mask contains no legal placement action")
    return actions


def _occupied_cells_from_observation(observation: Any, topology: Topology) -> frozenset[int]:
    """Recover only the public occupied-cell channel from an observation."""

    array = np.asarray(observation)
    expected_shape = (3, topology.rows, topology.columns)
    if array.shape != expected_shape:
        raise ValueError(f"observation must have shape {expected_shape}")
    return frozenset(
        action
        for action in topology.valid_cells
        if array[(1, *topology.coordinate_for(action))] > 0.5
    )


def _next_ship_length(occupied: frozenset[int]) -> int:
    """Infer the canonical next ship solely from public occupied segment count."""

    placed_segments = len(occupied)
    total = 0
    for ship in CANONICAL_FLEET:
        if total == placed_segments:
            return ship.length
        total += ship.length
    raise ValueError("observation does not contain a valid partial canonical fleet")


def _candidate_cells(topology: Topology, action: int, length: int) -> tuple[int, ...]:
    """Decode one legal action into its fixed-topology segment cells."""

    anchor = action % 180
    orientation = Orientation.HORIZONTAL if action < 180 else Orientation.VERTICAL
    cells = topology.segment_from(anchor, orientation, length)
    if cells is None:
        raise RuntimeError("legal action mask contains an invalid placement action")
    return cells


def _dispersion_score(
    topology: Topology, action: int, length: int, occupied: frozenset[int]
) -> int:
    """Return a deterministic integer score where larger means more separation."""

    cells = _candidate_cells(topology, action, length)
    return sum(
        min(_manhattan_distance(topology, candidate, previous) for previous in occupied)
        for candidate in cells
    )


def _hunt_target_score(
    topology: Topology, action: int, length: int, occupied: frozenset[int]
) -> tuple[int, int, int]:
    """Return lexicographic anti-hunt-target scores where larger is preferable."""

    cells = frozenset(_candidate_cells(topology, action, length))
    cross_ship_adjacencies = sum(
        neighbor in occupied for cell in cells for neighbor in topology.neighbors(cell)
    )
    empty_neighbours = sum(
        neighbor not in cells and neighbor not in occupied
        for cell in cells
        for neighbor in topology.neighbors(cell)
    )
    dispersion = (
        _dispersion_score(topology, action, length, occupied) if occupied else 0
    )
    return (-cross_ship_adjacencies, empty_neighbours, dispersion)


def _manhattan_distance(topology: Topology, left: int, right: int) -> int:
    """Compute canvas Manhattan distance without reading any environment state."""

    left_row, left_column = topology.coordinate_for(left)
    right_row, right_column = topology.coordinate_for(right)
    return abs(left_row - right_row) + abs(left_column - right_column)


def _best_actions(scores: dict[int, int] | dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Return score-maximizing actions in stable action-index order."""

    best_score = max(scores.values())
    return np.fromiter(
        (action for action, score in scores.items() if score == best_score), dtype=np.int64
    )
