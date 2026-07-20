"""Pure, masked attack baselines.

The policies receive only public state: an action mask and, for hunt-target,
the currently active hits.  They never receive fleet placements or any other
hidden-state representation.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from periodic_table_battleship_rl.topology import Topology


class NoValidActionError(ValueError):
    """Raised when a policy is asked to act with no legal action."""


def random_masked_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    """Choose one currently valid action uniformly with the supplied RNG.

    ``action_mask`` must be a one-dimensional boolean NumPy array.  Candidate
    actions are collected in increasing action-index order; randomness and all
    tie-breaking therefore come exclusively from ``rng``.
    """

    valid_actions = _valid_actions(action_mask)
    return int(valid_actions[int(rng.integers(len(valid_actions)))])


def hunt_target_action(
    topology: Topology,
    action_mask: np.ndarray,
    active_hits: Collection[int],
    rng: np.random.Generator,
) -> int:
    """Choose a valid neighbour of an active hit, otherwise a random action.

    The policy considers only orthogonal neighbours provided by ``topology``.
    It never infers occupancy from a non-hit cell.  Candidate neighbours are
    deduplicated and sorted by action index before selection, so ties are
    deterministically resolved by the supplied ``rng``.  If no active hit has
    a valid neighbour, the policy falls back to :func:`random_masked_action`.
    """

    valid_actions = _valid_actions(action_mask, action_count=topology.action_count)
    valid_action_set = frozenset(valid_actions.tolist())
    target_actions = sorted(
        {
            neighbor
            for hit in active_hits
            if topology.is_valid_action(hit)
            for neighbor in topology.neighbors(hit)
            if neighbor in valid_action_set
        }
    )
    if target_actions:
        return int(target_actions[int(rng.integers(len(target_actions)))])

    return int(valid_actions[int(rng.integers(len(valid_actions)))])


def _valid_actions(
    action_mask: np.ndarray, *, action_count: int | None = None
) -> np.ndarray:
    """Validate a boolean mask and return its true indices in stable order."""

    if not isinstance(action_mask, np.ndarray):
        raise TypeError("action_mask must be a NumPy array")
    if action_mask.ndim != 1:
        raise ValueError("action_mask must be one-dimensional")
    if action_mask.dtype != np.bool_:
        raise TypeError("action_mask must have boolean dtype")
    if action_count is not None and action_mask.size != action_count:
        raise ValueError(
            f"action_mask must contain {action_count} entries, got {action_mask.size}"
        )

    valid_actions = np.flatnonzero(action_mask)
    if not len(valid_actions):
        raise NoValidActionError("action_mask contains no valid action")
    return valid_actions
