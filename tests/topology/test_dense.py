"""Tests for the dense-118 cardinality-control topology."""

from __future__ import annotations

from collections import deque

import numpy as np

from periodic_table_battleship_rl.game.fleet import (
    CANONICAL_FLEET,
    is_legal_fleet,
    sample_random_legal_fleet,
)
from periodic_table_battleship_rl.topology.dense import DENSE_118, DENSE_118_SCENARIO


def _connected_component(start: int) -> set[int]:
    visited = {start}
    pending = deque([start])
    while pending:
        action = pending.popleft()
        for neighbor in DENSE_118.neighbors(action):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return visited


def test_dense_118_has_the_periodic_cardinality_on_the_standard_canvas():
    assert DENSE_118.name == "dense-118"
    assert (DENSE_118.rows, DENSE_118.columns) == (10, 18)
    assert DENSE_118.action_count == 180
    assert DENSE_118.valid_cell_count == 118
    assert DENSE_118.valid_actions == frozenset(range(118))
    assert DENSE_118_SCENARIO["purpose"] == "cardinality control for periodic-table-battleship"


def test_dense_118_has_no_internal_gaps_and_is_orthogonally_connected():
    for row in range(6):
        assert all(DENSE_118.is_valid_action(DENSE_118.action_for(row, column)) for column in range(18))
    assert all(DENSE_118.is_valid_action(DENSE_118.action_for(6, column)) for column in range(10))
    assert not DENSE_118.is_valid_action(DENSE_118.action_for(6, 10))

    assert _connected_component(0) == DENSE_118.valid_actions


def test_dense_118_segments_and_fleet_sampling_are_legal():
    assert DENSE_118.segment_from(0, "horizontal", 5) == (0, 1, 2, 3, 4)
    assert DENSE_118.segment_from(0, "vertical", 5) == (0, 18, 36, 54, 72)
    assert DENSE_118.segment_from(DENSE_118.action_for(6, 7), "horizontal", 4) is None

    fleet = sample_random_legal_fleet(DENSE_118, np.random.default_rng(20260720))
    assert is_legal_fleet(DENSE_118, fleet, CANONICAL_FLEET)
    assert fleet.segment_count == 17
