"""Tests for explainable action selection over a finite belief."""

from __future__ import annotations

import numpy as np

from periodic_table_battleship_rl.belief import (
    BeliefPopulation,
    PublicAttackState,
    information_action,
    information_gain,
    probability_action,
    short_horizon_action,
)
from periodic_table_battleship_rl.game import Fleet, ShipPlacement
from periodic_table_battleship_rl.topology import BATTLESHIP


def _belief() -> BeliefPopulation:
    first = Fleet((ShipPlacement("ship", 2, 0, "horizontal", (0, 1)),))
    second = Fleet((ShipPlacement("ship", 2, 0, "vertical", (0, 18)),))
    return BeliefPopulation(
        PublicAttackState(BATTLESHIP, frozenset(), frozenset()),
        (first, second),
        "test",
        True,
    )


def test_probability_and_information_have_distinct_explainable_objectives() -> None:
    mask = np.zeros(BATTLESHIP.action_count, dtype=np.bool_)
    mask[[0, 1]] = True
    belief = _belief()

    assert probability_action(belief, mask) == 0
    assert information_action(belief, mask) == 1
    assert short_horizon_action(belief, mask) == 0
    assert information_gain(0.5) > information_gain(1.0)


def test_information_gain_handles_deterministic_outcomes() -> None:
    assert information_gain(0.0) == 0.0
    assert information_gain(1.0) == 0.0
