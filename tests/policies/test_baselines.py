"""Tests for public-state attack baselines."""

from __future__ import annotations

import numpy as np
import pytest

from periodic_table_battleship_rl.policies import (
    NoValidActionError,
    hunt_target_action,
    random_masked_action,
)
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP


def test_random_masked_action_only_chooses_a_true_mask_entry() -> None:
    mask = np.zeros(BATTLESHIP.action_count, dtype=bool)
    allowed = np.array([0, 53, 162])
    mask[allowed] = True

    actions = {
        random_masked_action(mask, np.random.default_rng(seed)) for seed in range(50)
    }

    assert actions <= set(allowed)
    assert actions == set(allowed)


def test_random_masked_action_is_reproducible_for_the_same_seed() -> None:
    mask = np.zeros(BATTLESHIP.action_count, dtype=bool)
    mask[[1, 4, 50, 99]] = True

    first_rng = np.random.default_rng(83)
    second_rng = np.random.default_rng(83)
    first = [random_masked_action(mask, first_rng) for _ in range(8)]
    second = [random_masked_action(mask, second_rng) for _ in range(8)]

    assert first == second


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros(3, dtype=bool),
        np.array([[True, False]], dtype=bool),
        np.array([1, 0], dtype=np.int64),
    ],
)
def test_random_masked_action_rejects_empty_or_malformed_masks(mask: np.ndarray) -> None:
    with pytest.raises((NoValidActionError, TypeError, ValueError)):
        random_masked_action(mask, np.random.default_rng(1))


def test_hunt_target_prioritizes_an_unshot_orthogonal_neighbour() -> None:
    hit = BATTLESHIP.action_for(4, 4)
    neighbours = BATTLESHIP.neighbors(hit)
    mask = np.zeros(BATTLESHIP.action_count, dtype=bool)
    mask[list(neighbours)] = True
    mask[BATTLESHIP.action_for(0, 0)] = True

    action = hunt_target_action(BATTLESHIP, mask, {hit}, np.random.default_rng(9))

    assert action in neighbours


def test_hunt_target_uses_union_of_neighbours_without_duplicate_bias() -> None:
    left_hit = BATTLESHIP.action_for(4, 4)
    right_hit = BATTLESHIP.action_for(4, 6)
    shared_target = BATTLESHIP.action_for(4, 5)
    other_target = BATTLESHIP.action_for(3, 4)
    mask = np.zeros(BATTLESHIP.action_count, dtype=bool)
    mask[[shared_target, other_target]] = True

    seed = 17
    expected_index = int(np.random.default_rng(seed).integers(2))
    expected = sorted([shared_target, other_target])[expected_index]
    actual = hunt_target_action(
        BATTLESHIP,
        mask,
        {right_hit, left_hit},
        np.random.default_rng(seed),
    )

    assert actual == expected


def test_hunt_target_ignores_gaps_and_uses_only_valid_periodic_neighbours() -> None:
    beryllium = PERIODIC_TABLE_BATTLESHIP.action_for(1, 1)
    allowed_neighbour = PERIODIC_TABLE_BATTLESHIP.action_for(2, 1)
    non_neighbour_across_gap = PERIODIC_TABLE_BATTLESHIP.action_for(1, 12)
    mask = np.zeros(PERIODIC_TABLE_BATTLESHIP.action_count, dtype=bool)
    mask[[allowed_neighbour, non_neighbour_across_gap]] = True

    action = hunt_target_action(
        PERIODIC_TABLE_BATTLESHIP,
        mask,
        {beryllium},
        np.random.default_rng(3),
    )

    assert action == allowed_neighbour


def test_hunt_target_falls_back_to_a_random_valid_action() -> None:
    hit = BATTLESHIP.action_for(4, 4)
    fallback_actions = [BATTLESHIP.action_for(0, 0), BATTLESHIP.action_for(9, 9)]
    mask = np.zeros(BATTLESHIP.action_count, dtype=bool)
    mask[fallback_actions] = True

    seed = 5
    expected = sorted(fallback_actions)[int(np.random.default_rng(seed).integers(2))]
    action = hunt_target_action(BATTLESHIP, mask, {hit}, np.random.default_rng(seed))

    assert action == expected


def test_hunt_target_rejects_a_mask_with_wrong_canvas_size() -> None:
    with pytest.raises(ValueError, match="180"):
        hunt_target_action(
            BATTLESHIP,
            np.ones(BATTLESHIP.action_count - 1, dtype=bool),
            set(),
            np.random.default_rng(1),
        )


def test_hunt_target_does_not_require_active_hits_to_be_valid_actions() -> None:
    mask = np.zeros(BATTLESHIP.action_count, dtype=bool)
    allowed = BATTLESHIP.action_for(2, 2)
    mask[allowed] = True

    action = hunt_target_action(BATTLESHIP, mask, {999, -1}, np.random.default_rng(4))

    assert action == allowed
