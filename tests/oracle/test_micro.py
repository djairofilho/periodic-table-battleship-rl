"""Tests for the exact public-belief microboard oracle."""

from __future__ import annotations

import pytest

from periodic_table_battleship_rl.oracle import (
    BeliefState,
    ExactBattleshipOracle,
    MicroBoardConfig,
    enumerate_fleets,
    evaluate_baselines,
)
from periodic_table_battleship_rl.oracle.micro import ShotOutcome


def test_enumerates_all_unique_length_two_fleets_on_three_by_three_board() -> None:
    fleets = enumerate_fleets(MicroBoardConfig())

    assert len(fleets) == 12
    assert len({fleet.occupied_mask for fleet in fleets}) == 12
    assert all(fleet.occupied_mask.bit_count() == 2 for fleet in fleets)


def test_equal_length_ship_labels_do_not_duplicate_physical_layouts() -> None:
    config = MicroBoardConfig(rows=3, columns=3, ship_lengths=(1, 1))
    fleets = enumerate_fleets(config)

    assert len(fleets) == 36
    assert len({fleet.occupied_mask for fleet in fleets}) == 36


def test_initial_belief_has_exact_posterior_occupancy_probabilities() -> None:
    oracle = ExactBattleshipOracle()
    probabilities = oracle.occupancy_probabilities(oracle.initial_state)

    assert probabilities.tolist() == pytest.approx(
        [1 / 6, 1 / 4, 1 / 6, 1 / 4, 1 / 3, 1 / 4, 1 / 6, 1 / 4, 1 / 6]
    )


def test_transition_partitions_only_by_public_outcome() -> None:
    oracle = ExactBattleshipOracle()
    branches = oracle.transitions(oracle.initial_state, action=4)

    assert [(outcome, probability) for outcome, probability, _ in branches] == [
        (ShotOutcome.MISS, pytest.approx(2 / 3)),
        (ShotOutcome.HIT, pytest.approx(1 / 3)),
    ]
    hit_state = next(state for outcome, _, state in branches if outcome is ShotOutcome.HIT)
    assert hit_state is not None
    assert hit_state.tried_mask == 1 << 4
    assert hit_state.hit_mask == 1 << 4
    assert len(hit_state.candidate_ids) == 4


def test_memoized_oracle_solves_default_microboard_exactly() -> None:
    oracle = ExactBattleshipOracle()
    solution = oracle.solve()

    assert solution.expected_shots == pytest.approx(4.5)
    assert solution.optimal_actions == (1, 3, 4, 5, 7)
    assert solution.solved_states > 100
    assert set(solution.action_values) == set(range(9))


def test_baselines_have_exact_non_negative_regret_against_oracle() -> None:
    comparison = evaluate_baselines()
    results = {result.name: result for result in comparison.baselines}

    assert results["random-masked"].expected_shots == pytest.approx(20 / 3)
    assert results["hunt-target"].expected_shots == pytest.approx(89 / 18)
    assert results["posterior-greedy"].expected_shots == pytest.approx(4.5)
    assert all(result.regret_vs_oracle >= 0 for result in comparison.baselines)


def test_oracle_rejects_non_public_invalid_actions_and_bad_policy_mass() -> None:
    oracle = ExactBattleshipOracle()
    state = BeliefState(tuple(range(len(oracle.fleets))), tried_mask=1)

    with pytest.raises(ValueError, match="untried"):
        oracle.transitions(state, action=0)

    with pytest.raises(ValueError, match="invalid action"):
        oracle.evaluate_policy(lambda _state, _oracle: {99: 1.0})
