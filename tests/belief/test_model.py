"""Tests for public-state compatible fleet beliefs."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from periodic_table_battleship_rl.belief import (
    BeliefPopulation,
    CompatibleFleetLimitError,
    PublicAttackState,
    enumerate_compatible_fleets,
    exact_belief,
    sample_compatible_fleets,
)
from periodic_table_battleship_rl.game import Fleet, ShipPlacement, ShipSpec
from periodic_table_battleship_rl.topology import BATTLESHIP, Cell, Topology


def _small_topology() -> Topology:
    cells = {
        row * 3 + column: Cell(action=row * 3 + column, row=row, column=column)
        for row in range(3)
        for column in range(3)
    }
    neighbors = {}
    for action, cell in cells.items():
        adjacent = []
        for row, column in (
            (cell.row - 1, cell.column),
            (cell.row, cell.column - 1),
            (cell.row, cell.column + 1),
            (cell.row + 1, cell.column),
        ):
            candidate = row * 3 + column
            if 0 <= row < 3 and 0 <= column < 3 and candidate in cells:
                adjacent.append(candidate)
        neighbors[action] = tuple(sorted(adjacent))
    return Topology(
        name="belief-test-3x3",
        rows=3,
        columns=3,
        cells_by_action=MappingProxyType(cells),
        neighbors_by_action=MappingProxyType(neighbors),
    )


SPECS = (ShipSpec("alpha", 2), ShipSpec("beta", 2))


def test_exact_belief_contains_only_fleets_consistent_with_public_history() -> None:
    topology = _small_topology()
    state = PublicAttackState(
        topology=topology,
        hit_cells=frozenset({0}),
        missed_cells=frozenset({8}),
    )

    belief = exact_belief(state, SPECS, max_fleets=1_000)

    assert belief.exact
    assert belief.size > 0
    assert np.allclose(belief.occupancy_probabilities()[[0]], 1.0)
    for fleet in belief.fleets:
        assert 0 in fleet.occupied_cells
        assert 8 not in fleet.occupied_cells


def test_exact_enumeration_refuses_to_silently_explode() -> None:
    with pytest.raises(CompatibleFleetLimitError):
        enumerate_compatible_fleets(
            PublicAttackState(_small_topology(), frozenset(), frozenset()),
            SPECS,
            max_fleets=1,
        )


def test_monte_carlo_samples_are_compatible_reproducible_and_diagnosed() -> None:
    topology = _small_topology()
    state = PublicAttackState(topology, frozenset({0}), frozenset({8}))

    first, first_diagnostics = sample_compatible_fleets(
        state, SPECS, sample_count=32, rng=np.random.default_rng(12)
    )
    second, second_diagnostics = sample_compatible_fleets(
        state, SPECS, sample_count=32, rng=np.random.default_rng(12)
    )

    assert not first.exact
    assert first_diagnostics.posterior_exact is False
    assert first_diagnostics.accepted_samples == 32
    assert first_diagnostics.to_dict() == second_diagnostics.to_dict()
    assert np.array_equal(
        first.occupancy_probabilities(), second.occupancy_probabilities()
    )
    assert all(0 in fleet.occupied_cells and 8 not in fleet.occupied_cells for fleet in first.fleets)


def test_public_state_from_observation_reads_only_outcome_channels() -> None:
    observation = np.zeros((4, BATTLESHIP.rows, BATTLESHIP.columns), dtype=np.uint8)
    observation[1, 1, 2] = 1
    observation[2, 3, 4] = 1
    observation[3, 5, 6] = 1

    state = PublicAttackState.from_observation(BATTLESHIP, observation)

    assert state.hit_cells == {20, 58}
    assert state.sunk_cells == {58}
    assert state.missed_cells == {96}


def test_belief_population_rejects_hidden_state_that_conflicts_with_history() -> None:
    placement = ShipPlacement("alpha", 2, 0, "horizontal", (0, 1))
    fleet = Fleet((placement,))
    state = PublicAttackState(BATTLESHIP, frozenset(), frozenset({0}))

    with pytest.raises(ValueError, match="incompatible"):
        BeliefPopulation(state, (fleet,), "test", True)


def test_constrained_sampler_handles_sunk_and_active_hits_on_full_board() -> None:
    """Regression: announced sinks must not make valid public histories fail."""
    state = PublicAttackState(
        BATTLESHIP,
        frozenset({54, 55, 56, 57, 58, 59, 60, 98, 116, 134, 152, 163, 164, 165}),
        frozenset(
            {2, 7, 21, 25, 40, 45, 75, 78, 80, 94, 95, 97, 110, 112, 115, 117,
             126, 127, 131, 147, 150, 169}
        ),
        frozenset({54, 55, 56, 57, 58, 59, 60, 163, 164, 165}),
    )

    belief, diagnostics = sample_compatible_fleets(
        state, sample_count=2, rng=np.random.default_rng(42)
    )

    assert belief.size == 2
    assert diagnostics.completion_rate == 1.0


@pytest.mark.parametrize(
    "sampler_id, kwargs",
    (
        ("constrained-backtracking-v1", {}),
        ("constrained-backtracking-short-v1", {}),
        ("importance-v1", {"importance_resamples": 3}),
        ("mcmc-v1", {"mcmc_steps": 16}),
    ),
)
def test_sampler_variants_preserve_public_compatibility_constraints(
    sampler_id: str, kwargs: dict[str, int | float]
) -> None:
    """Different samplers should all return compatible finite populations."""
    state = PublicAttackState(
        BATTLESHIP,
        frozenset({54, 55, 56, 57, 58, 59, 60, 98, 116, 134, 152, 163, 164, 165}),
        frozenset(
            {2, 7, 21, 25, 40, 45, 75, 78, 80, 94, 95, 97, 110, 112, 115, 117,
             126, 127, 131, 147, 150, 169}
        ),
        frozenset({54, 55, 56, 57, 58, 59, 60, 163, 164, 165}),
    )

    belief, diagnostics = sample_compatible_fleets(
        state,
        sample_count=3,
        rng=np.random.default_rng(123),
        sampler_id=sampler_id,
        max_restarts_per_sample=64,
        max_nodes_per_sample=4_096,
        **kwargs,
    )

    assert not diagnostics.posterior_exact
    assert belief.sampler_id == sampler_id
    assert belief.size == 3


def test_sampler_rejects_invalid_sampler_id() -> None:
    with pytest.raises(ValueError, match="unsupported sampler_id"):
        sample_compatible_fleets(
            PublicAttackState(_small_topology(), frozenset(), frozenset()),
            sample_count=2,
            rng=np.random.default_rng(1),
            sampler_id="unsupported-v1",
        )
