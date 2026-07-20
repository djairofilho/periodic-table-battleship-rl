from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from periodic_table_battleship_rl.game import (
    CANONICAL_FLEET,
    Fleet,
    FleetSamplingError,
    Orientation,
    ShipPlacement,
    ShipSpec,
    candidate_placements,
    is_legal_fleet,
    sample_random_legal_fleet,
)


@dataclass(frozen=True)
class GridTopology:
    rows: int
    columns: int
    valid_cells: frozenset[int]

    @classmethod
    def rectangle(cls, rows: int, columns: int) -> GridTopology:
        return cls(rows, columns, frozenset(range(rows * columns)))

    def segment_from(
        self, anchor: int, orientation: Orientation, length: int
    ) -> tuple[int, ...] | None:
        if anchor not in self.valid_cells:
            return None
        row, column = divmod(anchor, self.columns)
        row_step, column_step = (
            (0, 1) if orientation is Orientation.HORIZONTAL else (1, 0)
        )
        end_row = row + (length - 1) * row_step
        end_column = column + (length - 1) * column_step
        if end_row >= self.rows or end_column >= self.columns:
            return None
        cells = tuple(
            (row + offset * row_step) * self.columns + column + offset * column_step
            for offset in range(length)
        )
        return cells if set(cells).issubset(self.valid_cells) else None


def test_canonical_fleet_has_stable_order_and_seventeen_segments() -> None:
    assert [(spec.ship_id, spec.length) for spec in CANONICAL_FLEET] == [
        ("ship-5", 5),
        ("ship-4", 4),
        ("ship-3a", 3),
        ("ship-3b", 3),
        ("ship-2", 2),
    ]
    assert sum(spec.length for spec in CANONICAL_FLEET) == 17


def test_candidate_placements_respect_geometry_overlap_and_permitted_contact() -> None:
    topology = GridTopology.rectangle(3, 4)

    candidates = candidate_placements(topology, 2, occupied_cells={0, 1})

    assert candidates
    assert all({0, 1}.isdisjoint(candidate.cells) for candidate in candidates)
    # (4, 5) touches the occupied ship vertically and must remain valid.
    assert any(candidate.cells == (4, 5) for candidate in candidates)
    assert all(len(candidate.cells) == 2 for candidate in candidates)


def test_candidate_placements_exclude_gaps_supplied_by_topology() -> None:
    topology = GridTopology(2, 4, frozenset({0, 1, 3, 4, 5, 7}))

    candidates = candidate_placements(topology, 2)

    assert all(set(candidate.cells).issubset(topology.valid_cells) for candidate in candidates)
    assert not any(candidate.cells == (0, 1, 2) for candidate in candidates)


def test_sampler_returns_legal_complete_fleet() -> None:
    topology = GridTopology.rectangle(10, 10)

    fleet = sample_random_legal_fleet(topology, np.random.default_rng(7))

    assert is_legal_fleet(topology, fleet)
    assert fleet.segment_count == 17
    assert len(fleet.occupied_cells) == 17
    assert fleet.restarts >= 0


def test_same_seed_produces_same_fleet_and_restart_count() -> None:
    topology = GridTopology.rectangle(10, 10)

    first = sample_random_legal_fleet(topology, np.random.default_rng(839))
    second = sample_random_legal_fleet(topology, np.random.default_rng(839))

    assert first == second


def test_sampler_restarts_a_dead_end_with_the_same_random_source() -> None:
    # Choosing the central (1, 2) segment first on this line leaves no legal
    # segment for the second ship, although a complete fleet does exist.
    topology = GridTopology.rectangle(1, 4)
    specs = (ShipSpec("first", 2), ShipSpec("second", 2))

    fleet = sample_random_legal_fleet(
        topology, np.random.default_rng(1), specs, max_restarts=10
    )

    assert fleet.restarts >= 1
    assert is_legal_fleet(topology, fleet, specs)


def test_sampler_accepts_a_non_rectangular_topology_protocol() -> None:
    # Two 5x5 components demonstrate that the sampler does not depend on a
    # specific topology implementation or assume rectangular validity.
    valid = frozenset(range(25)) | frozenset(range(50, 75))
    topology = GridTopology(15, 5, valid)

    fleet = sample_random_legal_fleet(topology, np.random.default_rng(3))

    assert is_legal_fleet(topology, fleet)


def test_sampler_raises_after_configured_restart_limit_for_infeasible_topology() -> None:
    topology = GridTopology.rectangle(2, 2)

    with pytest.raises(FleetSamplingError, match="could not sample"):
        sample_random_legal_fleet(topology, np.random.default_rng(1), max_restarts=0)


def test_fleet_legality_rejects_overlap_and_wrong_ship_identity() -> None:
    topology = GridTopology.rectangle(1, 4)
    specs = (ShipSpec("first", 2), ShipSpec("second", 2))
    overlapping = Fleet(
        (
            ShipPlacement("first", 2, 0, Orientation.HORIZONTAL, (0, 1)),
            ShipPlacement("second", 2, 1, Orientation.HORIZONTAL, (1, 2)),
        )
    )
    wrong_identity = Fleet(
        (
            ShipPlacement("other", 2, 0, Orientation.HORIZONTAL, (0, 1)),
            ShipPlacement("second", 2, 2, Orientation.HORIZONTAL, (2, 3)),
        )
    )

    assert not is_legal_fleet(topology, overlapping, specs)
    assert not is_legal_fleet(topology, wrong_identity, specs)
