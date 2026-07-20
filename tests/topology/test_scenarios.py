from __future__ import annotations

from collections import deque

import pytest

from periodic_table_battleship_rl.topology import (
    BATTLESHIP,
    CANVAS_COLUMNS,
    CANVAS_ROWS,
    IUPAC_PERIODIC_TABLE_SOURCE,
    PERIODIC_TABLE_BATTLESHIP,
    get_topology,
)


def _connected_component(topology, start: int) -> set[int]:
    visited = {start}
    pending = deque([start])
    while pending:
        action = pending.popleft()
        for neighbor in topology.neighbors(action):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return visited


@pytest.mark.parametrize(
    ("topology", "expected_cell_count"),
    [(BATTLESHIP, 100), (PERIODIC_TABLE_BATTLESHIP, 118)],
)
def test_topologies_have_the_versioned_canvas_and_cell_counts(topology, expected_cell_count):
    assert (topology.rows, topology.columns) == (CANVAS_ROWS, CANVAS_COLUMNS)
    assert topology.action_count == 180
    assert topology.valid_cell_count == expected_cell_count


def test_classic_topology_is_a_ten_by_ten_rectangle_on_the_canvas():
    assert BATTLESHIP.valid_actions == frozenset(range(10)) | frozenset(
        row * CANVAS_COLUMNS + column for row in range(1, 10) for column in range(10)
    )
    assert not BATTLESHIP.is_valid_action(BATTLESHIP.action_for(0, 10))


def test_action_coordinate_mapping_is_stable_and_validates_bounds():
    action = PERIODIC_TABLE_BATTLESHIP.action_for(8, 3)
    assert action == 147
    assert PERIODIC_TABLE_BATTLESHIP.coordinate_for(action) == (8, 3)
    assert PERIODIC_TABLE_BATTLESHIP.cell(action).symbol == "La"
    with pytest.raises(ValueError):
        PERIODIC_TABLE_BATTLESHIP.action_for(10, 0)
    with pytest.raises(ValueError):
        PERIODIC_TABLE_BATTLESHIP.coordinate_for(180)


def test_periodic_table_has_exactly_one_cell_per_iupac_element():
    cells = list(PERIODIC_TABLE_BATTLESHIP.cells_by_action.values())
    assert {cell.atomic_number for cell in cells} == set(range(1, 119))
    assert len({cell.symbol for cell in cells}) == 118
    assert IUPAC_PERIODIC_TABLE_SOURCE["publisher"].startswith("International Union")
    assert IUPAC_PERIODIC_TABLE_SOURCE["url"].startswith("https://iupac.org/")


def test_f_block_is_separated_from_main_table_by_an_empty_canvas_row():
    lanthanum = next(
        action
        for action, cell in PERIODIC_TABLE_BATTLESHIP.cells_by_action.items()
        if cell.atomic_number == 57
    )
    main_actions = {
        action
        for action, cell in PERIODIC_TABLE_BATTLESHIP.cells_by_action.items()
        if cell.series == "main"
    }
    assert _connected_component(PERIODIC_TABLE_BATTLESHIP, lanthanum).isdisjoint(main_actions)
    assert not PERIODIC_TABLE_BATTLESHIP.is_valid_action(
        PERIODIC_TABLE_BATTLESHIP.action_for(7, 3)
    )


def test_orthogonal_neighbors_never_cross_a_gap_or_use_diagonals():
    hydrogen = PERIODIC_TABLE_BATTLESHIP.action_for(0, 0)
    assert PERIODIC_TABLE_BATTLESHIP.neighbors(hydrogen) == (
        PERIODIC_TABLE_BATTLESHIP.action_for(1, 0),
    )

    beryllium = PERIODIC_TABLE_BATTLESHIP.action_for(1, 1)
    # Boron is separated by ten group columns and is not an orthogonal neighbor.
    assert PERIODIC_TABLE_BATTLESHIP.neighbors(beryllium) == (
        PERIODIC_TABLE_BATTLESHIP.action_for(1, 0),
        PERIODIC_TABLE_BATTLESHIP.action_for(2, 1),
    )

    for action in PERIODIC_TABLE_BATTLESHIP.valid_actions:
        row, column = PERIODIC_TABLE_BATTLESHIP.coordinate_for(action)
        for neighbor in PERIODIC_TABLE_BATTLESHIP.neighbors(action):
            neighbor_row, neighbor_column = PERIODIC_TABLE_BATTLESHIP.coordinate_for(neighbor)
            assert abs(row - neighbor_row) + abs(column - neighbor_column) == 1
            assert action in PERIODIC_TABLE_BATTLESHIP.neighbors(neighbor)


def test_segment_from_requires_a_complete_contiguous_legal_segment():
    assert BATTLESHIP.segment_from(0, "horizontal", 5) == (0, 1, 2, 3, 4)
    assert BATTLESHIP.segment_from(0, "vertical", 3) == (0, 18, 36)
    assert BATTLESHIP.segment_from(8, "horizontal", 3) is None

    beryllium = PERIODIC_TABLE_BATTLESHIP.action_for(1, 1)
    assert PERIODIC_TABLE_BATTLESHIP.segment_from(beryllium, "horizontal", 2) is None
    lanthanum = PERIODIC_TABLE_BATTLESHIP.action_for(8, 3)
    assert PERIODIC_TABLE_BATTLESHIP.segment_from(lanthanum, "vertical", 3) is None
    assert PERIODIC_TABLE_BATTLESHIP.valid_cells == PERIODIC_TABLE_BATTLESHIP.valid_actions


def test_periodic_cell_metadata_matches_known_iupac_positions():
    oxygen = PERIODIC_TABLE_BATTLESHIP.cell(
        PERIODIC_TABLE_BATTLESHIP.action_for(1, 15)
    )
    assert (oxygen.atomic_number, oxygen.symbol, oxygen.period, oxygen.group, oxygen.series) == (
        8,
        "O",
        2,
        16,
        "main",
    )
    lawrencium = PERIODIC_TABLE_BATTLESHIP.cell(
        PERIODIC_TABLE_BATTLESHIP.action_for(8, 17)
    )
    assert (lawrencium.atomic_number, lawrencium.symbol, lawrencium.period, lawrencium.group, lawrencium.series) == (
        71,
        "Lu",
        6,
        None,
        "lanthanide",
    )


def test_topology_lookup_rejects_unknown_names():
    assert get_topology("battleship") is BATTLESHIP
    with pytest.raises(ValueError, match="unknown topology"):
        get_topology("dense-118")
