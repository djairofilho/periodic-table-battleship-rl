"""Dense 118-cell control topology for causal benchmark comparisons.

``dense-118`` preserves the 10 by 18 action canvas and the number of playable
cells in Periodic Table Battleship, while removing its gaps and disconnected
f-block.  It therefore isolates irregular geometry from cell cardinality in
experiments that compare the two 118-cell scenarios.
"""

from __future__ import annotations

from types import MappingProxyType

from .model import Cell, Topology
from .scenarios import CANVAS_COLUMNS, CANVAS_ROWS


DENSE_118_SCENARIO = MappingProxyType(
    {
        "name": "dense-118",
        "version": "dense-118-v1",
        "purpose": "cardinality control for periodic-table-battleship",
        "geometry": "six full rows plus ten cells in the seventh row",
        "valid_cell_count": 118,
    }
)
"""Immutable metadata describing the causal-control scenario."""


def _action(row: int, column: int) -> int:
    return row * CANVAS_COLUMNS + column


def _cells() -> list[Cell]:
    """Return a connected, left-aligned 118-cell polyomino.

    Six complete rows contain 108 cells.  The first ten cells of row seven
    complete the 118-cell control without introducing an internal gap.
    """

    return [
        Cell(action=_action(row, column), row=row, column=column)
        for row in range(7)
        for column in range(CANVAS_COLUMNS if row < 6 else 10)
    ]


def _make_dense_118() -> Topology:
    cells = _cells()
    by_action = {cell.action: cell for cell in cells}
    neighbors: dict[int, tuple[int, ...]] = {}

    for action, cell in by_action.items():
        candidates = (
            _action(cell.row - 1, cell.column) if cell.row > 0 else None,
            _action(cell.row, cell.column - 1) if cell.column > 0 else None,
            _action(cell.row, cell.column + 1)
            if cell.column + 1 < CANVAS_COLUMNS
            else None,
            _action(cell.row + 1, cell.column)
            if cell.row + 1 < CANVAS_ROWS
            else None,
        )
        neighbors[action] = tuple(
            sorted(candidate for candidate in candidates if candidate in by_action)
        )

    return Topology(
        name=DENSE_118_SCENARIO["name"],
        rows=CANVAS_ROWS,
        columns=CANVAS_COLUMNS,
        cells_by_action=MappingProxyType(by_action),
        neighbors_by_action=MappingProxyType(neighbors),
    )


DENSE_118 = _make_dense_118()
"""The versioned dense 118-cell causal-control topology."""
