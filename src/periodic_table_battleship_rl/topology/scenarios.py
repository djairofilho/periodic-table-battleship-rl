"""Versioned topologies for Battleship and Periodic Table Battleship.

The periodic-table layout follows the IUPAC periodic table.  The 18 group
columns are represented directly; lanthanides and actinides use rows 8 and 9
(zero based) under groups 4 through 18.  Canvas row 7 is intentionally empty,
which prevents edges from joining the f-block to the main table.
"""

from __future__ import annotations

from types import MappingProxyType

from .model import Cell, Series, Topology

CANVAS_ROWS = 10
CANVAS_COLUMNS = 18
TOPOLOGY_VERSION = "topology-v1"

# This citation is kept with the layout because the atomic-number-to-position
# mapping is scientific source data, not presentation-only metadata.
IUPAC_PERIODIC_TABLE_SOURCE = MappingProxyType(
    {
        "publisher": "International Union of Pure and Applied Chemistry (IUPAC)",
        "title": "Periodic Table of the Elements",
        "edition": "2022-05-04",
        "url": "https://iupac.org/what-we-do/periodic-table-of-elements/",
        "accessed": "2026-07-20",
    }
)

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf "
    "Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()

_GROUPS_BY_PERIOD: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((1, 1), (2, 18)),
    2: tuple((number, group) for number, group in zip(range(3, 11), (1, 2, 13, 14, 15, 16, 17, 18), strict=True)),
    3: tuple((number, group) for number, group in zip(range(11, 19), (1, 2, 13, 14, 15, 16, 17, 18), strict=True)),
    4: tuple((number, group) for number, group in zip(range(19, 37), range(1, 19), strict=True)),
    5: tuple((number, group) for number, group in zip(range(37, 55), range(1, 19), strict=True)),
    6: ((55, 1), (56, 2), *((number, group) for number, group in zip(range(72, 87), range(4, 19), strict=True))),
    7: ((87, 1), (88, 2), *((number, group) for number, group in zip(range(104, 119), range(4, 19), strict=True))),
}


def _action(row: int, column: int) -> int:
    return row * CANVAS_COLUMNS + column


def _make_topology(name: str, cells: list[Cell]) -> Topology:
    by_action = {cell.action: cell for cell in cells}
    if len(by_action) != len(cells):
        raise ValueError(f"{name} contains duplicated actions")

    neighbors: dict[int, tuple[int, ...]] = {}
    for action, cell in by_action.items():
        candidates = (
            _action(cell.row - 1, cell.column) if cell.row > 0 else None,
            _action(cell.row, cell.column - 1) if cell.column > 0 else None,
            _action(cell.row, cell.column + 1) if cell.column + 1 < CANVAS_COLUMNS else None,
            _action(cell.row + 1, cell.column) if cell.row + 1 < CANVAS_ROWS else None,
        )
        neighbors[action] = tuple(sorted(candidate for candidate in candidates if candidate in by_action))

    return Topology(
        name=name,
        rows=CANVAS_ROWS,
        columns=CANVAS_COLUMNS,
        cells_by_action=MappingProxyType(by_action),
        neighbors_by_action=MappingProxyType(neighbors),
    )


def _classic_cells() -> list[Cell]:
    return [
        Cell(action=_action(row, column), row=row, column=column)
        for row in range(10)
        for column in range(10)
    ]


def _element_cell(
    atomic_number: int, *, row: int, column: int, period: int, group: int | None, series: Series
) -> Cell:
    return Cell(
        action=_action(row, column),
        row=row,
        column=column,
        atomic_number=atomic_number,
        symbol=_SYMBOLS[atomic_number - 1],
        period=period,
        group=group,
        series=series,
    )


def _periodic_table_cells() -> list[Cell]:
    cells: list[Cell] = []
    for period, entries in _GROUPS_BY_PERIOD.items():
        for atomic_number, group in entries:
            cells.append(
                _element_cell(
                    atomic_number,
                    row=period - 1,
                    column=group - 1,
                    period=period,
                    group=group,
                    series="main",
                )
            )

    for atomic_number in range(57, 72):
        cells.append(
            _element_cell(
                atomic_number,
                row=8,
                column=atomic_number - 54,
                period=6,
                group=None,
                series="lanthanide",
            )
        )
    for atomic_number in range(89, 104):
        cells.append(
            _element_cell(
                atomic_number,
                row=9,
                column=atomic_number - 86,
                period=7,
                group=None,
                series="actinide",
            )
        )
    return cells


BATTLESHIP = _make_topology("battleship", _classic_cells())
PERIODIC_TABLE_BATTLESHIP = _make_topology(
    "periodic-table-battleship", _periodic_table_cells()
)

TOPOLOGIES = MappingProxyType(
    {
        BATTLESHIP.name: BATTLESHIP,
        PERIODIC_TABLE_BATTLESHIP.name: PERIODIC_TABLE_BATTLESHIP,
    }
)


def get_topology(name: str) -> Topology:
    """Return one of the immutable ``topology-v1`` scenarios."""

    if name == "dense-118":
        # Kept lazy because the control topology reuses this module's canvas
        # constants to make its geometry explicit.
        from .dense import DENSE_118

        return DENSE_118

    try:
        return TOPOLOGIES[name]
    except KeyError as error:
        raise ValueError(f"unknown topology: {name!r}") from error
