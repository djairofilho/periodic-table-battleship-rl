"""Immutable topology primitives shared by the game environments.

The coordinate system is deliberately independent from an element's atomic
number: actions always refer to one cell of the 10 by 18 logical canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Series = Literal["main", "lanthanide", "actinide"]
Orientation = Literal["horizontal", "vertical"]


@dataclass(frozen=True, slots=True)
class Cell:
    """A playable location on a scenario canvas."""

    action: int
    row: int
    column: int
    atomic_number: int | None = None
    symbol: str | None = None
    period: int | None = None
    group: int | None = None
    series: Series | None = None


@dataclass(frozen=True, slots=True)
class Topology:
    """A fixed set of cells and their orthogonal adjacency graph."""

    name: str
    rows: int
    columns: int
    cells_by_action: dict[int, Cell]
    neighbors_by_action: dict[int, tuple[int, ...]]

    @property
    def action_count(self) -> int:
        """Number of actions in the fixed logical canvas."""

        return self.rows * self.columns

    @property
    def valid_actions(self) -> frozenset[int]:
        return frozenset(self.cells_by_action)

    @property
    def valid_cells(self) -> frozenset[int]:
        """Alias used by the game engine for playable action indices."""

        return self.valid_actions

    @property
    def valid_cell_count(self) -> int:
        return len(self.cells_by_action)

    def is_valid_action(self, action: int) -> bool:
        return action in self.cells_by_action

    def cell(self, action: int) -> Cell:
        """Return a cell, raising ``KeyError`` for a gap or out-of-range action."""

        return self.cells_by_action[action]

    def action_for(self, row: int, column: int) -> int:
        """Map a canvas coordinate to its stable action index.

        This method maps coordinates in the canvas, even when the resulting
        action is a gap. Call :meth:`is_valid_action` when a playable cell is
        required.
        """

        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise ValueError(f"coordinate outside {self.rows}x{self.columns} canvas")
        return row * self.columns + column

    def coordinate_for(self, action: int) -> tuple[int, int]:
        """Map an action index to a canvas coordinate."""

        if not (0 <= action < self.action_count):
            raise ValueError(f"action {action} outside [0, {self.action_count})")
        return divmod(action, self.columns)

    def neighbors(self, action: int) -> tuple[int, ...]:
        """Return valid orthogonal neighbours in action-index order."""

        return self.neighbors_by_action[action]

    def segment_from(
        self, anchor: int, orientation: Orientation, length: int
    ) -> tuple[int, ...] | None:
        """Build one contiguous segment or return ``None`` when it is illegal.

        The complete candidate must stay inside this topology.  In particular,
        gaps and the empty row separating the f-block prevent a ship from
        crossing between disconnected parts of the periodic table.
        """

        if length < 1 or anchor not in self.cells_by_action:
            return None
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(f"unknown orientation: {orientation!r}")

        row, column = self.coordinate_for(anchor)
        row_step, column_step = (0, 1) if orientation == "horizontal" else (1, 0)
        actions: list[int] = []
        for offset in range(length):
            candidate_row = row + offset * row_step
            candidate_column = column + offset * column_step
            if not (0 <= candidate_row < self.rows and 0 <= candidate_column < self.columns):
                return None
            candidate = self.action_for(candidate_row, candidate_column)
            if candidate not in self.cells_by_action:
                return None
            actions.append(candidate)
        return tuple(actions)
