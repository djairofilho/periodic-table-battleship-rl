"""Compatible-fleet beliefs derived only from public attack observations.

``exact_belief`` is deliberately bounded and intended for microboards.  The
Monte Carlo sampler is a constraint-preserving proposal for full boards: all
returned fleets are compatible, but their frequency is not claimed to be an
exact posterior.  Its diagnostics make that approximation explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from periodic_table_battleship_rl.game import (
    CANONICAL_FLEET,
    Fleet,
    ShipPlacement,
    ShipSpec,
    candidate_placements,
)
from periodic_table_battleship_rl.topology import Topology


class CompatibleFleetLimitError(RuntimeError):
    """Exact enumeration exceeded its explicit safety limit."""


@dataclass(frozen=True, slots=True)
class PublicAttackState:
    """The complete public history needed to constrain a hidden fleet.

    ``sunk_cells`` are provided by attack observation channel 2.  They avoid
    guessing ship identities while still enforcing that a compatible fleet
    explains every announced sink.
    """

    topology: Topology
    hit_cells: frozenset[int]
    missed_cells: frozenset[int]
    sunk_cells: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        valid = self.topology.valid_cells
        for name, cells in (
            ("hit_cells", self.hit_cells),
            ("missed_cells", self.missed_cells),
            ("sunk_cells", self.sunk_cells),
        ):
            if not cells.issubset(valid):
                raise ValueError(f"{name} must contain only valid topology cells")
        if self.hit_cells.intersection(self.missed_cells):
            raise ValueError("hit_cells and missed_cells must be disjoint")
        if not self.sunk_cells.issubset(self.hit_cells):
            raise ValueError("sunk_cells must be a subset of hit_cells")

    @classmethod
    def from_observation(
        cls,
        topology: Topology,
        observation: np.ndarray,
    ) -> "PublicAttackState":
        """Restore public state from ``AttackEnv`` outcome channels only."""
        if not isinstance(observation, np.ndarray) or observation.ndim != 3:
            raise TypeError("observation must be a three-dimensional NumPy array")
        expected_shape = (topology.rows, topology.columns)
        if observation.shape[1:] != expected_shape or observation.shape[0] < 4:
            raise ValueError(
                "observation must have at least four channels on the topology canvas"
            )

        def cells(channel: int) -> frozenset[int]:
            rows, columns = np.nonzero(observation[channel])
            return frozenset(
                topology.action_for(int(row), int(column))
                for row, column in zip(rows, columns, strict=True)
            )

        active_hits = cells(1)
        sunk_hits = cells(2)
        return cls(
            topology=topology,
            hit_cells=active_hits.union(sunk_hits),
            missed_cells=cells(3),
            sunk_cells=sunk_hits,
        )

    @property
    def called_cells(self) -> frozenset[int]:
        """Return all public shots already resolved."""
        return self.hit_cells.union(self.missed_cells)


@dataclass(frozen=True, slots=True)
class BeliefPopulation:
    """A finite, equally weighted population of compatible hidden fleets."""

    state: PublicAttackState
    fleets: tuple[Fleet, ...]
    sampler_id: str
    exact: bool

    def __post_init__(self) -> None:
        if not self.fleets:
            raise ValueError("a belief population must contain at least one fleet")
        if not self.sampler_id:
            raise ValueError("sampler_id must not be empty")
        if any(not _fleet_is_compatible(fleet, self.state) for fleet in self.fleets):
            raise ValueError("belief population contains an incompatible fleet")

    @property
    def size(self) -> int:
        return len(self.fleets)

    def occupancy_probabilities(self) -> np.ndarray:
        """Return a flat action-indexed occupancy probability vector."""
        counts = np.zeros(self.state.topology.action_count, dtype=np.float64)
        for fleet in self.fleets:
            counts[list(fleet.occupied_cells)] += 1.0
        return counts / float(self.size)

    def action_probabilities(self, action_mask: np.ndarray) -> np.ndarray:
        """Return occupancy probabilities masked to currently legal shots."""
        _validate_action_mask(self.state.topology, action_mask)
        probabilities = self.occupancy_probabilities()
        probabilities[~action_mask] = 0.0
        return probabilities

    def conditional(self, action: int, *, is_hit: bool) -> "BeliefPopulation":
        """Filter the finite belief after a hypothetical public shot result.

        The original state is intentionally retained: the finite-horizon
        planner uses this only to score a second action.  Constructing the
        full next public state would also require the announced sunk-ship
        result, which is not determined by a binary hit alone.
        """
        if action not in self.state.topology.valid_cells:
            raise ValueError("action must be a valid topology cell")
        fleets = tuple(
            fleet
            for fleet in self.fleets
            if (action in fleet.occupied_cells) is is_hit
        )
        if not fleets:
            raise ValueError("hypothetical outcome has zero support in this belief")
        return BeliefPopulation(
            state=self.state,
            fleets=fleets,
            sampler_id=f"{self.sampler_id}:conditioned",
            exact=self.exact,
        )


@dataclass(frozen=True, slots=True)
class MonteCarloDiagnostics:
    """Auditable work and limitations of constrained fleet sampling."""

    requested_samples: int
    accepted_samples: int
    restart_count: int
    backtrack_count: int
    max_restarts_per_sample: int
    max_nodes_per_sample: int
    sampler_id: str = "constrained-backtracking-v1"
    posterior_exact: bool = False

    @property
    def completion_rate(self) -> float:
        return self.accepted_samples / self.requested_samples

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "requested_samples": self.requested_samples,
            "accepted_samples": self.accepted_samples,
            "completion_rate": self.completion_rate,
            "restart_count": self.restart_count,
            "backtrack_count": self.backtrack_count,
            "max_restarts_per_sample": self.max_restarts_per_sample,
            "max_nodes_per_sample": self.max_nodes_per_sample,
            "sampler_id": self.sampler_id,
            "posterior_exact": self.posterior_exact,
        }


def enumerate_compatible_fleets(
    state: PublicAttackState,
    specs: Sequence[ShipSpec] = CANONICAL_FLEET,
    *,
    max_fleets: int = 100_000,
) -> tuple[Fleet, ...]:
    """Enumerate every compatible fleet, stopping before an unsafe explosion.

    The order is deterministic.  It is intended for small topologies and for
    validating approximations, not for the 118-cell production scenarios.
    """
    if max_fleets <= 0:
        raise ValueError("max_fleets must be positive")
    candidates = _state_candidates(state, specs)
    fleets: list[Fleet] = []

    def visit(index: int, occupied: frozenset[int], placements: tuple[ShipPlacement, ...]) -> None:
        if len(fleets) >= max_fleets:
            raise CompatibleFleetLimitError(
                f"compatible fleet enumeration reached max_fleets={max_fleets}"
            )
        if index == len(specs):
            fleet = Fleet(placements)
            if _fleet_is_compatible(fleet, state):
                fleets.append(fleet)
            return
        spec = specs[index]
        for placement in candidates[index]:
            cells = frozenset(placement.cells)
            if occupied.isdisjoint(cells):
                visit(
                    index + 1,
                    occupied.union(cells),
                    placements
                    + (
                        ShipPlacement(
                            ship_id=spec.ship_id,
                            length=spec.length,
                            anchor=placement.anchor,
                            orientation=placement.orientation,
                            cells=placement.cells,
                        ),
                    ),
                )

    visit(0, frozenset(), ())
    return tuple(fleets)


def exact_belief(
    state: PublicAttackState,
    specs: Sequence[ShipSpec] = CANONICAL_FLEET,
    *,
    max_fleets: int = 100_000,
) -> BeliefPopulation:
    """Create an equally weighted exact compatible-fleet belief."""
    return BeliefPopulation(
        state=state,
        fleets=enumerate_compatible_fleets(state, specs, max_fleets=max_fleets),
        sampler_id="exact-enumeration-v1",
        exact=True,
    )


def sample_compatible_fleets(
    state: PublicAttackState,
    specs: Sequence[ShipSpec] = CANONICAL_FLEET,
    *,
    sample_count: int,
    rng: np.random.Generator,
    max_restarts_per_sample: int = 128,
    max_nodes_per_sample: int = 8_192,
) -> tuple[BeliefPopulation, MonteCarloDiagnostics]:
    """Generate compatible fleets by randomized constrained backtracking.

    It never inspects the opponent fleet and never emits an incompatible
    sample.  Candidate order and backtracking induce a proposal distribution,
    therefore callers must not label it an exact posterior; the returned
    diagnostics make this explicit in persisted reports.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if max_restarts_per_sample <= 0:
        raise ValueError("max_restarts_per_sample must be positive")
    if max_nodes_per_sample <= 0:
        raise ValueError("max_nodes_per_sample must be positive")
    candidates = _state_candidates(state, specs)
    samples: list[Fleet] = []
    restart_count = 0
    backtrack_count = 0
    for _ in range(sample_count):
        fleet: Fleet | None = None
        for restart in range(max_restarts_per_sample):
            built, backtracks = _random_compatible_fleet(
                state, specs, candidates, rng, max_nodes=max_nodes_per_sample
            )
            backtrack_count += backtracks
            if built is not None:
                fleet = built
                restart_count += restart
                break
        if fleet is None:
            raise RuntimeError(
                "constrained sampler could not construct a compatible fleet; "
                "increase max_restarts_per_sample or inspect public state"
            )
        samples.append(fleet)
    diagnostics = MonteCarloDiagnostics(
        requested_samples=sample_count,
        accepted_samples=len(samples),
        restart_count=restart_count,
        backtrack_count=backtrack_count,
        max_restarts_per_sample=max_restarts_per_sample,
        max_nodes_per_sample=max_nodes_per_sample,
    )
    return (
        BeliefPopulation(
            state=state,
            fleets=tuple(samples),
            sampler_id=diagnostics.sampler_id,
            exact=False,
        ),
        diagnostics,
    )


def _state_candidates(
    state: PublicAttackState, specs: Sequence[ShipSpec]) -> tuple[tuple, ...]:
    candidates: list[tuple] = []
    for spec in specs:
        allowed = tuple(
            candidate
            for candidate in candidate_placements(state.topology, spec.length)
            if _placement_can_explain_public_history(
                frozenset(candidate.cells), state
            )
        )
        if not allowed:
            return tuple(() for _ in specs)
        candidates.append(allowed)
    return tuple(candidates)


def _placement_can_explain_public_history(
    cells: frozenset[int], state: PublicAttackState) -> bool:
    if cells.intersection(state.missed_cells):
        return False
    sunk_overlap = cells.intersection(state.sunk_cells)
    if sunk_overlap and not cells.issubset(state.sunk_cells):
        return False
    # An all-hit ship is necessarily already announced as sunk by AttackEnv.
    return not cells.issubset(state.hit_cells) or cells.issubset(state.sunk_cells)


def _fleet_is_compatible(fleet: Fleet, state: PublicAttackState) -> bool:
    occupied = fleet.occupied_cells
    if not state.hit_cells.issubset(occupied) or occupied.intersection(state.missed_cells):
        return False
    sunk_union: set[int] = set()
    for placement in fleet.placements:
        cells = frozenset(placement.cells)
        if not _placement_can_explain_public_history(cells, state):
            return False
        if cells.intersection(state.sunk_cells):
            sunk_union.update(cells)
    return frozenset(sunk_union) == state.sunk_cells


def _random_compatible_fleet(
    state: PublicAttackState,
    specs: Sequence[ShipSpec],
    candidates: tuple[tuple, ...],
    rng: np.random.Generator,
    *,
    max_nodes: int,
) -> tuple[Fleet | None, int]:
    backtracks = 0
    node_count = 0

    def visit(
        remaining_indices: tuple[int, ...],
        occupied: frozenset[int],
        placements: dict[int, ShipPlacement],
    ) -> Fleet | None:
        nonlocal backtracks, node_count
        node_count += 1
        if node_count > max_nodes:
            return None
        if not remaining_indices:
            fleet = Fleet(tuple(placements[index] for index in range(len(specs))))
            return fleet if _fleet_is_compatible(fleet, state) else None
        uncovered_hits = state.hit_cells.difference(occupied)
        remaining_candidates = tuple(candidates[index] for index in remaining_indices)
        if not _remaining_ships_can_cover(uncovered_hits, occupied, remaining_candidates):
            return None
        assignments = _next_assignments(
            remaining_indices, candidates, occupied, uncovered_hits
        )
        if not assignments:
            return None
        for assignment_index in rng.permutation(len(assignments)):
            index, candidate_index = assignments[int(assignment_index)]
            candidate = candidates[index][candidate_index]
            cells = frozenset(candidate.cells)
            spec = specs[index]
            candidate_fleet = visit(
                tuple(other for other in remaining_indices if other != index),
                occupied.union(cells),
                placements
                | {
                    index: ShipPlacement(
                        ship_id=spec.ship_id,
                        length=spec.length,
                        anchor=candidate.anchor,
                        orientation=candidate.orientation,
                        cells=candidate.cells,
                    )
                },
            )
            if candidate_fleet is not None:
                return candidate_fleet
            backtracks += 1
        return None

    return visit(tuple(range(len(specs))), frozenset(), {}), backtracks


def _next_assignments(
    remaining_indices: tuple[int, ...],
    candidates: tuple[tuple, ...],
    occupied: frozenset[int],
    uncovered_hits: frozenset[int],
) -> tuple[tuple[int, int], ...]:
    """Return assignments for the hardest unresolved hit, or a free ship."""
    if uncovered_hits:
        options_by_hit: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        for hit in sorted(uncovered_hits):
            options = tuple(
                (ship_index, candidate_index)
                for ship_index in remaining_indices
                for candidate_index, candidate in enumerate(candidates[ship_index])
                if hit in candidate.cells and occupied.isdisjoint(candidate.cells)
            )
            options_by_hit.append((hit, options))
        _, assignments = min(options_by_hit, key=lambda item: (len(item[1]), item[0]))
        return assignments

    # Once public constraints are explained, use minimum remaining values to
    # complete a valid fleet without spending search on uninformative cells.
    available_by_ship = [
        (
            ship_index,
            tuple(
                candidate_index
                for candidate_index, candidate in enumerate(candidates[ship_index])
                if occupied.isdisjoint(candidate.cells)
            ),
        )
        for ship_index in remaining_indices
    ]
    ship_index, candidate_indices = min(
        available_by_ship, key=lambda item: (len(item[1]), item[0])
    )
    return tuple((ship_index, candidate_index) for candidate_index in candidate_indices)


def _remaining_ships_can_cover(
    uncovered_hits: frozenset[int],
    occupied: frozenset[int],
    remaining_candidates: tuple[tuple, ...],
) -> bool:
    """Fast necessary condition used to prune impossible backtracking paths."""
    if not uncovered_hits:
        return True
    coverable: set[int] = set()
    for candidates in remaining_candidates:
        for candidate in candidates:
            cells = frozenset(candidate.cells)
            if occupied.isdisjoint(cells):
                coverable.update(cells)
    return uncovered_hits.issubset(coverable)


def _validate_action_mask(topology: Topology, action_mask: np.ndarray) -> None:
    if not isinstance(action_mask, np.ndarray) or action_mask.dtype != np.bool_:
        raise TypeError("action_mask must be a boolean NumPy array")
    if action_mask.ndim != 1 or action_mask.size != topology.action_count:
        raise ValueError("action_mask must match the topology action canvas")
