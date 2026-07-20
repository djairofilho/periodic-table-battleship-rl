"""Deterministic-by-seed defensive attackers for placement evaluation.

These evaluators simulate a complete legal attack after a placement episode.
They receive the fleet only inside :class:`~periodic_table_battleship_rl.envs.
placement.PlacementEnv`'s terminal transition; their action choice uses only
the public attack history represented by an action mask and active hits.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

import numpy as np

from periodic_table_battleship_rl.game import Fleet
from periodic_table_battleship_rl.policies import hunt_target_action, random_masked_action
from periodic_table_battleship_rl.topology import Topology


@runtime_checkable
class DefensiveEvaluator(Protocol):
    """A versioned attacker that can score a completed defensive fleet."""

    evaluator_id: str

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        """Return the number of valid shots needed to sink ``fleet``."""


@dataclass(frozen=True, slots=True)
class RandomMaskedEvaluator:
    """Uniform random attack over the currently legal shot mask."""

    topology: Topology

    evaluator_id = "random-masked-v1"

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        """Sink ``fleet`` by repeatedly sampling one uncalled valid cell."""

        return _simulate_attack(
            self.topology,
            fleet,
            choose_action=lambda mask, active_hits: random_masked_action(mask, rng),
        )


@dataclass(frozen=True, slots=True)
class HuntTargetEvaluator:
    """Attack randomly until a hit, then target its unsunk neighbours."""

    topology: Topology

    evaluator_id = "hunt-target-v1"

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        """Sink ``fleet`` with the public-state hunt-target baseline."""

        return _simulate_attack(
            self.topology,
            fleet,
            choose_action=lambda mask, active_hits: hunt_target_action(
                self.topology, mask, active_hits, rng
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenDefensiveMixture:
    """A fixed weighted mixture of versioned defensive attackers.

    The component list and normalized weights are immutable.  Each completed
    placement receives one component sampled with the environment's episode
    RNG, so resetting ``PlacementEnv`` with the same seed reproduces both the
    selected attacker and all of its tie breaks.
    """

    evaluators: tuple[DefensiveEvaluator, ...]
    weights: tuple[float, ...]
    evaluator_id: str = "frozen-defensive-mixture-v1"

    def __post_init__(self) -> None:
        if not self.evaluators:
            raise ValueError("a defensive mixture requires at least one evaluator")
        if len(self.evaluators) != len(self.weights):
            raise ValueError("mixture evaluators and weights must have the same length")
        if len({evaluator.evaluator_id for evaluator in self.evaluators}) != len(
            self.evaluators
        ):
            raise ValueError("mixture evaluator ids must be unique")
        if any(not isfinite(weight) or weight <= 0.0 for weight in self.weights):
            raise ValueError("mixture weights must be finite and positive")

        total = sum(self.weights)
        object.__setattr__(self, "weights", tuple(weight / total for weight in self.weights))

    @property
    def component_ids(self) -> tuple[str, ...]:
        """Return stable component identifiers in sampling order."""

        return tuple(evaluator.evaluator_id for evaluator in self.evaluators)

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        """Evaluate with one component sampled from the frozen weights."""

        index = int(rng.choice(len(self.evaluators), p=self.weights))
        return self.evaluators[index].evaluate(fleet, rng=rng)


DEFAULT_DEFENSIVE_WEIGHTS = (0.5, 0.5)
"""Frozen weights for the required random and hunt-target attacker mixture."""


def default_defensive_mixture(topology: Topology) -> FrozenDefensiveMixture:
    """Build the benchmark's versioned, topology-aware defensive suite."""

    return FrozenDefensiveMixture(
        evaluators=(RandomMaskedEvaluator(topology), HuntTargetEvaluator(topology)),
        weights=DEFAULT_DEFENSIVE_WEIGHTS,
    )


def _simulate_attack(
    topology: Topology,
    fleet: Fleet,
    *,
    choose_action: "AttackChooser",
) -> int:
    """Run a legal masked attack without exposing fleet state to a policy."""

    occupied_cells = _validate_fleet(topology, fleet)
    action_mask = np.zeros(topology.action_count, dtype=np.bool_)
    action_mask[list(topology.valid_cells)] = True
    cells_by_ship = {placement.ship_id: frozenset(placement.cells) for placement in fleet.placements}
    ship_by_cell = {
        cell: placement.ship_id for placement in fleet.placements for cell in placement.cells
    }
    hit_cells: set[int] = set()
    active_hits: set[int] = set()
    shots = 0

    while hit_cells != occupied_cells:
        action = choose_action(action_mask, active_hits)
        if (
            not isinstance(action, int)
            or isinstance(action, bool)
            or not 0 <= action < action_mask.size
            or not action_mask[action]
        ):
            raise RuntimeError("defensive attacker selected an illegal masked action")

        action_mask[action] = False
        shots += 1
        ship_id = ship_by_cell.get(action)
        if ship_id is None:
            continue

        hit_cells.add(action)
        active_hits.add(action)
        if cells_by_ship[ship_id].issubset(hit_cells):
            active_hits.difference_update(cells_by_ship[ship_id])

    return shots


class AttackChooser(Protocol):
    """Choose one legal action from a public masked attack state."""

    def __call__(self, action_mask: np.ndarray, active_hits: Collection[int]) -> int:
        """Return one currently valid action."""


def _validate_fleet(topology: Topology, fleet: Fleet) -> frozenset[int]:
    """Validate the minimum fleet invariants needed by a hidden simulation."""

    if not fleet.placements:
        raise ValueError("a defensive evaluator requires a non-empty fleet")
    occupied_cells = fleet.occupied_cells
    if len(occupied_cells) != sum(placement.length for placement in fleet.placements):
        raise ValueError("defensive fleet placements must not overlap")
    if not occupied_cells.issubset(topology.valid_cells):
        raise ValueError("defensive fleet must stay within the evaluator topology")
    return occupied_cells
