"""Frozen MaskablePPO attacker for defensive placement evaluation.

The evaluator keeps the learned policy on the attacking side of a placement
episode.  It builds the same public four-channel state used by ``AttackEnv``
and calls the policy with that observation plus an action mask.  Fleet state
is used only *after* the policy chooses a shot to resolve its outcome.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from periodic_table_battleship_rl.game import Fleet
from periodic_table_battleship_rl.topology import Topology
from periodic_table_battleship_rl.training.attack import (
    ATTACK_POLICY_ID,
    TRAINING_SCHEMA_VERSION,
    MaskableAttackPolicy,
)


@dataclass(frozen=True, slots=True)
class FrozenPPOEvaluator:
    """Evaluate fleets with one A3 MaskablePPO attacker checkpoint.

    ``training_metadata`` is the public ``training.json`` object written by
    A3.  ``checkpoint_id`` is deliberately supplied by the caller, rather
    than derived from an absolute local path, so that the evaluator identity
    remains portable across machines and result manifests.
    """

    policy: MaskableAttackPolicy
    topology: Topology
    training_metadata: Mapping[str, Any]
    checkpoint_id: str

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must not be empty")

        metadata = self.training_metadata
        if metadata.get("schema_version") != TRAINING_SCHEMA_VERSION:
            raise ValueError("unsupported A3 training metadata schema version")
        if metadata.get("policy_id") != ATTACK_POLICY_ID:
            raise ValueError("training metadata does not describe a MaskablePPO attacker")
        if self.policy.policy_id != ATTACK_POLICY_ID:
            raise ValueError("policy is not a MaskablePPO attack policy")
        if metadata.get("scenario") != self.topology.name:
            raise ValueError("A3 scenario does not match evaluator topology")

        environment = metadata.get("environment")
        if not isinstance(environment, Mapping):
            raise ValueError("A3 training metadata must include environment provenance")
        if environment.get("class") != "AttackEnv":
            raise ValueError("A3 metadata must describe an AttackEnv")
        if environment.get("action_mask_method") != "action_masks":
            raise ValueError("A3 metadata must use action_masks")
        if environment.get("action_count") != self.topology.action_count:
            raise ValueError("A3 action count does not match evaluator topology")
        if environment.get("valid_cells") != self.topology.valid_cell_count:
            raise ValueError("A3 valid-cell count does not match evaluator topology")

    @property
    def evaluator_id(self) -> str:
        """Return a portable, checkpoint-specific identifier for manifests."""

        run_id = self.training_metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("A3 training metadata must include a non-empty run_id")
        return f"{ATTACK_POLICY_ID}:{run_id}:{self.checkpoint_id}"

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        """Return valid shots required to sink ``fleet`` with public inputs only."""

        del rng
        occupied_cells = _validate_fleet(self.topology, fleet)
        action_mask = np.zeros(self.topology.action_count, dtype=np.bool_)
        action_mask[list(self.topology.valid_cells)] = True
        cells_by_ship = {
            placement.ship_id: frozenset(placement.cells) for placement in fleet.placements
        }
        ship_by_cell = {
            cell: placement.ship_id
            for placement in fleet.placements
            for cell in placement.cells
        }
        hit_cells: set[int] = set()
        active_hits: set[int] = set()
        sunk_cells: set[int] = set()
        missed_cells: set[int] = set()
        shots = 0

        while hit_cells != occupied_cells:
            observation = _public_observation(
                self.topology, active_hits, sunk_cells, missed_cells
            )
            action = self.policy.select_action(observation, action_mask, deterministic=True)
            if (
                not isinstance(action, int)
                or isinstance(action, bool)
                or not 0 <= action < action_mask.size
                or not action_mask[action]
            ):
                raise RuntimeError("frozen PPO attacker selected an illegal masked action")

            action_mask[action] = False
            shots += 1
            ship_id = ship_by_cell.get(action)
            if ship_id is None:
                missed_cells.add(action)
                continue

            hit_cells.add(action)
            active_hits.add(action)
            if cells_by_ship[ship_id].issubset(hit_cells):
                sunk_cells.update(cells_by_ship[ship_id])
                active_hits.difference_update(cells_by_ship[ship_id])

        return shots


def _public_observation(
    topology: Topology,
    active_hits: set[int],
    sunk_cells: set[int],
    missed_cells: set[int],
) -> np.ndarray:
    """Build ``AttackEnv``'s public observation without exposing a fleet."""

    observation = np.zeros((4, topology.rows, topology.columns), dtype=np.uint8)
    valid_cells = tuple(topology.valid_cells)
    valid_rows, valid_columns = zip(*(topology.coordinate_for(cell) for cell in valid_cells))
    observation[0, valid_rows, valid_columns] = 1
    for channel, cells in ((1, active_hits), (2, sunk_cells), (3, missed_cells)):
        if cells:
            rows, columns = zip(*(topology.coordinate_for(cell) for cell in cells))
            observation[channel, rows, columns] = 1
    return observation


def _validate_fleet(topology: Topology, fleet: Fleet) -> frozenset[int]:
    """Reject malformed fleets before they can enter hidden simulation state."""

    if not fleet.placements:
        raise ValueError("a frozen PPO evaluator requires a non-empty fleet")
    occupied_cells = fleet.occupied_cells
    if len(occupied_cells) != sum(placement.length for placement in fleet.placements):
        raise ValueError("defensive fleet placements must not overlap")
    if not occupied_cells.issubset(topology.valid_cells):
        raise ValueError("defensive fleet must stay within the evaluator topology")
    return occupied_cells
