"""Tests for the frozen A3 MaskablePPO defensive evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from periodic_table_battleship_rl.game import sample_random_legal_fleet
from periodic_table_battleship_rl.placement.ppo import FrozenPPOEvaluator
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP, Topology
from periodic_table_battleship_rl.training.attack import ATTACK_POLICY_ID, MaskableAttackPolicy


def _metadata(topology: Topology, **overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": "attack-training-v1",
        "policy_id": ATTACK_POLICY_ID,
        "run_id": "a3-benchmark-001",
        "scenario": topology.name,
        "environment": {
            "class": "AttackEnv",
            "action_mask_method": "action_masks",
            "action_count": topology.action_count,
            "valid_cells": topology.valid_cell_count,
        },
    }
    metadata.update(overrides)
    return metadata


@dataclass
class _FirstLegalModel:
    calls: list[tuple[np.ndarray, np.ndarray, bool]] = field(default_factory=list)

    def predict(
        self,
        observation: np.ndarray,
        *,
        action_masks: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.int64, None]:
        self.calls.append((observation.copy(), action_masks.copy(), deterministic))
        return np.int64(np.flatnonzero(action_masks)[0]), None


def _evaluator(topology: Topology, **metadata_overrides: Any) -> tuple[FrozenPPOEvaluator, _FirstLegalModel]:
    model = _FirstLegalModel()
    return (
        FrozenPPOEvaluator(
            policy=MaskableAttackPolicy(model),
            topology=topology,
            training_metadata=_metadata(topology, **metadata_overrides),
            checkpoint_id="model-sha256:deadbeef",
        ),
        model,
    )


@pytest.mark.parametrize("topology", (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP))
def test_frozen_ppo_evaluator_uses_only_public_observation_and_mask(topology: Topology) -> None:
    evaluator, model = _evaluator(topology)
    fleet = sample_random_legal_fleet(topology, np.random.default_rng(21))

    shots = evaluator.evaluate(fleet, rng=np.random.default_rng(7))

    assert fleet.segment_count <= shots <= topology.valid_cell_count
    assert len(model.calls) == shots
    first_observation, first_mask, deterministic = model.calls[0]
    assert deterministic is True
    assert first_observation.shape == (4, 10, 18)
    assert first_observation[1:].sum() == 0
    assert first_mask.dtype == np.bool_
    assert first_mask.sum() == topology.valid_cell_count
    for observation, mask, _ in model.calls:
        assert np.array_equal(observation[0].astype(bool).ravel(), np.isin(
            np.arange(topology.action_count), tuple(topology.valid_cells)
        ))
        assert not np.any(observation[1] & observation[2])


def test_frozen_ppo_evaluator_has_a_portable_checkpoint_identity() -> None:
    evaluator, _ = _evaluator(BATTLESHIP)

    assert evaluator.evaluator_id == "maskable-ppo-v1:a3-benchmark-001:model-sha256:deadbeef"


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"scenario": "periodic-table-battleship"}, "scenario"),
        ({"policy_id": "other-policy"}, "MaskablePPO"),
        ({"environment": {"class": "AttackEnv"}}, "action_masks"),
        ({"environment": {"class": "AttackEnv", "action_mask_method": "action_masks", "action_count": 100, "valid_cells": 100}}, "action count"),
    ),
)
def test_frozen_ppo_evaluator_rejects_incompatible_a3_metadata(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _evaluator(BATTLESHIP, **overrides)


def test_frozen_ppo_evaluator_rejects_a_policy_that_ignores_the_mask() -> None:
    class _IllegalModel:
        def predict(self, *args: Any, **kwargs: Any) -> tuple[int, None]:
            del args, kwargs
            return 179, None

    evaluator = FrozenPPOEvaluator(
        policy=MaskableAttackPolicy(_IllegalModel()),
        topology=BATTLESHIP,
        training_metadata=_metadata(BATTLESHIP),
        checkpoint_id="checkpoint-1",
    )
    fleet = sample_random_legal_fleet(BATTLESHIP, np.random.default_rng(4))

    with pytest.raises(RuntimeError, match="illegal masked"):
        evaluator.evaluate(fleet, rng=np.random.default_rng(1))
