"""Tests for blind P5 placement evaluation with fake masked PPO models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from periodic_table_battleship_rl.evaluation.schemas import (
    HardwareMetadata,
    RunConfig,
    SoftwareMetadata,
)
from periodic_table_battleship_rl.experiments.placement_evaluation import (
    PLACEMENT_ENVIRONMENT_VERSION,
    run_placement_evaluation,
    validate_placement_checkpoint,
)
from periodic_table_battleship_rl.game import Fleet
from periodic_table_battleship_rl.placement import FrozenDefensiveMixture
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP, Topology
from periodic_table_battleship_rl.training.placement import (
    PLACEMENT_POLICY_ID,
    MaskablePlacementPolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOFTWARE = SoftwareMetadata(python_version="3.11.9", platform="test-platform")
HARDWARE = HardwareMetadata(machine="test-machine", processor="test-cpu", cpu_count=8)


@dataclass(frozen=True)
class _FixedEvaluator:
    evaluator_id: str
    shots: int

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        del fleet, rng
        return self.shots


@dataclass
class _PublicOnlyModel:
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


def _mixture() -> FrozenDefensiveMixture:
    return FrozenDefensiveMixture(
        evaluators=(
            _FixedEvaluator("random-masked-v1", 80),
            _FixedEvaluator("hunt-target-v1", 50),
        ),
        weights=(0.5, 0.5),
    )


def _config() -> RunConfig:
    return RunConfig(
        run_id="ppo-placement-battleship-blind",
        experiment="placement",
        scenario="battleship",
        environment_version=PLACEMENT_ENVIRONMENT_VERSION,
        policy_id=PLACEMENT_POLICY_ID,
        split="test",
        seeds=(17, 29),
        episodes_per_seed=1,
    )


def _metadata(topology: Topology, mixture: FrozenDefensiveMixture) -> dict[str, Any]:
    return {
        "schema_version": "placement-training-v1",
        "algorithm": "MaskablePPO",
        "policy_id": PLACEMENT_POLICY_ID,
        "run_id": "p4-training-run",
        "seed": 3,
        "scenario": topology.name,
        "environment": {
            "class": "PlacementEnv",
            "action_mask_method": "action_masks",
            "action_count": 360,
            "valid_cells": topology.valid_cell_count,
            "fleet_order": [5, 4, 3, 3, 2],
        },
        "defensive_mixture": {
            "evaluator_id": mixture.evaluator_id,
            "component_ids": list(mixture.component_ids),
            "weights": list(mixture.weights),
        },
    }


def _artifacts(
    tmp_path: Path, topology: Topology, mixture: FrozenDefensiveMixture
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"frozen-placement-model")
    metadata = tmp_path / "training.json"
    metadata.write_text(json.dumps(_metadata(topology, mixture)), encoding="utf-8")
    return checkpoint, metadata


def _run(tmp_path: Path, directory_name: str):
    mixture = _mixture()
    checkpoint, metadata = _artifacts(tmp_path, BATTLESHIP, mixture)
    model = _PublicOnlyModel()
    evaluation = run_placement_evaluation(
        _config(),
        BATTLESHIP,
        MaskablePlacementPolicy(model=model),
        mixture,
        tmp_path / directory_name,
        checkpoint_path=checkpoint,
        training_metadata_path=metadata,
        git_commit="b" * 40,
        uv_lock_path=PROJECT_ROOT / "uv.lock",
        software=SOFTWARE,
        hardware=HARDWARE,
    )
    return evaluation, model


def test_evaluation_persists_paired_component_and_mixture_metrics(tmp_path: Path) -> None:
    evaluation, _ = _run(tmp_path, "run")

    assert len(evaluation.results) == 6
    assert {result.attacker_id for result in evaluation.results} == {
        "random-masked-v1",
        "hunt-target-v1",
        "frozen-defensive-mixture-v1",
    }
    assert evaluation.summary["episode_count"] == 6
    assert evaluation.summary["components"]["random-masked-v1"]["aggregate"][
        "valid_shots_to_sink"
    ]["mean"] == 80.0
    assert evaluation.summary["components"]["hunt-target-v1"]["aggregate"][
        "valid_shots_to_sink"
    ]["mean"] == 50.0
    assert evaluation.summary["mixture"]["episode_count"] == 2
    assert all(result.hit_segments == 17 for result in evaluation.results)
    assert all(result.all_sunk_shot == result.valid_shots_to_sink for result in evaluation.results)
    assert evaluation.persisted.manifest_path.is_file()
    assert evaluation.persisted.episodes_path.is_file()
    assert evaluation.manifest.config.parameters["evaluation_protocol"] == (
        "blind-public-observation-v1"
    )
    assert evaluation.manifest.config.parameters["defensive_mixture"] == _metadata(
        BATTLESHIP, _mixture()
    )["defensive_mixture"]


def test_policy_receives_only_public_placement_observation_and_mask(tmp_path: Path) -> None:
    evaluation, model = _run(tmp_path, "run")

    assert len(model.calls) == len(evaluation.results) * 5
    first_observation, first_mask, deterministic = model.calls[0]
    assert first_observation.shape == (3, 10, 18)
    assert first_observation.dtype == np.float32
    assert first_mask.shape == (360,)
    assert first_mask.dtype == np.bool_
    assert deterministic is True
    assert np.array_equal(
        first_observation[0].astype(bool),
        np.isin(np.arange(180), tuple(BATTLESHIP.valid_cells)).reshape(10, 18),
    )
    assert all(len(result.placement_actions) == 5 for result in evaluation.results)


def test_evaluation_is_reproducible_for_identical_held_out_schedule(tmp_path: Path) -> None:
    first, _ = _run(tmp_path / "first", "run")
    second, _ = _run(tmp_path / "second", "run")

    assert [result.to_dict() for result in first.results] == [
        result.to_dict() for result in second.results
    ]
    assert first.summary == second.summary
    assert first.persisted.manifest_path.read_bytes() == second.persisted.manifest_path.read_bytes()
    assert first.persisted.episodes_path.read_bytes() == second.persisted.episodes_path.read_bytes()


def test_checkpoint_metadata_must_match_topology_and_mixture(tmp_path: Path) -> None:
    mixture = _mixture()
    checkpoint, metadata = _artifacts(tmp_path, BATTLESHIP, mixture)

    with pytest.raises(ValueError, match="scenario"):
        validate_placement_checkpoint(
            PERIODIC_TABLE_BATTLESHIP,
            MaskablePlacementPolicy(_PublicOnlyModel()),
            mixture,
            checkpoint_path=checkpoint,
            training_metadata_path=metadata,
        )

    different_mixture = FrozenDefensiveMixture(
        evaluators=mixture.evaluators,
        weights=(0.8, 0.2),
    )
    with pytest.raises(ValueError, match="defensive mixture"):
        validate_placement_checkpoint(
            BATTLESHIP,
            MaskablePlacementPolicy(_PublicOnlyModel()),
            different_mixture,
            checkpoint_path=checkpoint,
            training_metadata_path=metadata,
        )


def test_missing_checkpoint_is_rejected_before_evaluation(tmp_path: Path) -> None:
    mixture = _mixture()
    _, metadata = _artifacts(tmp_path, BATTLESHIP, mixture)

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        validate_placement_checkpoint(
            BATTLESHIP,
            MaskablePlacementPolicy(_PublicOnlyModel()),
            mixture,
            checkpoint_path=tmp_path / "missing.zip",
            training_metadata_path=metadata,
        )
