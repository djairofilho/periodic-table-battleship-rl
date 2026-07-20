"""Tests for the explicit, public-only cross-topology PPO evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from periodic_table_battleship_rl.evaluation.schemas import (
    HardwareMetadata,
    RunConfig,
    SoftwareMetadata,
)
from periodic_table_battleship_rl.experiments.cross_topology import (
    CROSS_TOPOLOGY_PROTOCOL,
    CrossTopologyPpoSource,
    run_cross_topology_matrix,
    run_cross_topology_ppo_attack_evaluation,
)
from periodic_table_battleship_rl.experiments.ppo_evaluation import (
    validate_ppo_checkpoint,
)
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP, Topology
from periodic_table_battleship_rl.training.attack import ATTACK_POLICY_ID, MaskableAttackPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOFTWARE = SoftwareMetadata(python_version="3.11.9", platform="test-platform")
HARDWARE = HardwareMetadata(machine="test-machine", processor="test-cpu", cpu_count=8)


class _PublicOnlyModel:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        action_masks: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.int64, None]:
        assert deterministic is True
        self.calls.append((observation.copy(), action_masks.copy()))
        return np.int64(np.flatnonzero(action_masks)[0]), None


def _metadata(topology: Topology) -> dict[str, Any]:
    return {
        "schema_version": "attack-training-v1",
        "algorithm": "MaskablePPO",
        "policy_id": ATTACK_POLICY_ID,
        "run_id": f"training-{topology.name}",
        "seed": 3,
        "scenario": topology.name,
        "environment": {
            "class": "AttackEnv",
            "action_mask_method": "action_masks",
            "action_count": topology.action_count,
            "valid_cells": topology.valid_cell_count,
        },
    }


def _source(tmp_path: Path, topology: Topology) -> tuple[CrossTopologyPpoSource, _PublicOnlyModel]:
    source_directory = tmp_path / topology.name
    source_directory.mkdir(parents=True)
    checkpoint = source_directory / "model.zip"
    checkpoint.write_bytes(f"frozen-{topology.name}".encode())
    metadata = source_directory / "training.json"
    metadata.write_text(json.dumps(_metadata(topology)), encoding="utf-8")
    model = _PublicOnlyModel()
    return (
        CrossTopologyPpoSource(
            topology=topology,
            policy=MaskableAttackPolicy(model=model),
            checkpoint_path=checkpoint,
            training_metadata_path=metadata,
        ),
        model,
    )


def _config(source: Topology, target: Topology) -> RunConfig:
    return RunConfig(
        run_id=f"cross-{source.name}-to-{target.name}",
        experiment="attack",
        scenario=target.name,
        environment_version="attack-env-v1",
        policy_id=ATTACK_POLICY_ID,
        split="test",
        seeds=(17, 29),
        episodes_per_seed=1,
    )


def test_cross_topology_is_auditable_and_uses_only_public_state(tmp_path: Path) -> None:
    source, model = _source(tmp_path, BATTLESHIP)
    evaluation = run_cross_topology_ppo_attack_evaluation(
        _config(BATTLESHIP, PERIODIC_TABLE_BATTLESHIP),
        source,
        PERIODIC_TABLE_BATTLESHIP,
        tmp_path / "run",
        git_commit="a" * 40,
        uv_lock_path=PROJECT_ROOT / "uv.lock",
        software=SOFTWARE,
        hardware=HARDWARE,
    )

    parameters = evaluation.manifest.config.parameters
    assert evaluation.source_topology == "battleship"
    assert evaluation.target_topology == "periodic-table-battleship"
    assert parameters["evaluation_protocol"] == CROSS_TOPOLOGY_PROTOCOL
    assert parameters["source_scenario"] == "battleship"
    assert parameters["target_scenario"] == "periodic-table-battleship"
    assert parameters["source_valid_cells"] == 100
    assert parameters["target_valid_cells"] == 118
    assert {result.scenario for result in evaluation.results} == {
        "periodic-table-battleship"
    }
    first_observation, first_mask = model.calls[0]
    assert first_observation.shape == (4, 10, 18)
    assert first_observation.dtype == np.uint8
    assert first_mask.shape == (180,)
    assert first_mask.dtype == np.bool_
    assert np.array_equal(first_observation[0].reshape(-1), first_mask)


def test_standard_evaluator_remains_strict_for_cross_topology_source(tmp_path: Path) -> None:
    source, _ = _source(tmp_path, BATTLESHIP)

    with pytest.raises(ValueError, match="scenario"):
        validate_ppo_checkpoint(
            PERIODIC_TABLE_BATTLESHIP,
            source.policy,
            checkpoint_path=source.checkpoint_path,
            training_metadata_path=source.training_metadata_path,
        )


def test_matrix_covers_every_ordered_pair_with_target_config(tmp_path: Path) -> None:
    classic, _ = _source(tmp_path, BATTLESHIP)
    periodic, _ = _source(tmp_path, PERIODIC_TABLE_BATTLESHIP)
    matrix = run_cross_topology_matrix(
        (classic, periodic),
        (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP),
        tmp_path / "matrix",
        _config,
        git_commit="b" * 40,
        uv_lock_path=PROJECT_ROOT / "uv.lock",
        software=SOFTWARE,
        hardware=HARDWARE,
    )

    assert set(matrix.by_pair()) == {
        ("battleship", "battleship"),
        ("battleship", "periodic-table-battleship"),
        ("periodic-table-battleship", "battleship"),
        ("periodic-table-battleship", "periodic-table-battleship"),
    }
    assert all(
        evaluation.manifest.config.scenario == evaluation.target_topology
        for evaluation in matrix.evaluations
    )


def test_cross_topology_rejects_conflicting_provenance_parameter(tmp_path: Path) -> None:
    source, _ = _source(tmp_path, BATTLESHIP)
    config = RunConfig(
        **{
            **_config(BATTLESHIP, PERIODIC_TABLE_BATTLESHIP).to_dict(),
            "parameters": {"source_scenario": "wrong-source"},
        }
    )

    with pytest.raises(ValueError, match="source_scenario"):
        run_cross_topology_ppo_attack_evaluation(
            config,
            source,
            PERIODIC_TABLE_BATTLESHIP,
            tmp_path / "run",
            git_commit="a" * 40,
            uv_lock_path=PROJECT_ROOT / "uv.lock",
            software=SOFTWARE,
            hardware=HARDWARE,
        )
