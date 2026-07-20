"""Tests for the public-only, fixed-seed PPO attack evaluator."""

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
from periodic_table_battleship_rl.experiments.ppo_evaluation import (
    run_ppo_attack_evaluation,
    validate_ppo_checkpoint,
)
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP
from periodic_table_battleship_rl.training.attack import ATTACK_POLICY_ID, MaskableAttackPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOFTWARE = SoftwareMetadata(python_version="3.11.9", platform="test-platform")
HARDWARE = HardwareMetadata(machine="test-machine", processor="test-cpu", cpu_count=8)


class _PublicOnlyModel:
    """A fake PPO model that can decide solely from observation and mask."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, bool]] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        action_masks: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.int64, None]:
        self.calls.append((observation.copy(), action_masks.copy(), deterministic))
        return np.int64(np.flatnonzero(action_masks)[0]), None


def _config() -> RunConfig:
    return RunConfig(
        run_id="ppo-battleship-blind",
        experiment="attack",
        scenario="battleship",
        environment_version="attack-env-v1",
        policy_id=ATTACK_POLICY_ID,
        split="test",
        seeds=(17, 29),
        episodes_per_seed=1,
    )


def _metadata(topology_name: str = "battleship") -> dict[str, Any]:
    valid_cells = 100 if topology_name == "battleship" else 118
    return {
        "schema_version": "attack-training-v1",
        "algorithm": "MaskablePPO",
        "policy_id": ATTACK_POLICY_ID,
        "run_id": "training-run",
        "seed": 3,
        "scenario": topology_name,
        "environment": {
            "class": "AttackEnv",
            "action_mask_method": "action_masks",
            "action_count": 180,
            "valid_cells": valid_cells,
        },
    }


def _artifacts(tmp_path: Path, topology_name: str = "battleship") -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"frozen-model")
    metadata = tmp_path / "training.json"
    metadata.write_text(json.dumps(_metadata(topology_name)), encoding="utf-8")
    return checkpoint, metadata


def _run(tmp_path: Path, directory_name: str):
    checkpoint, metadata = _artifacts(tmp_path)
    model = _PublicOnlyModel()
    return run_ppo_attack_evaluation(
        _config(),
        BATTLESHIP,
        MaskableAttackPolicy(model=model),
        tmp_path / directory_name,
        checkpoint_path=checkpoint,
        training_metadata_path=metadata,
        git_commit="a" * 40,
        uv_lock_path=PROJECT_ROOT / "uv.lock",
        software=SOFTWARE,
        hardware=HARDWARE,
    ), model


def test_blind_evaluation_is_reproducible_persisted_and_comparable(tmp_path: Path) -> None:
    first, first_model = _run(tmp_path / "first", "run")
    second, _ = _run(tmp_path / "second", "run")

    assert [result.to_dict() for result in first.results] == [
        result.to_dict() for result in second.results
    ]
    assert first.summary == second.summary
    assert first.persisted.manifest_path.read_bytes() == second.persisted.manifest_path.read_bytes()
    assert first.persisted.episodes_path.read_bytes() == second.persisted.episodes_path.read_bytes()
    assert set(first.summary["aggregate"]) == {
        "valid_shots",
        "valid_shots_normalized",
        "shots_excess",
        "win_rate",
        "truncation_rate",
        "hit_rate",
        "invalid_attempts",
        "auc_discovery",
        "first_hit_shot",
        "first_sunk_shot",
    }
    assert first.manifest.config.parameters["evaluation_protocol"] == (
        "blind-public-observation-v1"
    )
    assert len(first_model.calls) == sum(result.valid_shots for result in first.results)


def test_policy_receives_only_public_observation_and_mask(tmp_path: Path) -> None:
    evaluation, model = _run(tmp_path, "run")

    first_observation, first_mask, deterministic = model.calls[0]
    assert first_observation.shape == (4, 10, 18)
    assert first_observation.dtype == np.uint8
    assert first_mask.shape == (180,)
    assert first_mask.dtype == np.bool_
    assert deterministic is True
    assert np.array_equal(first_observation[0].reshape(-1), first_mask)
    assert all(result.invalid_attempts == 0 for result in evaluation.results)


def test_checkpoint_metadata_must_match_supplied_topology(tmp_path: Path) -> None:
    checkpoint, metadata = _artifacts(tmp_path, topology_name="battleship")

    with pytest.raises(ValueError, match="scenario"):
        validate_ppo_checkpoint(
            PERIODIC_TABLE_BATTLESHIP,
            MaskableAttackPolicy(model=_PublicOnlyModel()),
            checkpoint_path=checkpoint,
            training_metadata_path=metadata,
        )


def test_missing_checkpoint_is_rejected_before_evaluation(tmp_path: Path) -> None:
    _, metadata = _artifacts(tmp_path)

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        validate_ppo_checkpoint(
            BATTLESHIP,
            MaskableAttackPolicy(model=_PublicOnlyModel()),
            checkpoint_path=tmp_path / "missing.zip",
            training_metadata_path=metadata,
        )
