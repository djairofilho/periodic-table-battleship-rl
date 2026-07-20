"""Tests for the isolated spatial MaskablePPO candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training import cnn
from periodic_table_battleship_rl.training.attack import AttackValidationConfig
from periodic_table_battleship_rl.training.cnn import (
    CNN_ATTACK_POLICY_ID,
    CnnAttackTrainingConfig,
    load_cnn_training_metadata,
    train_cnn_attack_policy,
)


class _FakeModel:
    def __init__(self, *arguments: object, **keywords: object) -> None:
        self.arguments = arguments
        self.keywords = keywords

    def learn(self, *, total_timesteps: int, **_kwargs: object) -> "_FakeModel":
        assert total_timesteps > 0
        return self

    def save(self, path: str) -> None:
        Path(path).with_suffix(".zip").write_bytes(b"cnn")

    def predict(
        self, _observation: object, *, action_masks: np.ndarray, **_kwargs: object
    ) -> tuple[np.int64, None]:
        return np.int64(np.flatnonzero(action_masks)[0]), None


def _config(tmp_path: Path, **overrides: object) -> CnnAttackTrainingConfig:
    values: dict[str, object] = {
        "run_id": "cnn-smoke",
        "seed": 31,
        "total_timesteps": 8,
        "checkpoint_directory": tmp_path,
        "n_steps": 8,
        "batch_size": 8,
        "features_dim": 16,
    }
    values.update(overrides)
    return CnnAttackTrainingConfig(**values)  # type: ignore[arg-type]


def test_cnn_training_persists_distinct_public_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cnn, "_require_maskable_ppo", lambda: _FakeModel)
    monkeypatch.setattr(cnn, "cnn_policy_kwargs", lambda _features_dim: {"cnn": True})

    artifact = train_cnn_attack_policy(BATTLESHIP, _config(tmp_path))

    metadata = load_cnn_training_metadata(artifact.metadata_path)
    assert artifact.policy_id == CNN_ATTACK_POLICY_ID
    assert metadata["policy_id"] == CNN_ATTACK_POLICY_ID
    assert metadata["architecture"]["features_extractor"] == "spatial-cnn-adaptive-pool-v1"
    assert metadata["environment"]["configuration"]["observation_profile"] == "outcomes-v1"


def test_cnn_training_captures_validation_without_a_test_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cnn, "_require_maskable_ppo", lambda: _FakeModel)
    monkeypatch.setattr(cnn, "cnn_policy_kwargs", lambda _features_dim: {})

    artifact = train_cnn_attack_policy(
        BATTLESHIP,
        _config(tmp_path),
        validation=AttackValidationConfig(seeds=(101,), checkpoint_steps=(8,)),
    )

    assert artifact.checkpoints[0].training_step == 8
    assert artifact.checkpoints[0].checkpoint_path.is_file()


def test_cnn_metadata_rejects_an_mlp_or_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"schema_version": "attack-training-v1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported CNN"):
        load_cnn_training_metadata(path)
