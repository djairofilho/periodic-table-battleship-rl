"""Tests for the optional MaskablePPO attack training interface."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training import attack
from periodic_table_battleship_rl.training.attack import (
    ATTACK_POLICY_ID,
    AttackTrainingConfig,
    MaskableAttackPolicy,
    TrainingDependencyError,
    load_training_metadata,
    train_attack_policy,
)


def _config(tmp_path: Path, **overrides: object) -> AttackTrainingConfig:
    values: dict[str, object] = {
        "run_id": "attack-smoke",
        "seed": 11,
        "total_timesteps": 8,
        "checkpoint_directory": tmp_path,
        "n_steps": 8,
        "batch_size": 8,
    }
    values.update(overrides)
    return AttackTrainingConfig(**values)  # type: ignore[arg-type]


def test_config_rejects_invalid_batch_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size cannot exceed n_steps"):
        _config(tmp_path, batch_size=9)


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def missing_sb3(name: str, *args: object, **kwargs: object) -> object:
        if name == "sb3_contrib":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sb3)
    with pytest.raises(TrainingDependencyError, match="uv sync --extra train"):
        attack._require_maskable_ppo()


class _FakeModel:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.arguments = args
        self.keywords = kwargs
        self.learn_timesteps: int | None = None

    def learn(self, *, total_timesteps: int, progress_bar: bool) -> _FakeModel:
        assert progress_bar is False
        self.learn_timesteps = total_timesteps
        return self

    def save(self, path: str) -> None:
        Path(path).with_suffix(".zip").write_bytes(b"checkpoint")

    def predict(
        self,
        observation: object,
        *,
        action_masks: object,
        deterministic: bool,
    ) -> tuple[np.int64, None]:
        del observation, action_masks, deterministic
        return np.int64(7), None


def test_train_writes_checkpoint_and_public_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attack, "_require_maskable_ppo", lambda: _FakeModel)

    artifact = train_attack_policy(BATTLESHIP, _config(tmp_path))

    assert artifact.checkpoint_path.read_bytes() == b"checkpoint"
    metadata = load_training_metadata(artifact.metadata_path)
    assert metadata["scenario"] == "battleship"
    assert metadata["environment"]["action_mask_method"] == "action_masks"
    assert metadata["config"]["seed"] == 11
    assert artifact.policy_id == ATTACK_POLICY_ID


def test_load_metadata_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_training_metadata(path)


def test_policy_passes_action_masks_to_loaded_model() -> None:
    policy = MaskableAttackPolicy(model=_FakeModel())

    assert policy.select_action(np.zeros((4, 10, 18)), np.ones(180, dtype=bool)) == 7
