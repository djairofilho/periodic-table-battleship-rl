"""Tests for the optional MaskablePPO attack training interface."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.envs import AttackEnvironmentConfig
from periodic_table_battleship_rl.training import attack
from periodic_table_battleship_rl.training.attack import (
    ATTACK_POLICY_ID,
    VALIDATION_CURVE_SCHEMA_VERSION,
    AttackValidationConfig,
    AttackValidationResult,
    AttackTrainingConfig,
    MaskableAttackPolicy,
    TrainingDependencyError,
    evaluate_attack_validation,
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
        self.learn_timesteps: list[int] = []

    def learn(
        self,
        *,
        total_timesteps: int,
        progress_bar: bool,
        reset_num_timesteps: bool = True,
    ) -> _FakeModel:
        assert progress_bar is False
        del reset_num_timesteps
        self.learn_timesteps.append(total_timesteps)
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
    assert metadata["environment"]["configuration"] == {
        "observation_profile": "outcomes-v1",
        "reward_profile": "hit-miss-terminal-v1",
    }
    assert metadata["config"]["seed"] == 11
    assert artifact.policy_id == ATTACK_POLICY_ID


def test_validation_config_rejects_unsorted_checkpoint_steps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        AttackValidationConfig(seeds=(101,), checkpoint_steps=(16, 8))


def test_train_captures_validation_checkpoints_without_test_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attack, "_require_maskable_ppo", lambda: _FakeModel)
    observed_steps: list[int] = []

    def fake_evaluate(
        topology: object,
        policy: object,
        validation: AttackValidationConfig,
        *,
        training_step: int,
        environment_config: AttackEnvironmentConfig | None = None,
    ) -> tuple[AttackValidationResult, ...]:
        del topology, policy, environment_config
        observed_steps.append(training_step)
        return tuple(
            AttackValidationResult(
                training_step=training_step,
                seed=seed,
                episode_index=0,
                valid_shots=80 - training_step,
                hit_segments=17,
                won=True,
                truncated=False,
                auc_discovery=0.5,
            )
            for seed in validation.seeds
        )

    monkeypatch.setattr(attack, "evaluate_attack_validation", fake_evaluate)
    validation = AttackValidationConfig(
        seeds=(101, 102), checkpoint_steps=(4, 8)
    )

    artifact = train_attack_policy(
        BATTLESHIP,
        _config(tmp_path, total_timesteps=12),
        validation=validation,
    )

    assert observed_steps == [4, 8]
    assert [checkpoint.training_step for checkpoint in artifact.checkpoints] == [4, 8]
    assert artifact.checkpoints[0].checkpoint_path.is_file()
    assert artifact.validation_curve_path is not None
    curve = json.loads(artifact.validation_curve_path.read_text(encoding="utf-8"))
    assert curve["schema_version"] == VALIDATION_CURVE_SCHEMA_VERSION
    assert curve["validation"] == {
        "split": "validation",
        "seeds": [101, 102],
        "checkpoint_steps": [4, 8],
        "episodes_per_seed": 1,
    }
    assert curve["checkpoints"][1]["mean_valid_shots"] == 72.0
    metadata = load_training_metadata(artifact.metadata_path)
    assert metadata["validation_curve"]["schedule"] == curve["validation"]


def test_validation_schedule_cannot_exceed_training_budget(tmp_path: Path) -> None:
    validation = AttackValidationConfig(seeds=(101,), checkpoint_steps=(9,))

    with pytest.raises(ValueError, match="cannot exceed"):
        attack._validate_schedule(_config(tmp_path, total_timesteps=8), validation)


def test_public_validation_evaluator_records_one_result_per_seed() -> None:
    class FirstLegalActionModel:
        def predict(
            self,
            observation: object,
            *,
            action_masks: np.ndarray,
            deterministic: bool,
        ) -> tuple[np.int64, None]:
            del observation, deterministic
            return np.int64(np.flatnonzero(action_masks)[0]), None

    results = evaluate_attack_validation(
        BATTLESHIP,
        MaskableAttackPolicy(model=FirstLegalActionModel()),
        AttackValidationConfig(seeds=(101, 102), checkpoint_steps=(8,)),
        training_step=8,
    )

    assert [(result.seed, result.episode_index) for result in results] == [
        (101, 0),
        (102, 0),
    ]
    assert all(result.won for result in results)
    assert all(result.hit_segments == 17 for result in results)


def test_load_metadata_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_training_metadata(path)


def test_policy_passes_action_masks_to_loaded_model() -> None:
    policy = MaskableAttackPolicy(model=_FakeModel())

    assert policy.select_action(np.zeros((4, 10, 18)), np.ones(180, dtype=bool)) == 7
