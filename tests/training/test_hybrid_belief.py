"""Tests for the isolated public-belief PPO candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from periodic_table_battleship_rl.belief import BeliefFeatureConfig
from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training import hybrid_belief
from periodic_table_battleship_rl.training.attack import AttackValidationConfig
from periodic_table_battleship_rl.training.hybrid_belief import (
    HYBRID_BELIEF_PPO_POLICY_ID,
    HybridBeliefAttackTrainingConfig,
    load_hybrid_belief_training_metadata,
    train_hybrid_belief_attack_policy,
)


class _FakeModel:
    def __init__(self, *_arguments: object, **_keywords: object) -> None:
        pass

    def learn(self, *, total_timesteps: int, **_keywords: object) -> "_FakeModel":
        assert total_timesteps > 0
        return self

    def save(self, path: str) -> None:
        Path(path).with_suffix(".zip").write_bytes(b"hybrid")

    def predict(
        self, _observation: object, *, action_masks: np.ndarray, **_keywords: object
    ) -> tuple[np.int64, None]:
        return np.int64(np.flatnonzero(action_masks)[0]), None


def _config(tmp_path: Path) -> HybridBeliefAttackTrainingConfig:
    return HybridBeliefAttackTrainingConfig(
        run_id="hybrid-smoke",
        seed=41,
        total_timesteps=8,
        checkpoint_directory=tmp_path,
        n_steps=8,
        batch_size=8,
        features_dim=16,
        belief_config=BeliefFeatureConfig(sample_count=1, max_nodes_per_sample=4_096),
    )


def test_hybrid_training_persists_public_belief_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hybrid_belief, "_require_maskable_ppo", lambda: _FakeModel)
    monkeypatch.setattr(hybrid_belief, "cnn_policy_kwargs", lambda _features_dim: {})

    artifact = train_hybrid_belief_attack_policy(BATTLESHIP, _config(tmp_path))
    metadata = load_hybrid_belief_training_metadata(artifact.metadata_path)

    assert artifact.policy_id == HYBRID_BELIEF_PPO_POLICY_ID
    assert metadata["environment"]["belief_features"]["posterior_exact"] is False
    assert metadata["architecture"]["extra_public_channels"] == [
        "occupancy_probability",
        "outcome_entropy",
    ]


def test_hybrid_training_captures_validation_without_test_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hybrid_belief, "_require_maskable_ppo", lambda: _FakeModel)
    monkeypatch.setattr(hybrid_belief, "cnn_policy_kwargs", lambda _features_dim: {})

    artifact = train_hybrid_belief_attack_policy(
        BATTLESHIP,
        _config(tmp_path),
        validation=AttackValidationConfig(seeds=(101,), checkpoint_steps=(8,)),
    )

    assert artifact.checkpoints[0].training_step == 8
    assert artifact.checkpoints[0].checkpoint_path.is_file()
