"""Unit tests for masked-DQN legality and checkpoint provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training.dqn import (
    DqnAttackTrainingConfig,
    DqnTransition,
    ReplayBuffer,
    dqn_targets,
    load_dqn_training_metadata,
    load_masked_dqn_attack_policy,
    masked_bootstrap_values,
    train_masked_dqn_attack_policy,
)


def _config(tmp_path: Path, **overrides: object) -> DqnAttackTrainingConfig:
    values: dict[str, object] = {
        "run_id": "dqn-smoke",
        "seed": 71,
        "total_steps": 16,
        "checkpoint_directory": tmp_path,
        "batch_size": 4,
        "warmup_steps": 4,
        "replay_capacity": 16,
        "target_update_interval": 4,
        "hidden_dim": 16,
    }
    values.update(overrides)
    return DqnAttackTrainingConfig(**values)  # type: ignore[arg-type]


def test_config_rejects_invalid_epsilon_schedule(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        _config(tmp_path, epsilon_start=0.1, epsilon_end=0.2)


def test_replay_rejects_non_boolean_next_mask() -> None:
    buffer = ReplayBuffer(2)
    with pytest.raises(TypeError, match="dtype bool"):
        buffer.append(
            DqnTransition(
                observation=np.zeros((4, 10, 18), dtype=np.uint8),
                action=1,
                reward=1.0,
                next_observation=np.zeros((4, 10, 18), dtype=np.uint8),
                terminated=False,
                truncated=False,
                next_action_mask=np.ones(180, dtype=np.uint8),
            )
        )


def test_target_mask_excludes_high_valued_illegal_action() -> None:
    torch = pytest.importorskip("torch")
    next_values = torch.tensor([[1.0, 999.0, 4.0], [7.0, 8.0, 9.0]])
    masks = torch.tensor([[True, False, True], [False, False, False]])

    bootstrap = masked_bootstrap_values(next_values, masks)
    targets = dqn_targets(
        torch.tensor([2.0, 3.0]),
        torch.tensor([False, True]),
        next_values,
        masks,
        gamma=0.5,
    )

    assert bootstrap.tolist() == [4.0, 0.0]
    assert targets.tolist() == [4.0, 3.0]


def test_tiny_train_persists_loadable_public_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    artifact = train_masked_dqn_attack_policy(BATTLESHIP, _config(tmp_path))

    metadata = load_dqn_training_metadata(artifact.metadata_path)
    policy = load_masked_dqn_attack_policy(artifact.checkpoint_path)
    observation = np.zeros((4, 10, 18), dtype=np.uint8)
    mask = np.zeros(180, dtype=np.bool_)
    mask[3] = True

    assert metadata["scenario"] == "battleship"
    assert artifact.replay_size == 16
    assert policy.select_action(observation, mask) == 3
