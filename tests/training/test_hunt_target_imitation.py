"""Public-data guarantees and schemas for hunt-target imitation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training.imitation import (
    HUNT_TARGET_DATASET_SCHEMA_VERSION,
    HuntTargetDatasetConfig,
    ImitationTrainingConfig,
    generate_hunt_target_dataset,
    load_hunt_target_dataset,
    load_imitation_training_metadata,
)


def test_dataset_is_deterministic_and_certifies_only_public_fields(tmp_path: Path) -> None:
    config = HuntTargetDatasetConfig(
        dataset_id="teacher",
        seeds=(73, 79),
        output_directory=tmp_path,
    )
    first = generate_hunt_target_dataset(BATTLESHIP, config)
    first_bytes = first.data_path.read_bytes()
    second = generate_hunt_target_dataset(BATTLESHIP, config)
    metadata = json.loads(second.metadata_path.read_text(encoding="utf-8"))

    observations, masks, actions = load_hunt_target_dataset(second.data_path)
    assert first_bytes == second.data_path.read_bytes()
    assert metadata["schema_version"] == HUNT_TARGET_DATASET_SCHEMA_VERSION
    assert metadata["public_fields"] == ["observations", "action_masks", "actions"]
    assert "fleet" in metadata["excluded_hidden_fields"]
    assert observations.shape[0] == masks.shape[0] == actions.shape[0] == first.sample_count
    assert np.all(masks[np.arange(len(actions)), actions])


def test_dataset_loader_rejects_hidden_or_illegal_shapes(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        observations=np.zeros((1, 4, 10, 18), dtype=np.uint8),
        action_masks=np.ones((1, 180), dtype=bool),
        actions=np.array([0], dtype=np.int64),
        fleet=np.array([1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="exactly public fields"):
        load_hunt_target_dataset(path)


def test_imitation_config_rejects_checkpoint_outside_finetuning_budget(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="within the fine-tune budget"):
        ImitationTrainingConfig(
            run_id="bad",
            seed=1,
            dataset_path=tmp_path / "dataset.npz",
            checkpoint_directory=tmp_path,
            fine_tune_timesteps=8,
            fine_tune_checkpoint_steps=(16,),
        )


def test_imitation_metadata_rejects_other_training_formats(tmp_path: Path) -> None:
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"schema_version": "attack-training-v1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported imitation"):
        load_imitation_training_metadata(path)
