"""Schema, determinism, and public-data guarantees for Bayesian teachers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training.bayesian_distillation import (
    BAYESIAN_DEMONSTRATION_SCHEMA_VERSION,
    BayesianDemonstrationConfig,
    generate_bayesian_demonstrations,
    load_bayesian_demonstration_metadata,
    load_bayesian_demonstrations,
)


def _config(tmp_path: Path) -> BayesianDemonstrationConfig:
    return BayesianDemonstrationConfig(
        dataset_id="teacher",
        seeds=(9701,),
        output_directory=tmp_path,
        sample_count=2,
        max_nodes_per_sample=4_096,
    )


def test_bayesian_dataset_is_deterministic_and_public_only(tmp_path: Path) -> None:
    first = generate_bayesian_demonstrations(BATTLESHIP, _config(tmp_path))
    first_bytes = first.data_path.read_bytes()
    second = generate_bayesian_demonstrations(BATTLESHIP, _config(tmp_path))
    metadata = load_bayesian_demonstration_metadata(second.data_path)
    demonstrations = load_bayesian_demonstrations(second.data_path)

    assert first_bytes == second.data_path.read_bytes()
    assert metadata["schema_version"] == BAYESIAN_DEMONSTRATION_SCHEMA_VERSION
    assert metadata["public_fields"] == [
        "observations",
        "action_masks",
        "teacher_actions",
        "teacher_occupancy_probabilities",
    ]
    assert "fleet" in metadata["excluded_hidden_fields"]
    assert metadata["teacher"]["posterior_exact"] is False
    assert demonstrations.sample_count == second.sample_count
    assert np.all(
        demonstrations.action_masks[
            np.arange(demonstrations.sample_count), demonstrations.teacher_actions
        ]
    )
    assert np.all(
        demonstrations.teacher_occupancy_probabilities[
            ~demonstrations.action_masks
        ]
        == 0.0
    )


def test_bayesian_loader_rejects_hidden_or_nonmaximal_teacher_fields(tmp_path: Path) -> None:
    hidden_path = tmp_path / "hidden.npz"
    np.savez_compressed(
        hidden_path,
        observations=np.zeros((1, 4, 10, 18), dtype=np.uint8),
        action_masks=np.ones((1, 180), dtype=bool),
        teacher_actions=np.array([0], dtype=np.int64),
        teacher_occupancy_probabilities=np.ones((1, 180), dtype=np.float32),
        fleet=np.array([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="exactly public fields"):
        load_bayesian_demonstrations(hidden_path)

    inconsistent_path = tmp_path / "inconsistent.npz"
    scores = np.zeros((1, 180), dtype=np.float32)
    scores[0, 1] = 1.0
    np.savez_compressed(
        inconsistent_path,
        observations=np.zeros((1, 4, 10, 18), dtype=np.uint8),
        action_masks=np.ones((1, 180), dtype=bool),
        teacher_actions=np.array([0], dtype=np.int64),
        teacher_occupancy_probabilities=scores,
    )
    with pytest.raises(ValueError, match="stable public occupancy argmax"):
        load_bayesian_demonstrations(inconsistent_path)


def test_bayesian_metadata_rejects_missing_public_only_certificate(tmp_path: Path) -> None:
    data_path = tmp_path / "demonstrations.npz"
    data_path.write_bytes(b"placeholder")
    data_path.with_name("dataset.json").write_text(
        json.dumps({"schema_version": BAYESIAN_DEMONSTRATION_SCHEMA_VERSION}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not certify public fields"):
        load_bayesian_demonstration_metadata(data_path)
