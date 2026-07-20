"""Tests for public-only frozen masked-DQN evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from periodic_table_battleship_rl.evaluation import (
    HardwareMetadata,
    RunConfig,
    SoftwareMetadata,
)
from periodic_table_battleship_rl.experiments.dqn_evaluation import (
    run_dqn_attack_evaluation,
)
from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training.dqn import DQN_ATTACK_POLICY_ID


ROOT = Path(__file__).resolve().parents[2]


class _FirstLegalDqn:
    policy_id = DQN_ATTACK_POLICY_ID

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray, *, deterministic: bool = True
    ) -> int:
        del observation, deterministic
        return int(np.flatnonzero(action_mask)[0])


def test_dqn_evaluation_records_public_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    metadata = tmp_path / "training.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "dqn-attack-training-v1",
                "algorithm": "MaskedDQN",
                "policy_id": DQN_ATTACK_POLICY_ID,
                "scenario": "battleship",
                "environment": {
                    "class": "AttackEnv",
                    "action_mask_method": "action_masks",
                    "action_count": 180,
                    "valid_cells": 100,
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation = run_dqn_attack_evaluation(
        RunConfig(
            run_id="dqn-blind",
            experiment="attack",
            scenario="battleship",
            environment_version="attack-env-v1",
            policy_id=DQN_ATTACK_POLICY_ID,
            split="test",
            seeds=(17,),
            episodes_per_seed=1,
        ),
        BATTLESHIP,
        _FirstLegalDqn(),  # type: ignore[arg-type]
        tmp_path / "run",
        checkpoint_path=checkpoint,
        training_metadata_path=metadata,
        git_commit="a" * 40,
        uv_lock_path=ROOT / "uv.lock",
        software=SoftwareMetadata(python_version="3.11", platform="test"),
        hardware=HardwareMetadata(machine="test", processor="test", cpu_count=1),
    )

    assert evaluation.results[0].invalid_attempts == 0
    assert evaluation.manifest.config.parameters["evaluation_protocol"] == (
        "blind-public-observation-masked-dqn-v1"
    )
