"""Validation guardrails for public-history belief planner runs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from periodic_table_battleship_rl.evaluation import RunConfig
from periodic_table_battleship_rl.experiments import (
    BELIEF_PROBABILITY_POLICY_ID,
    run_belief_planner_evaluation,
)
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.topology import BATTLESHIP


ROOT = Path(__file__).resolve().parents[2]


def _config(split: str) -> RunConfig:
    return RunConfig(
        run_id=f"belief-{split}",
        experiment="attack",
        scenario=BATTLESHIP.name,
        environment_version=ENVIRONMENT_VERSION,
        policy_id=BELIEF_PROBABILITY_POLICY_ID,
        split=split,
        seeds=(8611,),
        episodes_per_seed=1,
        parameters={"promotion_eligible": False},
    )


def test_belief_pilot_persists_public_run_and_sampler_diagnostics(tmp_path: Path) -> None:
    evaluation = run_belief_planner_evaluation(
        _config("validation"),
        BATTLESHIP,
        tmp_path,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip(),
        uv_lock_path=ROOT / "uv.lock",
        sample_count=8,
    )

    assert evaluation.results[0].won
    assert evaluation.summary["belief_sampler"]["posterior_exact"] is False
    assert evaluation.summary["belief_sampler"]["decision_count"] > 1
    assert evaluation.persisted.manifest_path.exists()
    assert evaluation.persisted.episodes_path.exists()


def test_belief_pilot_refuses_to_consume_blind_test_seed_inventory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="blind test"):
        run_belief_planner_evaluation(
            _config("test"),
            BATTLESHIP,
            tmp_path,
            git_commit="a" * 40,
            uv_lock_path=ROOT / "uv.lock",
            sample_count=8,
        )
