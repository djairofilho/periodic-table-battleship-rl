"""Tests for deterministic, public initial attack-baseline benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from periodic_table_battleship_rl.evaluation.schemas import (
    HardwareMetadata,
    RunConfig,
    SoftwareMetadata,
)
from periodic_table_battleship_rl.experiments import (
    HUNT_TARGET_POLICY_ID,
    RANDOM_MASKED_POLICY_ID,
    run_attack_baseline,
    run_initial_attack_baselines,
)
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOFTWARE = SoftwareMetadata(python_version="3.11.9", platform="test-platform")
HARDWARE = HardwareMetadata(machine="test-machine", processor="test-cpu", cpu_count=8)


def _config(*, policy_id: str = HUNT_TARGET_POLICY_ID) -> RunConfig:
    return RunConfig(
        run_id=f"attack-battleship-{policy_id}",
        experiment="attack",
        scenario="battleship",
        environment_version="attack-env-v1",
        policy_id=policy_id,
        split="test",
        seeds=(31, 47),
        episodes_per_seed=2,
    )


def _run(tmp_path: Path, *, directory_name: str):
    return run_attack_baseline(
        _config(),
        BATTLESHIP,
        tmp_path / directory_name,
        git_commit="c" * 40,
        uv_lock_path=PROJECT_ROOT / "uv.lock",
        software=SOFTWARE,
        hardware=HARDWARE,
    )


def test_baseline_run_is_reproducible_and_persists_manifest_results_and_summary(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path, directory_name="first")
    second = _run(tmp_path, directory_name="second")

    assert [result.to_dict() for result in first.results] == [
        result.to_dict() for result in second.results
    ]
    assert first.manifest.to_dict() == second.manifest.to_dict()
    assert first.summary == second.summary
    assert first.persisted.episodes_path.read_bytes() == second.persisted.episodes_path.read_bytes()
    assert first.persisted.manifest_path.read_bytes() == second.persisted.manifest_path.read_bytes()
    assert first.summary_path.read_bytes() == second.summary_path.read_bytes()


def test_initial_runner_covers_two_baselines_and_two_topologies(tmp_path: Path) -> None:
    runs = run_initial_attack_baselines(
        tmp_path / "runs",
        seeds=(11,),
        episodes_per_seed=1,
        git_commit="d" * 40,
        uv_lock_path=PROJECT_ROOT / "uv.lock",
        software=SOFTWARE,
        hardware=HARDWARE,
    )

    assert {(run.manifest.config.scenario, run.manifest.config.policy_id) for run in runs} == {
        (topology.name, policy_id)
        for topology in (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP)
        for policy_id in (RANDOM_MASKED_POLICY_ID, HUNT_TARGET_POLICY_ID)
    }
    assert all(run.persisted.manifest_path.is_file() for run in runs)
    assert all(run.summary_path.is_file() for run in runs)


def test_summary_is_numeric_and_public_only(tmp_path: Path) -> None:
    run = _run(tmp_path, directory_name="public")
    payload = {
        "manifest": run.manifest.to_dict(),
        "episodes": [result.to_dict() for result in run.results],
        "summary": run.summary,
    }

    _assert_only_json_primitives(payload)
    serialized = json.dumps(payload, sort_keys=True)
    assert "_fleet" not in serialized
    assert "occupied_cells" not in serialized
    assert "ship_id_by_cell" not in serialized
    assert run.summary["seed_count"] == 2
    assert run.summary["episode_count"] == 4
    assert set(run.summary["per_seed"]) == {"31", "47"}
    assert run.summary["aggregate"]["invalid_attempts"]["mean"] == 0.0


def _assert_only_json_primitives(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            assert isinstance(key, str)
            _assert_only_json_primitives(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _assert_only_json_primitives(nested_value)
    else:
        assert isinstance(value, (str, int, float, bool)) or value is None
