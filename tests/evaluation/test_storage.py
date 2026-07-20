from __future__ import annotations

import json
from pathlib import Path

import pytest

from periodic_table_battleship_rl.evaluation.schemas import (
    EpisodeManifest,
    EpisodeResult,
    HardwareMetadata,
    RunConfig,
    RunManifest,
    SoftwareMetadata,
    sha256_file,
)
from periodic_table_battleship_rl.evaluation.storage import (
    load_run,
    persist_run,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)


def _manifest() -> RunManifest:
    config = RunConfig(
        run_id="attack-battleship-test-001",
        experiment="attack",
        scenario="battleship",
        environment_version="attack-env-v1",
        policy_id="random-mask-v1",
        split="test",
        seeds=(101,),
        episodes_per_seed=2,
    )
    return RunManifest(
        config=config,
        git_commit="a" * 40,
        uv_lock_sha256="b" * 64,
        software=SoftwareMetadata(python_version="3.11.9", platform="Windows-11"),
        hardware=HardwareMetadata(machine="AMD64", processor="test CPU", cpu_count=8),
        episodes=EpisodeManifest(
            run_id=config.run_id,
            episode_ids=("attack-101-000", "attack-101-001"),
        ),
    )


def _results() -> tuple[EpisodeResult, EpisodeResult]:
    common = {
        "run_id": "attack-battleship-test-001",
        "seed": 101,
        "scenario": "battleship",
        "valid_cells": 100,
        "invalid_attempts": 0,
        "hit_segments": 17,
        "sunk_ship_lengths": (2, 3, 3, 4, 5),
        "won": True,
        "truncated": False,
        "auc_discovery": 0.71,
        "first_hit_shot": 2,
        "first_sunk_shot": 13,
    }
    return (
        EpisodeResult(episode_id="attack-101-000", valid_shots=58, **common),
        EpisodeResult(episode_id="attack-101-001", valid_shots=64, **common),
    )


def test_atomic_json_round_trip_is_utf8_and_replaces_previous_contents(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "summary.json"
    write_json_atomic(destination, {"nome": "s\u00edlica", "count": 1})
    write_json_atomic(destination, {"nome": "s\u00edlica", "count": 2})

    assert read_json(destination) == {"count": 2, "nome": "s\u00edlica"}
    assert destination.read_bytes() == b'{"count":2,"nome":"s\xc3\xadlica"}\n'


def test_atomic_jsonl_round_trip_preserves_record_boundaries(tmp_path: Path) -> None:
    destination = tmp_path / "episodes.jsonl"
    write_jsonl_atomic(destination, ({"episode_id": "a"}, {"episode_id": "b"}))

    assert read_jsonl(destination) == ({"episode_id": "a"}, {"episode_id": "b"})
    assert [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()] == [
        {"episode_id": "a"},
        {"episode_id": "b"},
    ]


def test_persisted_run_round_trips_public_records_and_hashes(tmp_path: Path) -> None:
    persisted = persist_run(tmp_path / "runs" / "run-001", _manifest(), _results())
    loaded = load_run(tmp_path / "runs" / "run-001")

    assert loaded.manifest["config"]["run_id"] == "attack-battleship-test-001"
    assert [episode["episode_id"] for episode in loaded.episodes] == [
        "attack-101-000",
        "attack-101-001",
    ]
    assert persisted.manifest_sha256 == sha256_file(persisted.manifest_path)
    assert persisted.episodes_sha256 == sha256_file(persisted.episodes_path)
    assert persisted.manifest_path.stat().st_mtime_ns >= persisted.episodes_path.stat().st_mtime_ns


def test_persist_run_rejects_results_that_do_not_match_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="match the manifest order"):
        persist_run(tmp_path / "run", _manifest(), tuple(reversed(_results())))
