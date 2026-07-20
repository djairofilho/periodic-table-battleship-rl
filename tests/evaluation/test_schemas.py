from __future__ import annotations

import json
from pathlib import Path

import pytest

from periodic_table_battleship_rl.evaluation import (
    EpisodeManifest,
    EpisodeResult,
    HardwareMetadata,
    PlacementResult,
    RunConfig,
    RunManifest,
    SoftwareMetadata,
    canonical_json,
    sha256_file,
)


def _run_config() -> RunConfig:
    return RunConfig(
        run_id="attack-battleship-test-001",
        experiment="attack",
        scenario="battleship",
        environment_version="topology-v1",
        policy_id="random-mask-v1",
        split="test",
        seeds=(101, 202),
        episodes_per_seed=3,
        parameters={"max_total_attempts_multiplier": 2, "reward": "efficiency-v0"},
    )


def test_run_manifest_serializes_deterministically() -> None:
    config = _run_config()
    manifest = RunManifest(
        config=config,
        git_commit="a" * 40,
        uv_lock_sha256="b" * 64,
        software=SoftwareMetadata(
            python_version="3.11.9",
            platform="Windows-11",
            dependencies={"numpy": "2.0.0", "gymnasium": "1.0.0"},
        ),
        hardware=HardwareMetadata(
            machine="AMD64", processor="test CPU", cpu_count=8, accelerator=None
        ),
        episodes=EpisodeManifest(
            run_id=config.run_id,
            episode_ids=("attack-202-000", "attack-101-000"),
        ),
    )

    serialized = canonical_json(manifest)

    assert serialized == canonical_json(manifest)
    assert json.loads(serialized)["config"]["parameters"] == {
        "max_total_attempts_multiplier": 2,
        "reward": "efficiency-v0",
    }
    assert serialized.index('"gymnasium"') < serialized.index('"numpy"')


def test_records_keep_episode_ids_and_public_metrics() -> None:
    attack = EpisodeResult(
        episode_id="attack-101-000",
        run_id="run-001",
        seed=101,
        scenario="battleship",
        valid_cells=100,
        valid_shots=75,
        invalid_attempts=0,
        hit_segments=17,
        sunk_ship_lengths=(2, 3, 3, 4, 5),
        won=True,
        truncated=False,
        auc_discovery=0.62,
        first_hit_shot=3,
        first_sunk_shot=14,
    )
    placement = PlacementResult(
        episode_id="placement-101-000",
        run_id="run-002",
        seed=101,
        scenario="periodic-table-battleship",
        attacker_id="hunt-target-v1",
        attacker_seed=77,
        placement_actions=(2, 185, 7, 204, 20),
        valid_cells=118,
        valid_shots_to_sink=100,
        hit_segments=17,
        sunk_ship_lengths=(2, 3, 3, 4, 5),
        auc_discovery=0.71,
        first_hit_shot=2,
        first_sunk_shot=16,
        all_sunk_shot=100,
    )

    assert attack.to_dict()["episode_id"] == "attack-101-000"
    assert placement.to_dict()["attacker_id"] == "hunt-target-v1"
    assert json.loads(canonical_json(placement))["placement_actions"] == [2, 185, 7, 204, 20]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: RunConfig(
                run_id="r",
                experiment="attack",
                scenario="battleship",
                environment_version="topology-v1",
                policy_id="random",
                split="test",
                seeds=(1, 1),
                episodes_per_seed=1,
            ),
            "duplicates",
        ),
        (
            lambda: EpisodeResult(
                episode_id="episode",
                run_id="r",
                seed=1,
                scenario="battleship",
                valid_cells=100,
                valid_shots=1,
                invalid_attempts=0,
                hit_segments=16,
                sunk_ship_lengths=(),
                won=True,
                truncated=False,
                auc_discovery=0.0,
            ),
            "all 17",
        ),
        (
            lambda: PlacementResult(
                episode_id="episode",
                run_id="r",
                seed=1,
                scenario="battleship",
                attacker_id="random",
                attacker_seed=1,
                placement_actions=(0, 1, 2, 3),
                valid_cells=100,
                valid_shots_to_sink=0,
                hit_segments=0,
                sunk_ship_lengths=(),
                auc_discovery=0.0,
            ),
            "one action per ship",
        ),
    ],
)
def test_invalid_records_are_rejected(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_sha256_file_is_content_hash(tmp_path: Path) -> None:
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock data\n")

    assert sha256_file(lock_file) == (
        "14a6f98590049c93d0033178d6aa899669b4ac55393fc9628e05facc76d2be72"
    )
