"""Tests for train/validation-only attack hyperparameter selection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from periodic_table_battleship_rl.evaluation.schemas import EpisodeResult
from periodic_table_battleship_rl.experiments.attack_tuning import (
    AttackHyperparameterCandidate,
    AttackTuningConfig,
    AttackTuningTrial,
    PpoAttackTuningExecutor,
    persist_attack_tuning_result,
    run_attack_hyperparameter_search,
)
from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training.attack import AttackTrainingArtifact, MaskableAttackPolicy


def _config() -> AttackTuningConfig:
    return AttackTuningConfig(
        search_id="attack-v0.3",
        scenario="battleship",
        training_seeds=(3, 5),
        validation_seeds=(11, 13),
        validation_episodes_per_seed=2,
    )


def _candidate(candidate_id: str) -> AttackHyperparameterCandidate:
    return AttackHyperparameterCandidate(candidate_id=candidate_id, total_timesteps=256)


def _result(
    *,
    run_id: str,
    seed: int,
    episode_index: int,
    valid_shots: int,
) -> EpisodeResult:
    return EpisodeResult(
        episode_id=f"{run_id}-seed-{seed}-episode-{episode_index}",
        run_id=run_id,
        seed=seed,
        scenario="battleship",
        valid_cells=100,
        valid_shots=valid_shots,
        invalid_attempts=0,
        hit_segments=17,
        sunk_ship_lengths=(5, 4, 3, 3, 2),
        won=True,
        truncated=False,
        auc_discovery=0.5,
        first_hit_shot=1,
        first_sunk_shot=2,
    )


def _trial(request, shots: int) -> AttackTuningTrial:
    run_id = f"{request.candidate.candidate_id}-{request.training_seed}-validation"
    return AttackTuningTrial(
        candidate_id=request.candidate.candidate_id,
        training_seed=request.training_seed,
        training_run_id=f"{run_id}-train",
        validation_run_id=run_id,
        results=tuple(
            _result(
                run_id=run_id,
                seed=seed,
                episode_index=episode_index,
                valid_shots=shots,
            )
            for seed in request.config.validation_seeds
            for episode_index in range(request.config.validation_episodes_per_seed)
        ),
    )


def test_search_is_order_independent_and_selects_lowest_validation_shots(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int, tuple[int, ...]]] = []

    def executor(request):
        calls.append(
            (
                request.candidate.candidate_id,
                request.training_seed,
                request.config.validation_seeds,
            )
        )
        return _trial(request, 42 if request.candidate.candidate_id == "fast" else 57)

    result = run_attack_hyperparameter_search(
        _config(),
        BATTLESHIP,
        (_candidate("slow"), _candidate("fast")),
        executor,
    )

    assert result.selected_candidate_id == "fast"
    assert [score.candidate_id for score in result.ranking] == ["fast", "slow"]
    assert calls == [
        ("fast", 3, (11, 13)),
        ("fast", 5, (11, 13)),
        ("slow", 3, (11, 13)),
        ("slow", 5, (11, 13)),
    ]
    path = persist_attack_tuning_result(tmp_path, result)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["selected_candidate_id"] == "fast"
    assert stored["config"]["validation_seeds"] == [11, 13]
    assert "test" not in stored["config"]


def test_tied_validation_scores_use_candidate_id_as_deterministic_tie_break() -> None:
    result = run_attack_hyperparameter_search(
        _config(),
        BATTLESHIP,
        (_candidate("zulu"), _candidate("alpha")),
        lambda request: _trial(request, 50),
    )

    assert result.selected_candidate_id == "alpha"
    assert [score.candidate_id for score in result.ranking] == ["alpha", "zulu"]


def test_search_rejects_results_outside_the_fixed_validation_schedule() -> None:
    def executor(request):
        trial = _trial(request, 50)
        bad_result = _result(
            run_id=trial.validation_run_id,
            seed=99,
            episode_index=0,
            valid_shots=50,
        )
        return AttackTuningTrial(
            candidate_id=trial.candidate_id,
            training_seed=trial.training_seed,
            training_run_id=trial.training_run_id,
            validation_run_id=trial.validation_run_id,
            results=(bad_result,) + trial.results[1:],
        )

    with pytest.raises(ValueError, match="outside the fixed schedule"):
        run_attack_hyperparameter_search(
            _config(), BATTLESHIP, (_candidate("candidate"),), executor
        )


def test_production_executor_builds_validation_only_run_with_injected_operations(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    checkpoint = tmp_path / "model.zip"
    metadata = tmp_path / "training.json"
    manifest = tmp_path / "manifest.json"
    episodes = tmp_path / "episodes.jsonl"

    def trainer(topology, training_config):
        captured["topology"] = topology
        captured["training_config"] = training_config
        return AttackTrainingArtifact(
            checkpoint_path=checkpoint,
            metadata_path=metadata,
            policy_id="maskable-ppo-v1",
            scenario="battleship",
            seed=training_config.seed,
        )

    def policy_loader(path, *, device):
        captured["loaded"] = (path, device)
        return MaskableAttackPolicy(model=object())

    def evaluator(config, topology, policy, directory, **kwargs):
        captured["evaluation"] = (config, topology, policy, directory, kwargs)
        results = tuple(
            _result(
                run_id=config.run_id,
                seed=seed,
                episode_index=episode_index,
                valid_shots=60,
            )
            for seed in config.seeds
            for episode_index in range(config.episodes_per_seed)
        )
        return SimpleNamespace(
            results=results,
            persisted=SimpleNamespace(manifest_path=manifest, episodes_path=episodes),
        )

    executor = PpoAttackTuningExecutor(
        topology=BATTLESHIP,
        checkpoint_directory=tmp_path / "checkpoints",
        validation_directory=tmp_path / "validation",
        git_commit="a" * 40,
        uv_lock_path=tmp_path / "uv.lock",
        trainer=trainer,
        policy_loader=policy_loader,
        evaluator=evaluator,
    )
    result = run_attack_hyperparameter_search(
        _config(), BATTLESHIP, (_candidate("candidate"),), executor
    )

    training_config = captured["training_config"]
    validation_config = captured["evaluation"][0]
    assert training_config.total_timesteps == 256
    assert validation_config.split == "validation"
    assert validation_config.seeds == (11, 13)
    assert validation_config.episodes_per_seed == 2
    assert result.trials[0].provenance["checkpoint_path"] == str(checkpoint)
