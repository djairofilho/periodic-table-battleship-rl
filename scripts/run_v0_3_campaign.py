"""Run the reproducible v0.3 fixed-opponent RL campaign.

The full protocol is intentionally sequential: it tunes an attacker on
train/validation data, trains five final attacker seeds per topology, then
uses the best validation checkpoint as the frozen PPO component for placement.
Only held-out test seeds enter the published comparisons.  It writes private
checkpoints below ``.local-runs`` and public runs and visual artifacts below
``runs`` and ``artifacts``.

Run the complete CPU campaign with::

    uv run --extra train --extra visual python scripts/run_v0_3_campaign.py

Use ``--smoke`` to exercise the same wiring with a much smaller schedule.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Iterable, Sequence

import numpy as np

from periodic_table_battleship_rl.analysis.statistics import bootstrap_mean_interval
from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.evaluation import RunConfig
from periodic_table_battleship_rl.evaluation.schemas import EpisodeResult, PlacementResult
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.experiments import (
    HUNT_TARGET_POLICY_ID,
    RANDOM_MASKED_POLICY_ID,
    AttackHyperparameterCandidate,
    AttackTuningConfig,
    PpoAttackTuningExecutor,
    persist_attack_tuning_result,
    run_attack_baseline,
    run_attack_hyperparameter_search,
    run_placement_baseline_evaluation,
    run_placement_evaluation,
    run_ppo_attack_evaluation,
)
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.experiments.placement_evaluation import (
    PLACEMENT_ENVIRONMENT_VERSION,
)
from periodic_table_battleship_rl.placement import (
    DispersionPlacementPolicy,
    FrozenDefensiveMixture,
    FrozenPPOEvaluator,
    HuntTargetEvaluator,
    HuntTargetResistantPlacementPolicy,
    RandomLegalPlacementPolicy,
    RandomMaskedEvaluator,
)
from periodic_table_battleship_rl.rendering.attack import AttackTraceRecorder
from periodic_table_battleship_rl.topology import (
    BATTLESHIP,
    DENSE_118,
    PERIODIC_TABLE_BATTLESHIP,
    Topology,
)
from periodic_table_battleship_rl.training import (
    ATTACK_POLICY_ID,
    PLACEMENT_POLICY_ID,
    AttackCheckpointArtifact,
    AttackTrainingArtifact,
    AttackTrainingConfig,
    AttackValidationConfig,
    MaskableAttackPolicy,
    MaskablePlacementPolicy,
    PlacementTrainingArtifact,
    PlacementTrainingConfig,
    load_attack_policy,
    load_placement_policy,
    load_training_metadata,
    train_attack_policy,
    train_placement_policy,
)
from periodic_table_battleship_rl.visualization import (
    LearningCurvePoint,
    plot_attack_comparison,
    plot_learning_curve,
    plot_placement_comparison,
    plot_placement_segment_heatmap,
    write_attack_results_csv,
    write_attack_summary_markdown,
    write_attack_trace_gif,
    write_learning_progress_gif,
    write_placement_results_csv,
    write_placement_summary_markdown,
    write_placement_trace_gif,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "v0.3-fixed-suite"
ATTACK_TOPOLOGIES = (BATTLESHIP, DENSE_118, PERIODIC_TABLE_BATTLESHIP)
PLACEMENT_TOPOLOGIES = (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP)


@dataclass(frozen=True, slots=True)
class CampaignSchedule:
    """All public randomization and training-budget choices for one campaign."""

    hpo_train_seeds: tuple[int, ...]
    hpo_validation_seeds: tuple[int, ...]
    final_train_seeds: tuple[int, ...]
    final_validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    hpo_timesteps: int
    final_timesteps: int
    checkpoint_steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AttackFinalModel:
    """A final attack seed plus the checkpoint selected on validation only."""

    artifact: AttackTrainingArtifact
    checkpoint: AttackCheckpointArtifact
    policy: MaskableAttackPolicy

    @property
    def seed(self) -> int:
        return self.artifact.seed


@dataclass(frozen=True, slots=True)
class PlacementFinalModel:
    """One independently trained placement PPO policy."""

    artifact: PlacementTrainingArtifact
    policy: MaskablePlacementPolicy

    @property
    def seed(self) -> int:
        return self.artifact.seed


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the protocol with reduced seeds, steps, and held-out episodes",
    )
    return parser.parse_args()


def _schedule(smoke: bool) -> CampaignSchedule:
    if smoke:
        return CampaignSchedule(
            hpo_train_seeds=(1101,),
            hpo_validation_seeds=(2101, 2102),
            final_train_seeds=(3101,),
            final_validation_seeds=(4101, 4102),
            test_seeds=(5101, 5102, 5103),
            hpo_timesteps=512,
            final_timesteps=1_024,
            checkpoint_steps=(512, 1_024),
        )
    return CampaignSchedule(
        hpo_train_seeds=(1101, 1102, 1103),
        hpo_validation_seeds=tuple(range(2101, 2111)),
        final_train_seeds=(3101, 3102, 3103, 3104, 3105),
        final_validation_seeds=tuple(range(4101, 4111)),
        test_seeds=tuple(range(5101, 5201)),
        hpo_timesteps=20_000,
        final_timesteps=50_000,
        checkpoint_steps=(10_000, 20_000, 30_000, 40_000, 50_000),
    )


def _configure_torch() -> None:
    """Keep the single-environment campaign from oversubscribing CPU cores."""

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _attack_candidates(schedule: CampaignSchedule) -> tuple[AttackHyperparameterCandidate, ...]:
    """Return the narrow, pre-declared PPO search grid used by every topology."""

    return (
        AttackHyperparameterCandidate(
            candidate_id="conservative-lr",
            total_timesteps=schedule.hpo_timesteps,
            n_steps=256,
            batch_size=64,
            learning_rate=1e-4,
            device="cpu",
        ),
        AttackHyperparameterCandidate(
            candidate_id="standard",
            total_timesteps=schedule.hpo_timesteps,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            device="cpu",
        ),
        AttackHyperparameterCandidate(
            candidate_id="fast-lr",
            total_timesteps=schedule.hpo_timesteps,
            n_steps=256,
            batch_size=64,
            learning_rate=1e-3,
            device="cpu",
        ),
    )


def _attack_run_config(
    run_id: str,
    topology: Topology,
    split: str,
    seeds: tuple[int, ...],
    *,
    parameters: dict[str, object],
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        experiment="attack",
        scenario=topology.name,
        environment_version=ENVIRONMENT_VERSION,
        policy_id=ATTACK_POLICY_ID,
        split=split,
        seeds=seeds,
        episodes_per_seed=1,
        parameters={"campaign": CAMPAIGN_ID, **parameters},
    )


def _attack_baseline_config(
    run_id: str, topology: Topology, policy_id: str, schedule: CampaignSchedule
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        experiment="attack",
        scenario=topology.name,
        environment_version=ENVIRONMENT_VERSION,
        policy_id=policy_id,
        split="test",
        seeds=schedule.test_seeds,
        episodes_per_seed=1,
        parameters={
            "campaign": CAMPAIGN_ID,
            "fleet_sampler": "random_legal-v1",
            "comparison_split": "blind-test",
        },
    )


def _placement_config(
    run_id: str,
    topology: Topology,
    split: str,
    seeds: tuple[int, ...],
    policy_id: str,
    *,
    parameters: dict[str, object],
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        experiment="placement",
        scenario=topology.name,
        environment_version=PLACEMENT_ENVIRONMENT_VERSION,
        policy_id=policy_id,
        split=split,
        seeds=seeds,
        episodes_per_seed=1,
        parameters={"campaign": CAMPAIGN_ID, **parameters},
    )


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    if not array.size:
        raise ValueError("cannot average an empty result collection")
    return float(array.mean())


def _select_attack_checkpoint(artifact: AttackTrainingArtifact) -> AttackCheckpointArtifact:
    if not artifact.checkpoints:
        raise ValueError("final attack training must contain validation checkpoints")
    return min(
        artifact.checkpoints,
        key=lambda checkpoint: (checkpoint.mean_valid_shots, checkpoint.training_step),
    )


def _tune_attack(
    topology: Topology,
    schedule: CampaignSchedule,
    *,
    local_models: Path,
    runs: Path,
    commit: str,
) -> AttackHyperparameterCandidate:
    config = AttackTuningConfig(
        search_id=f"{CAMPAIGN_ID}-attack-hpo-{topology.name}",
        scenario=topology.name,
        training_seeds=schedule.hpo_train_seeds,
        validation_seeds=schedule.hpo_validation_seeds,
        validation_episodes_per_seed=1,
    )
    executor = PpoAttackTuningExecutor(
        topology=topology,
        checkpoint_directory=local_models / "attack-hpo" / topology.name,
        validation_directory=runs / "attack" / topology.name / "hpo-validation",
        git_commit=commit,
        uv_lock_path=ROOT / "uv.lock",
    )
    result = run_attack_hyperparameter_search(
        config,
        topology,
        _attack_candidates(schedule),
        executor,
    )
    persist_attack_tuning_result(runs / "attack" / topology.name / "hpo", result)
    return result.selected_candidate


def _train_final_attack_models(
    topology: Topology,
    candidate: AttackHyperparameterCandidate,
    schedule: CampaignSchedule,
    *,
    local_models: Path,
) -> tuple[AttackFinalModel, ...]:
    validation = AttackValidationConfig(
        seeds=schedule.final_validation_seeds,
        checkpoint_steps=schedule.checkpoint_steps,
        episodes_per_seed=1,
    )
    models: list[AttackFinalModel] = []
    for seed in schedule.final_train_seeds:
        run_id = f"{CAMPAIGN_ID}-attack-{topology.name}-seed-{seed}"
        artifact = train_attack_policy(
            topology,
            AttackTrainingConfig(
                run_id=run_id,
                seed=seed,
                total_timesteps=schedule.final_timesteps,
                checkpoint_directory=local_models / "attack-final" / topology.name,
                n_steps=candidate.n_steps,
                batch_size=candidate.batch_size,
                learning_rate=candidate.learning_rate,
                device=candidate.device,
            ),
            validation=validation,
        )
        checkpoint = _select_attack_checkpoint(artifact)
        models.append(
            AttackFinalModel(
                artifact=artifact,
                checkpoint=checkpoint,
                policy=load_attack_policy(checkpoint.checkpoint_path, device=candidate.device),
            )
        )
    return tuple(models)


def _evaluate_attack(
    topology: Topology,
    models: Sequence[AttackFinalModel],
    schedule: CampaignSchedule,
    *,
    runs: Path,
    commit: str,
) -> tuple[list[EpisodeResult], dict[str, str], dict[str, object]]:
    results: list[EpisodeResult] = []
    labels: dict[str, str] = {}
    selected: list[dict[str, int]] = []
    for model in models:
        run_id = f"{CAMPAIGN_ID}-attack-{topology.name}-seed-{model.seed}-test"
        evaluation = run_ppo_attack_evaluation(
            _attack_run_config(
                run_id,
                topology,
                "test",
                schedule.test_seeds,
                parameters={
                    "training_seed": model.seed,
                    "selected_checkpoint_step": model.checkpoint.training_step,
                    "comparison_split": "blind-test",
                },
            ),
            topology,
            model.policy,
            runs / "attack" / topology.name / run_id,
            checkpoint_path=model.checkpoint.checkpoint_path,
            training_metadata_path=model.artifact.metadata_path,
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        results.extend(evaluation.results)
        labels[run_id] = "MaskablePPO (multi-seed)"
        selected.append({"seed": model.seed, "checkpoint_step": model.checkpoint.training_step})

    baseline_results: dict[str, tuple[EpisodeResult, ...]] = {}
    for policy_id, label in (
        (RANDOM_MASKED_POLICY_ID, "Random masked"),
        (HUNT_TARGET_POLICY_ID, "Hunt-target"),
    ):
        run_id = f"{CAMPAIGN_ID}-attack-{topology.name}-{policy_id}-test"
        baseline = run_attack_baseline(
            _attack_baseline_config(run_id, topology, policy_id, schedule),
            topology,
            runs / "attack" / topology.name / run_id,
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        results.extend(baseline.results)
        baseline_results[policy_id] = baseline.results
        labels[run_id] = label
    return results, labels, {
        "selected_checkpoints": selected,
        "baseline_results": baseline_results,
    }


def _frozen_mixture(topology: Topology, models: Sequence[AttackFinalModel]) -> FrozenDefensiveMixture:
    """Freeze the best final validation checkpoint into the defensive suite."""

    frozen = min(
        models,
        key=lambda model: (
            model.checkpoint.mean_valid_shots,
            model.seed,
            model.checkpoint.training_step,
        ),
    )
    ppo_evaluator = FrozenPPOEvaluator(
        policy=frozen.policy,
        topology=topology,
        training_metadata=load_training_metadata(frozen.artifact.metadata_path),
        checkpoint_id=(
            f"{CAMPAIGN_ID}-seed-{frozen.seed}-step-{frozen.checkpoint.training_step}"
        ),
    )
    return FrozenDefensiveMixture(
        evaluators=(
            RandomMaskedEvaluator(topology),
            HuntTargetEvaluator(topology),
            ppo_evaluator,
        ),
        weights=(1.0, 1.0, 1.0),
        evaluator_id=f"{CAMPAIGN_ID}-random-hunt-frozen-ppo",
    )


def _train_final_placement_models(
    topology: Topology,
    mixture: FrozenDefensiveMixture,
    schedule: CampaignSchedule,
    *,
    local_models: Path,
) -> tuple[PlacementFinalModel, ...]:
    models: list[PlacementFinalModel] = []
    for seed in schedule.final_train_seeds:
        run_id = f"{CAMPAIGN_ID}-placement-{topology.name}-seed-{seed}"
        artifact = train_placement_policy(
            topology,
            PlacementTrainingConfig(
                run_id=run_id,
                seed=seed,
                total_timesteps=schedule.final_timesteps,
                checkpoint_directory=local_models / "placement-final" / topology.name,
                n_steps=256,
                batch_size=64,
                learning_rate=3e-4,
                device="cpu",
            ),
            defensive_mixture=mixture,
        )
        models.append(
            PlacementFinalModel(
                artifact=artifact,
                policy=load_placement_policy(artifact.checkpoint_path, device="cpu"),
            )
        )
    return tuple(models)


def _evaluate_placement(
    topology: Topology,
    mixture: FrozenDefensiveMixture,
    models: Sequence[PlacementFinalModel],
    schedule: CampaignSchedule,
    *,
    runs: Path,
    commit: str,
) -> tuple[list[PlacementResult], dict[str, str], dict[str, object]]:
    results: list[PlacementResult] = []
    labels: dict[str, str] = {}
    for model in models:
        validation_run_id = (
            f"{CAMPAIGN_ID}-placement-{topology.name}-seed-{model.seed}-validation"
        )
        run_placement_evaluation(
            _placement_config(
                validation_run_id,
                topology,
                "validation",
                schedule.final_validation_seeds,
                PLACEMENT_POLICY_ID,
                parameters={"training_seed": model.seed, "selection": "validation-only"},
            ),
            topology,
            model.policy,
            mixture,
            runs / "placement" / topology.name / validation_run_id,
            checkpoint_path=model.artifact.checkpoint_path,
            training_metadata_path=model.artifact.metadata_path,
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        test_run_id = f"{CAMPAIGN_ID}-placement-{topology.name}-seed-{model.seed}-test"
        evaluation = run_placement_evaluation(
            _placement_config(
                test_run_id,
                topology,
                "test",
                schedule.test_seeds,
                PLACEMENT_POLICY_ID,
                parameters={"training_seed": model.seed, "comparison_split": "blind-test"},
            ),
            topology,
            model.policy,
            mixture,
            runs / "placement" / topology.name / test_run_id,
            checkpoint_path=model.artifact.checkpoint_path,
            training_metadata_path=model.artifact.metadata_path,
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        results.extend(evaluation.results)
        labels[test_run_id] = "MaskablePPO placement (multi-seed)"

    baselines = (
        RandomLegalPlacementPolicy(topology),
        DispersionPlacementPolicy(topology),
        HuntTargetResistantPlacementPolicy(topology),
    )
    baseline_results: dict[str, tuple[PlacementResult, ...]] = {}
    for policy in baselines:
        run_id = f"{CAMPAIGN_ID}-placement-{topology.name}-{policy.policy_id}-test"
        evaluation = run_placement_baseline_evaluation(
            _placement_config(
                run_id,
                topology,
                "test",
                schedule.test_seeds,
                policy.policy_id,
                parameters={"comparison_split": "blind-test"},
            ),
            topology,
            policy,
            mixture,
            runs / "placement" / topology.name / run_id,
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        results.extend(evaluation.results)
        baseline_results[policy.policy_id] = evaluation.results
        labels[run_id] = policy.policy_id
    return results, labels, {"baseline_results": baseline_results}


def _by_seed_mean(
    results: Sequence[EpisodeResult | PlacementResult],
    metric: str,
    *,
    attacker_id: str | None = None,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for result in results:
        if attacker_id is not None and getattr(result, "attacker_id", None) != attacker_id:
            continue
        grouped.setdefault(result.seed, []).append(float(getattr(result, metric)))
    return {seed: _mean(values) for seed, values in sorted(grouped.items())}


def _paired_bootstrap(
    candidate: Sequence[EpisodeResult | PlacementResult],
    reference: Sequence[EpisodeResult | PlacementResult],
    metric: str,
    *,
    attacker_id: str | None = None,
) -> dict[str, float | int]:
    candidate_by_seed = _by_seed_mean(candidate, metric, attacker_id=attacker_id)
    reference_by_seed = _by_seed_mean(reference, metric, attacker_id=attacker_id)
    if candidate_by_seed.keys() != reference_by_seed.keys():
        raise ValueError("candidate and reference must use the same held-out seeds")
    differences = [
        candidate_by_seed[seed] - reference_by_seed[seed]
        for seed in sorted(candidate_by_seed)
    ]
    interval = bootstrap_mean_interval(
        differences,
        rng=np.random.default_rng(20260720),
    )
    return {
        "candidate_minus_reference_mean": interval.mean,
        "lower_95": interval.lower,
        "upper_95": interval.upper,
        "resamples": interval.resamples,
        "seed_count": len(differences),
    }


def _attack_trace(topology: Topology, policy: MaskableAttackPolicy, seed: int):
    environment = AttackEnv(topology)
    observation, _ = environment.reset(seed=seed)
    recorder = AttackTraceRecorder(topology, observation)
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.select_action(observation, environment.action_masks(), deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        recorder.record(
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            observation=observation,
        )
    return recorder.build()


def _curve_points(models: Sequence[AttackFinalModel]) -> tuple[LearningCurvePoint, ...]:
    return tuple(
        LearningCurvePoint(
            seed=model.seed,
            stage=checkpoint.training_step,
            value=checkpoint.mean_valid_shots,
        )
        for model in models
        for checkpoint in model.artifact.checkpoints
    )


def _write_attack_curves(
    topology: Topology, models: Sequence[AttackFinalModel], figures: Path
) -> None:
    points = _curve_points(models)
    name = f"attack-{topology.name}-learning-curve"
    plot_learning_curve(
        points,
        figures / f"{name}.png",
        title=f"PPO attack validation curve: {topology.name}",
        x_label="Training timesteps",
        y_label="Mean valid shots (lower is better)",
    )
    write_learning_progress_gif(
        points,
        figures / f"{name}.gif",
        title=f"PPO attack validation progression: {topology.name}",
        x_label="Training timesteps",
        y_label="Mean valid shots (lower is better)",
    )


def main() -> None:
    arguments = _parse_arguments()
    schedule = _schedule(arguments.smoke)
    suffix = "-smoke" if arguments.smoke else ""
    local_models = ROOT / ".local-runs" / f"{CAMPAIGN_ID}{suffix}"
    runs = ROOT / "runs" / f"{CAMPAIGN_ID}{suffix}"
    artifacts = ROOT / "artifacts" / f"{CAMPAIGN_ID}{suffix}"
    tables = artifacts / "tables"
    figures = artifacts / "figures"
    _configure_torch()
    commit = _commit()

    attack_models: dict[str, tuple[AttackFinalModel, ...]] = {}
    attack_results: list[EpisodeResult] = []
    attack_labels: dict[str, str] = {}
    attack_statistics: dict[str, object] = {}
    for topology in ATTACK_TOPOLOGIES:
        candidate = _tune_attack(topology, schedule, local_models=local_models, runs=runs, commit=commit)
        models = _train_final_attack_models(
            topology, candidate, schedule, local_models=local_models
        )
        attack_models[topology.name] = models
        topology_results, labels, extra = _evaluate_attack(
            topology, models, schedule, runs=runs, commit=commit
        )
        attack_results.extend(topology_results)
        attack_labels.update(labels)
        attack_statistics[topology.name] = {
            "hpo_selected_candidate": candidate.to_dict(),
            "selected_final_checkpoints": extra["selected_checkpoints"],
            "ppo_minus_hunt_valid_shots": _paired_bootstrap(
                [result for result in topology_results if result.run_id in labels and labels[result.run_id] == "MaskablePPO (multi-seed)"],
                extra["baseline_results"][HUNT_TARGET_POLICY_ID],
                "valid_shots",
            ),
        }
        _write_attack_curves(topology, models, figures)

    placement_results: list[PlacementResult] = []
    placement_labels: dict[str, str] = {}
    placement_statistics: dict[str, object] = {}
    placement_evaluations: dict[str, list[PlacementResult]] = {}
    for topology in PLACEMENT_TOPOLOGIES:
        mixture = _frozen_mixture(topology, attack_models[topology.name])
        models = _train_final_placement_models(
            topology, mixture, schedule, local_models=local_models
        )
        topology_results, labels, extra = _evaluate_placement(
            topology,
            mixture,
            models,
            schedule,
            runs=runs,
            commit=commit,
        )
        placement_results.extend(topology_results)
        placement_evaluations[topology.name] = topology_results
        placement_labels.update(labels)
        ppo_results = [
            result
            for result in topology_results
            if labels.get(result.run_id) == "MaskablePPO placement (multi-seed)"
        ]
        placement_statistics[topology.name] = {
            "defensive_suite": {
                "components": list(mixture.component_ids),
                "weights": list(mixture.weights),
            },
            "ppo_minus_baselines_mixture_valid_shots": {
                policy_id: _paired_bootstrap(
                    ppo_results,
                    baseline_results,
                    "valid_shots_to_sink",
                    attacker_id=mixture.evaluator_id,
                )
                for policy_id, baseline_results in extra["baseline_results"].items()
            },
        }

    write_attack_results_csv(
        attack_results, tables / "attack-test-episodes.csv", policy_by_run=attack_labels
    )
    write_attack_summary_markdown(
        attack_results, tables / "attack-test-summary.md", policy_by_run=attack_labels
    )
    plot_attack_comparison(
        attack_results, figures / "attack-test-comparison.png", policy_by_run=attack_labels
    )
    write_placement_results_csv(
        placement_results,
        tables / "placement-test-episodes.csv",
        policy_by_run=placement_labels,
    )
    write_placement_summary_markdown(
        placement_results,
        tables / "placement-test-summary.md",
        policy_by_run=placement_labels,
    )
    plot_placement_comparison(
        placement_results,
        figures / "placement-test-comparison.png",
        policy_by_run=placement_labels,
    )
    for topology in PLACEMENT_TOPOLOGIES:
        ppo_results = [
            result
            for result in placement_evaluations[topology.name]
            if placement_labels.get(result.run_id) == "MaskablePPO placement (multi-seed)"
        ]
        plot_placement_segment_heatmap(
            ppo_results,
            topology,
            figures / f"placement-{topology.name}-ppo-heatmap.png",
        )

    periodic_models = attack_models[PERIODIC_TABLE_BATTLESHIP.name]
    frozen_periodic = min(
        periodic_models,
        key=lambda model: (model.checkpoint.mean_valid_shots, model.seed),
    )
    write_attack_trace_gif(
        _attack_trace(PERIODIC_TABLE_BATTLESHIP, frozen_periodic.policy, schedule.test_seeds[0]),
        figures / "periodic-ppo-attack.gif",
    )
    periodic_mixture_result = next(
        result
        for result in placement_evaluations[PERIODIC_TABLE_BATTLESHIP.name]
        if (
            placement_labels.get(result.run_id) == "MaskablePPO placement (multi-seed)"
            and result.attacker_id == f"{CAMPAIGN_ID}-random-hunt-frozen-ppo"
        )
    )
    write_placement_trace_gif(
        periodic_mixture_result,
        PERIODIC_TABLE_BATTLESHIP,
        figures / "periodic-ppo-placement.gif",
    )

    report = {
        "campaign": CAMPAIGN_ID,
        "mode": "smoke" if arguments.smoke else "full",
        "git_commit": commit,
        "protocol": {
            "hpo": {
                "candidates": [candidate.to_dict() for candidate in _attack_candidates(schedule)],
                "training_seeds": list(schedule.hpo_train_seeds),
                "validation_seeds": list(schedule.hpo_validation_seeds),
            },
            "final_training_seeds": list(schedule.final_train_seeds),
            "final_validation_seeds": list(schedule.final_validation_seeds),
            "test_seeds": list(schedule.test_seeds),
            "final_checkpoint_steps": list(schedule.checkpoint_steps),
            "cpu_threads": 1,
        },
        "attack": attack_statistics,
        "placement": placement_statistics,
        "artifact_paths": {
            "tables": str(tables.relative_to(ROOT)),
            "figures": str(figures.relative_to(ROOT)),
        },
    }
    write_json_atomic(artifacts / "campaign-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
