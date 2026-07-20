"""Run the controlled v0.2 pilot campaign and publish public artifacts.

Run with ``uv run --extra train --extra visual python scripts/run_v0_2_campaign.py``.
The script intentionally trains one environment at a time.  It writes models
below the ignored ``.local-runs`` directory and public, reviewable evaluation
records below ``runs`` and ``artifacts``.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

from periodic_table_battleship_rl.analysis.statistics import (
    bootstrap_mean_interval,
    paired_difference_by_seed,
)
from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.evaluation import RunConfig
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.experiments import (
    HUNT_TARGET_POLICY_ID,
    RANDOM_MASKED_POLICY_ID,
    run_attack_baseline,
    run_placement_evaluation,
    run_ppo_attack_evaluation,
)
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.experiments.placement_evaluation import (
    PLACEMENT_ENVIRONMENT_VERSION,
)
from periodic_table_battleship_rl.placement import (
    FrozenDefensiveMixture,
    FrozenPPOEvaluator,
    HuntTargetEvaluator,
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
    AttackTrainingConfig,
    PlacementTrainingConfig,
    load_attack_policy,
    load_placement_policy,
    load_training_metadata,
    train_attack_policy,
    train_placement_policy,
)
from periodic_table_battleship_rl.visualization import (
    plot_attack_comparison,
    plot_placement_comparison,
    plot_placement_segment_heatmap,
    write_attack_results_csv,
    write_attack_summary_markdown,
    write_attack_trace_gif,
    write_placement_results_csv,
    write_placement_summary_markdown,
    write_placement_trace_gif,
)


ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODELS = ROOT / ".local-runs" / "v0.2-controlled"
RUNS = ROOT / "runs" / "v0.2-controlled"
ARTIFACTS = ROOT / "artifacts" / "v0.2-controlled"
TRAIN_SEEDS = (1101, 1102, 1103)
VALIDATION_SEEDS = (2101, 2102, 2103, 2104, 2105)
TEST_SEEDS = tuple(range(3101, 3121))
TOTAL_TIMESTEPS = 512
N_STEPS = 256
BATCH_SIZE = 64
TOPOLOGIES = (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP, DENSE_118)
PLACEMENT_TOPOLOGIES = (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP)


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _attack_config(run_id: str, topology: Topology, split: str, seeds: tuple[int, ...]) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        experiment="attack",
        scenario=topology.name,
        environment_version=ENVIRONMENT_VERSION,
        policy_id=ATTACK_POLICY_ID,
        split=split,
        seeds=seeds,
        episodes_per_seed=1,
        parameters={"campaign": "v0.2-controlled", "selection": split == "validation"},
    )


def _baseline_config(
    run_id: str, topology: Topology, policy_id: str
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        experiment="attack",
        scenario=topology.name,
        environment_version=ENVIRONMENT_VERSION,
        policy_id=policy_id,
        split="test",
        seeds=TEST_SEEDS,
        episodes_per_seed=1,
        parameters={"campaign": "v0.2-controlled", "fleet_sampler": "random_legal-v1"},
    )


def _placement_config(run_id: str, topology: Topology, split: str, seeds: tuple[int, ...]) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        experiment="placement",
        scenario=topology.name,
        environment_version=PLACEMENT_ENVIRONMENT_VERSION,
        policy_id=PLACEMENT_POLICY_ID,
        split=split,
        seeds=seeds,
        episodes_per_seed=1,
        parameters={"campaign": "v0.2-controlled", "selection": split == "validation"},
    )


def _mean(results: Iterable[object], attribute: str, *, attacker_id: str | None = None) -> float:
    values = [
        float(getattr(result, attribute))
        for result in results
        if attacker_id is None or getattr(result, "attacker_id") == attacker_id
    ]
    if not values:
        raise ValueError(f"no values for {attribute}")
    return float(np.mean(values))


def _attack_trace(topology: Topology, policy, seed: int):
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


def _aligned(results, label: str):
    return tuple(
        replace(result, episode_id=f"{label}-seed-{result.seed}-episode-000")
        for result in results
    )


def _train_attack(topology: Topology, commit: str):
    candidates = []
    for seed in TRAIN_SEEDS:
        run_id = f"v0.2-attack-{topology.name}-seed-{seed}"
        artifact = train_attack_policy(
            topology,
            AttackTrainingConfig(
                run_id=run_id,
                seed=seed,
                total_timesteps=TOTAL_TIMESTEPS,
                checkpoint_directory=LOCAL_MODELS / "attack" / topology.name,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
            ),
        )
        policy = load_attack_policy(artifact.checkpoint_path)
        validation = run_ppo_attack_evaluation(
            _attack_config(f"{run_id}-validation", topology, "validation", VALIDATION_SEEDS),
            topology,
            policy,
            RUNS / "attack" / topology.name / run_id / "validation",
            checkpoint_path=artifact.checkpoint_path,
            training_metadata_path=artifact.metadata_path,
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        candidates.append((artifact, policy, validation))
    return min(candidates, key=lambda candidate: _mean(candidate[2].results, "valid_shots"))


def _evaluate_attack(topology: Topology, chosen, commit: str):
    artifact, policy, _validation = chosen
    ppo = run_ppo_attack_evaluation(
        _attack_config(f"v0.2-attack-{topology.name}-ppo-test", topology, "test", TEST_SEEDS),
        topology,
        policy,
        RUNS / "attack" / topology.name / "ppo-test",
        checkpoint_path=artifact.checkpoint_path,
        training_metadata_path=artifact.metadata_path,
        git_commit=commit,
        uv_lock_path=ROOT / "uv.lock",
    )
    baselines = [
        run_attack_baseline(
            _baseline_config(f"v0.2-attack-{topology.name}-{policy_id}-test", topology, policy_id),
            topology,
            RUNS / "attack" / topology.name / f"{policy_id}-test",
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        for policy_id in (RANDOM_MASKED_POLICY_ID, HUNT_TARGET_POLICY_ID)
    ]
    return ppo, baselines


def _mixture(topology: Topology, attack_choice) -> FrozenDefensiveMixture:
    artifact, policy, _validation = attack_choice
    return FrozenDefensiveMixture(
        evaluators=(
            RandomMaskedEvaluator(topology),
            HuntTargetEvaluator(topology),
            FrozenPPOEvaluator(
                policy=policy,
                topology=topology,
                training_metadata=load_training_metadata(artifact.metadata_path),
                checkpoint_id="v0.2-selected",
            ),
        ),
        weights=(1.0, 1.0, 1.0),
        evaluator_id="v0.2-random-hunt-ppo-mixture",
    )


def _train_placement(topology: Topology, mixture: FrozenDefensiveMixture, commit: str):
    candidates = []
    for seed in TRAIN_SEEDS:
        run_id = f"v0.2-placement-{topology.name}-seed-{seed}"
        artifact = train_placement_policy(
            topology,
            PlacementTrainingConfig(
                run_id=run_id,
                seed=seed,
                total_timesteps=TOTAL_TIMESTEPS,
                checkpoint_directory=LOCAL_MODELS / "placement" / topology.name,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
            ),
            defensive_mixture=mixture,
        )
        policy = load_placement_policy(artifact.checkpoint_path)
        validation = run_placement_evaluation(
            _placement_config(f"{run_id}-validation", topology, "validation", VALIDATION_SEEDS),
            topology,
            policy,
            mixture,
            RUNS / "placement" / topology.name / run_id / "validation",
            checkpoint_path=artifact.checkpoint_path,
            training_metadata_path=artifact.metadata_path,
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        candidates.append((artifact, policy, validation))
    return max(
        candidates,
        key=lambda candidate: _mean(
            candidate[2].results, "valid_shots_to_sink", attacker_id=mixture.evaluator_id
        ),
    )


def _evaluate_placement(topology: Topology, mixture: FrozenDefensiveMixture, chosen, commit: str):
    artifact, policy, _validation = chosen
    return run_placement_evaluation(
        _placement_config(f"v0.2-placement-{topology.name}-ppo-test", topology, "test", TEST_SEEDS),
        topology,
        policy,
        mixture,
        RUNS / "placement" / topology.name / "ppo-test",
        checkpoint_path=artifact.checkpoint_path,
        training_metadata_path=artifact.metadata_path,
        git_commit=commit,
        uv_lock_path=ROOT / "uv.lock",
    )


def main() -> None:
    commit = _commit()
    attack_choices = {topology.name: _train_attack(topology, commit) for topology in TOPOLOGIES}
    attack_test = {
        topology.name: _evaluate_attack(topology, attack_choices[topology.name], commit)
        for topology in TOPOLOGIES
    }
    placement_test = {}
    for topology in PLACEMENT_TOPOLOGIES:
        mixture = _mixture(topology, attack_choices[topology.name])
        chosen = _train_placement(topology, mixture, commit)
        placement_test[topology.name] = _evaluate_placement(topology, mixture, chosen, commit)

    attack_results = []
    policy_by_run = {}
    test_statistics = {}
    for topology in TOPOLOGIES:
        ppo, baselines = attack_test[topology.name]
        attack_results.extend(ppo.results)
        policy_by_run[ppo.manifest.config.run_id] = "MaskablePPO (selected)"
        baseline_by_id = {}
        for baseline in baselines:
            attack_results.extend(baseline.results)
            policy_by_run[baseline.manifest.config.run_id] = baseline.manifest.config.policy_id
            baseline_by_id[baseline.manifest.config.policy_id] = baseline
        comparison = paired_difference_by_seed(
            _aligned(ppo.results, "test"),
            _aligned(baseline_by_id[HUNT_TARGET_POLICY_ID].results, "test"),
            metric="valid_shots",
            direction="lower",
        )
        interval = bootstrap_mean_interval(
            [item.mean for item in comparison.by_seed],
            rng=np.random.default_rng(20260720),
        )
        test_statistics[topology.name] = {
            "ppo_minus_hunt_mean_shots": comparison.mean_difference,
            "ppo_improves_over_hunt": comparison.is_improvement,
            "bootstrap_95": {
                "mean": interval.mean,
                "lower": interval.lower,
                "upper": interval.upper,
                "resamples": interval.resamples,
            },
        }

    placement_results = [result for evaluation in placement_test.values() for result in evaluation.results]
    placement_labels = {
        evaluation.manifest.config.run_id: "MaskablePPO placement (selected)"
        for evaluation in placement_test.values()
    }
    tables = ARTIFACTS / "tables"
    figures = ARTIFACTS / "figures"
    write_attack_results_csv(attack_results, tables / "attack-test-episodes.csv", policy_by_run=policy_by_run)
    write_attack_summary_markdown(attack_results, tables / "attack-test-summary.md", policy_by_run=policy_by_run)
    plot_attack_comparison(attack_results, figures / "attack-test-comparison.png", policy_by_run=policy_by_run)
    write_placement_results_csv(placement_results, tables / "placement-test-episodes.csv", policy_by_run=placement_labels)
    write_placement_summary_markdown(placement_results, tables / "placement-test-summary.md", policy_by_run=placement_labels)
    plot_placement_comparison(placement_results, figures / "placement-test-comparison.png", policy_by_run=placement_labels)
    for topology in PLACEMENT_TOPOLOGIES:
        evaluation = placement_test[topology.name]
        plot_placement_segment_heatmap(
            evaluation.results,
            topology,
            figures / f"placement-{topology.name}-heatmap.png",
        )

    periodic_policy = attack_choices[PERIODIC_TABLE_BATTLESHIP.name][1]
    trace = _attack_trace(PERIODIC_TABLE_BATTLESHIP, periodic_policy, TEST_SEEDS[0])
    write_attack_trace_gif(trace, figures / "periodic-ppo-attack.gif")
    periodic_placement = placement_test[PERIODIC_TABLE_BATTLESHIP.name]
    mixture_result = next(
        result
        for result in periodic_placement.results
        if result.attacker_id == "v0.2-random-hunt-ppo-mixture"
    )
    write_placement_trace_gif(
        mixture_result,
        PERIODIC_TABLE_BATTLESHIP,
        figures / "periodic-ppo-placement.gif",
    )
    report = {
        "campaign": "v0.2-controlled",
        "git_commit": commit,
        "training": {"seeds": list(TRAIN_SEEDS), "timesteps_per_seed": TOTAL_TIMESTEPS},
        "validation_seeds": list(VALIDATION_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "attack_ppo_vs_hunt": test_statistics,
        "selected_attack_training_seed": {
            name: choice[0].seed for name, choice in attack_choices.items()
        },
    }
    write_json_atomic(ARTIFACTS / "campaign-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
