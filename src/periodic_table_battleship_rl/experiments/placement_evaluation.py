"""Blind, fixed-seed evaluation for a frozen MaskablePPO placer.

The placement policy receives only ``PlacementEnv`` observations and legal
action masks.  A held-out schedule is evaluated independently against every
component of the frozen defensive suite and once against the suite itself.
That makes component-specific claims distinguishable from the training-mixture
claim while keeping the placed fleet private to the environment and attacker.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any

import numpy as np

from periodic_table_battleship_rl.envs.placement import PlacementEnv
from periodic_table_battleship_rl.evaluation.schemas import (
    EpisodeManifest,
    HardwareMetadata,
    PlacementResult,
    RunConfig,
    RunManifest,
    SoftwareMetadata,
    sha256_file,
)
from periodic_table_battleship_rl.evaluation.storage import PersistedRun, persist_run
from periodic_table_battleship_rl.game import CANONICAL_FLEET
from periodic_table_battleship_rl.placement.defensive import (
    DefensiveEvaluator,
    FrozenDefensiveMixture,
)
from periodic_table_battleship_rl.topology import Topology
from periodic_table_battleship_rl.training.placement import (
    PLACEMENT_POLICY_ID,
    PLACEMENT_TRAINING_SCHEMA_VERSION,
    MaskablePlacementPolicy,
    load_placement_training_metadata,
)


PLACEMENT_ENVIRONMENT_VERSION = "placement-env-v1"
"""Versioned public contract for the sequential placement environment."""


@dataclass(frozen=True, slots=True)
class PlacementEvaluation:
    """Public artifacts emitted by one component and mixture evaluation."""

    manifest: RunManifest
    results: tuple[PlacementResult, ...]
    summary: Mapping[str, Any]
    persisted: PersistedRun


def run_placement_evaluation(
    config: RunConfig,
    topology: Topology,
    policy: MaskablePlacementPolicy,
    defensive_mixture: FrozenDefensiveMixture,
    run_directory: str | Path,
    *,
    checkpoint_path: str | Path,
    training_metadata_path: str | Path,
    git_commit: str,
    uv_lock_path: str | Path,
    software: SoftwareMetadata | None = None,
    hardware: HardwareMetadata | None = None,
) -> PlacementEvaluation:
    """Evaluate a frozen placer on held-out seeds without hidden-state access.

    The exact same policy episode is run against each component and the frozen
    mixture.  Every evaluator receives the same derived attacker seed for a
    given ``(seed, episode_index)`` pair, so per-attacker differences are
    paired.  The policy is never given the partial fleet, evaluator identity,
    or any attacker state.
    """

    checkpoint = Path(checkpoint_path)
    metadata_path = Path(training_metadata_path)
    metadata = validate_placement_checkpoint(
        topology,
        policy,
        defensive_mixture,
        checkpoint_path=checkpoint,
        training_metadata_path=metadata_path,
    )
    _validate_config(config, topology, policy)
    evaluated_config = _with_checkpoint_provenance(
        config, checkpoint, metadata_path, defensive_mixture
    )
    evaluators = _evaluation_evaluators(defensive_mixture)
    results = tuple(
        _run_episode(
            topology=topology,
            policy=policy,
            evaluator=evaluator,
            run_id=evaluated_config.run_id,
            seed=seed,
            episode_index=episode_index,
        )
        for evaluator in evaluators
        for seed in evaluated_config.seeds
        for episode_index in range(evaluated_config.episodes_per_seed)
    )
    manifest = RunManifest(
        config=evaluated_config,
        git_commit=git_commit,
        uv_lock_sha256=sha256_file(uv_lock_path),
        software=SoftwareMetadata.current() if software is None else software,
        hardware=HardwareMetadata.current() if hardware is None else hardware,
        episodes=EpisodeManifest(
            run_id=evaluated_config.run_id,
            episode_ids=tuple(result.episode_id for result in results),
        ),
    )
    del metadata
    summary = summarize_placement_results(results, defensive_mixture)
    persisted = persist_run(run_directory, manifest, results)
    return PlacementEvaluation(
        manifest=manifest,
        results=results,
        summary=summary,
        persisted=persisted,
    )


def validate_placement_checkpoint(
    topology: Topology,
    policy: MaskablePlacementPolicy,
    defensive_mixture: FrozenDefensiveMixture,
    *,
    checkpoint_path: str | Path,
    training_metadata_path: str | Path,
) -> dict[str, Any]:
    """Validate portable P4 provenance before an episode can begin."""

    checkpoint = Path(checkpoint_path)
    metadata_path = Path(training_metadata_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"placement checkpoint does not exist: {checkpoint}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"placement training metadata does not exist: {metadata_path}"
        )

    metadata = load_placement_training_metadata(metadata_path)
    if metadata.get("schema_version") != PLACEMENT_TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported placement training metadata schema version")
    if metadata.get("scenario") != topology.name:
        raise ValueError("placement checkpoint scenario does not match the supplied topology")
    if metadata.get("policy_id") != policy.policy_id:
        raise ValueError("placement checkpoint policy_id does not match the supplied policy")

    environment = metadata.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("placement training metadata must contain an environment object")
    expected_environment = {
        "class": "PlacementEnv",
        "action_mask_method": "action_masks",
        "action_count": 360,
        "valid_cells": topology.valid_cell_count,
        "fleet_order": [5, 4, 3, 3, 2],
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise ValueError(
                f"placement checkpoint environment {name!r} does not match topology"
            )

    mixture_metadata = metadata.get("defensive_mixture")
    if not isinstance(mixture_metadata, Mapping):
        raise ValueError("placement training metadata must contain defensive_mixture")
    expected_mixture = _mixture_metadata(defensive_mixture)
    for name, expected in expected_mixture.items():
        if mixture_metadata.get(name) != expected:
            raise ValueError(
                f"placement checkpoint defensive mixture {name!r} does not match evaluation"
            )
    return metadata


def summarize_placement_results(
    results: Sequence[PlacementResult],
    defensive_mixture: FrozenDefensiveMixture,
) -> dict[str, Any]:
    """Aggregate held-out results independently for components and mixture."""

    if not results:
        raise ValueError("results must contain at least one episode")

    by_attacker: dict[str, list[PlacementResult]] = defaultdict(list)
    for result in results:
        by_attacker[result.attacker_id].append(result)

    expected_ids = {
        *defensive_mixture.component_ids,
        defensive_mixture.evaluator_id,
    }
    if set(by_attacker) != expected_ids:
        raise ValueError("results must cover every mixture component and the mixture")

    component_summaries = {
        evaluator_id: _attacker_summary(by_attacker[evaluator_id])
        for evaluator_id in defensive_mixture.component_ids
    }
    mixture_summary = _attacker_summary(by_attacker[defensive_mixture.evaluator_id])
    return {
        "episode_count": len(results),
        "seed_count": len({result.seed for result in results}),
        "components": component_summaries,
        "mixture": mixture_summary,
    }


def _validate_config(
    config: RunConfig, topology: Topology, policy: MaskablePlacementPolicy
) -> None:
    if config.experiment != "placement":
        raise ValueError("placement evaluation requires a placement RunConfig")
    if config.scenario != topology.name:
        raise ValueError("RunConfig scenario must match the supplied topology")
    if config.environment_version != PLACEMENT_ENVIRONMENT_VERSION:
        raise ValueError(
            "placement evaluation requires environment version "
            f"{PLACEMENT_ENVIRONMENT_VERSION!r}"
        )
    if config.policy_id != policy.policy_id:
        raise ValueError("RunConfig policy_id must match the supplied policy")
    if config.policy_id != PLACEMENT_POLICY_ID:
        raise ValueError(
            f"placement evaluation requires policy {PLACEMENT_POLICY_ID!r}"
        )


def _with_checkpoint_provenance(
    config: RunConfig,
    checkpoint_path: Path,
    metadata_path: Path,
    defensive_mixture: FrozenDefensiveMixture,
) -> RunConfig:
    parameters = dict(config.parameters)
    provenance: dict[str, object] = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_metadata_sha256": sha256_file(metadata_path),
        "evaluation_protocol": "blind-public-observation-v1",
        "defensive_mixture": _mixture_metadata(defensive_mixture),
    }
    for name, value in provenance.items():
        if name in parameters and parameters[name] != value:
            raise ValueError(f"RunConfig parameter {name!r} conflicts with evaluated artifact")
        parameters[name] = value
    return replace(config, parameters=parameters)


def _evaluation_evaluators(
    defensive_mixture: FrozenDefensiveMixture,
) -> tuple[DefensiveEvaluator, ...]:
    evaluators = (*defensive_mixture.evaluators, defensive_mixture)
    identifiers = tuple(evaluator.evaluator_id for evaluator in evaluators)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("mixture evaluator_id must differ from every component evaluator_id")
    return evaluators


def _mixture_metadata(mixture: FrozenDefensiveMixture) -> dict[str, object]:
    return {
        "evaluator_id": mixture.evaluator_id,
        "component_ids": list(mixture.component_ids),
        "weights": list(mixture.weights),
    }


def _run_episode(
    *,
    topology: Topology,
    policy: MaskablePlacementPolicy,
    evaluator: DefensiveEvaluator,
    run_id: str,
    seed: int,
    episode_index: int,
) -> PlacementResult:
    """Run a complete placement episode with only public policy inputs."""

    attacker_seed = _attacker_seed(seed, episode_index)
    environment = PlacementEnv(topology, evaluator)
    observation, _ = environment.reset(seed=attacker_seed)
    terminated = truncated = False
    info: Mapping[str, object] = {}

    while not (terminated or truncated):
        action = policy.select_action(
            observation,
            environment.action_masks(),
            deterministic=True,
        )
        observation, _, terminated, truncated, info = environment.step(action)

    actions = info["placement_actions"]
    shots = info["valid_shots_to_sink"]
    if not isinstance(actions, tuple) or not all(isinstance(action, int) for action in actions):
        raise RuntimeError("PlacementEnv returned invalid public placement actions")
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise RuntimeError("PlacementEnv returned invalid public shot count")

    # The evaluator protocol intentionally exposes no hidden attack trajectory.
    # Terminal facts are still exact: all canonical ship segments were sunk at
    # ``shots``.  AUC is set to zero rather than inventing a hit chronology.
    return PlacementResult(
        episode_id=(
            f"{run_id}-attacker-{evaluator.evaluator_id}-seed-{seed}"
            f"-episode-{episode_index:03d}"
        ),
        run_id=run_id,
        seed=seed,
        scenario=topology.name,
        attacker_id=evaluator.evaluator_id,
        attacker_seed=attacker_seed,
        placement_actions=actions,
        valid_cells=topology.valid_cell_count,
        valid_shots_to_sink=shots,
        hit_segments=17,
        sunk_ship_lengths=tuple(ship.length for ship in CANONICAL_FLEET),
        auc_discovery=0.0,
        all_sunk_shot=shots,
        truncated=truncated,
    )


def _attacker_seed(seed: int, episode_index: int) -> int:
    """Derive a portable, paired attacker stream from the held-out schedule."""

    return int(np.random.SeedSequence((seed, episode_index)).generate_state(1)[0])


def _attacker_summary(results: Iterable[PlacementResult]) -> dict[str, Any]:
    records = tuple(results)
    by_seed: dict[int, list[PlacementResult]] = defaultdict(list)
    for result in records:
        by_seed[result.seed].append(result)
    per_seed = {
        str(seed): _seed_metrics(seed_results) for seed, seed_results in sorted(by_seed.items())
    }
    metric_names = tuple(next(iter(per_seed.values())).keys())
    aggregate = {
        name: _distribution([metrics[name] for metrics in per_seed.values()])
        for name in metric_names
    }
    return {
        "episode_count": len(records),
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def _seed_metrics(results: Sequence[PlacementResult]) -> dict[str, float]:
    shots = [float(result.valid_shots_to_sink) for result in results]
    return {
        "valid_shots_to_sink": fmean(shots),
        "valid_shots_to_sink_normalized": fmean(
            result.valid_shots_to_sink / result.valid_cells for result in results
        ),
        "shots_excess": fmean(result.valid_shots_to_sink - 17 for result in results),
        "placement_completion_rate": fmean(
            float(len(result.placement_actions) == len(CANONICAL_FLEET)) for result in results
        ),
        "truncation_rate": fmean(float(result.truncated) for result in results),
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": float(median(values)),
    }
