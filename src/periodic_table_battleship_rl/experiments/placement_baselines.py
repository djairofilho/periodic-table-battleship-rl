"""Blind fixed-seed evaluation for independent placement baselines.

The learned placer and each baseline are evaluated through the same public
``PlacementEnv`` contract.  A baseline is reset from a seed derived only from
the public episode schedule, so it proposes the same fleet against every
attacker component and the frozen mixture for a given episode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
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
from periodic_table_battleship_rl.placement.baselines import PlacementBaseline
from periodic_table_battleship_rl.placement.defensive import (
    DefensiveEvaluator,
    FrozenDefensiveMixture,
)
from periodic_table_battleship_rl.topology import Topology

from .placement_evaluation import (
    PLACEMENT_ENVIRONMENT_VERSION,
    summarize_placement_results,
)


@dataclass(frozen=True, slots=True)
class PlacementBaselineEvaluation:
    """Public artifacts from one independent baseline evaluation."""

    manifest: RunManifest
    results: tuple[PlacementResult, ...]
    summary: Mapping[str, Any]
    persisted: PersistedRun


def run_placement_baseline_evaluation(
    config: RunConfig,
    topology: Topology,
    policy: PlacementBaseline,
    defensive_mixture: FrozenDefensiveMixture,
    run_directory: str | Path,
    *,
    git_commit: str,
    uv_lock_path: str | Path,
    software: SoftwareMetadata | None = None,
    hardware: HardwareMetadata | None = None,
) -> PlacementBaselineEvaluation:
    """Evaluate one seeded legal placement baseline on fixed public episodes."""

    _validate_config(config, topology, policy)
    evaluated_config = _with_baseline_provenance(config, defensive_mixture)
    evaluators = (*defensive_mixture.evaluators, defensive_mixture)
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
    persisted = persist_run(run_directory, manifest, results)
    return PlacementBaselineEvaluation(
        manifest=manifest,
        results=results,
        summary=summarize_placement_results(results, defensive_mixture),
        persisted=persisted,
    )


def _validate_config(
    config: RunConfig, topology: Topology, policy: PlacementBaseline
) -> None:
    if config.experiment != "placement":
        raise ValueError("placement baseline evaluation requires a placement RunConfig")
    if config.scenario != topology.name:
        raise ValueError("RunConfig scenario must match the supplied topology")
    if config.environment_version != PLACEMENT_ENVIRONMENT_VERSION:
        raise ValueError(
            "placement baseline evaluation requires environment version "
            f"{PLACEMENT_ENVIRONMENT_VERSION!r}"
        )
    if config.policy_id != policy.policy_id:
        raise ValueError("RunConfig policy_id must match the supplied baseline")


def _with_baseline_provenance(
    config: RunConfig, defensive_mixture: FrozenDefensiveMixture
) -> RunConfig:
    parameters = dict(config.parameters)
    provenance: dict[str, object] = {
        "evaluation_protocol": "blind-public-observation-v1",
        "placement_policy_kind": "independent-baseline-v1",
        "defensive_mixture": {
            "evaluator_id": defensive_mixture.evaluator_id,
            "component_ids": list(defensive_mixture.component_ids),
            "weights": list(defensive_mixture.weights),
        },
    }
    for name, value in provenance.items():
        if name in parameters and parameters[name] != value:
            raise ValueError(f"RunConfig parameter {name!r} conflicts with baseline provenance")
        parameters[name] = value
    return replace(config, parameters=parameters)


def _run_episode(
    *,
    topology: Topology,
    policy: PlacementBaseline,
    evaluator: DefensiveEvaluator,
    run_id: str,
    seed: int,
    episode_index: int,
) -> PlacementResult:
    attacker_seed = _seed(seed, episode_index, stream=1)
    policy.reset(seed=_seed(seed, episode_index, stream=2))
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


def _seed(seed: int, episode_index: int, *, stream: int) -> int:
    """Derive independent, portable attacker and policy RNG streams."""

    return int(np.random.SeedSequence((seed, episode_index, stream)).generate_state(1)[0])
