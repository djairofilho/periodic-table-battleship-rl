"""Validation-only evaluation for public-history belief planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from periodic_table_battleship_rl.belief import BeliefPlanner, PublicAttackState
from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.evaluation.schemas import (
    EpisodeManifest,
    EpisodeResult,
    HardwareMetadata,
    RunConfig,
    RunManifest,
    SoftwareMetadata,
    sha256_file,
)
from periodic_table_battleship_rl.evaluation.storage import PersistedRun, persist_run, write_json_atomic
from periodic_table_battleship_rl.experiments.attack_baselines import (
    ENVIRONMENT_VERSION,
    summarize_attack_results,
)
from periodic_table_battleship_rl.topology import Topology


BELIEF_PROBABILITY_POLICY_ID = "belief_probability_mc-v1"
BELIEF_INFORMATION_POLICY_ID = "belief_information_mc-v1"
BELIEF_HORIZON_POLICY_ID = "belief_horizon2_mc-v1"
BELIEF_POLICY_IDS = (
    BELIEF_PROBABILITY_POLICY_ID,
    BELIEF_INFORMATION_POLICY_ID,
    BELIEF_HORIZON_POLICY_ID,
)


@dataclass(frozen=True, slots=True)
class BeliefPlannerRun:
    """Persisted public outcomes plus aggregate sampler diagnostics."""

    manifest: RunManifest
    results: tuple[EpisodeResult, ...]
    summary: dict[str, Any]
    persisted: PersistedRun
    summary_path: Path


def run_belief_planner_evaluation(
    config: RunConfig,
    topology: Topology,
    run_directory: str | Path,
    *,
    git_commit: str,
    uv_lock_path: str | Path,
    sample_count: int = 128,
    max_restarts_per_sample: int = 128,
    max_nodes_per_sample: int = 8_192,
    software: SoftwareMetadata | None = None,
    hardware: HardwareMetadata | None = None,
) -> BeliefPlannerRun:
    """Evaluate a Monte Carlo planner using only validation or train seeds.

    This runner rejects the ``test`` split by design.  A later candidate may
    opt into blind confirmation only through the persisted promotion protocol.
    """
    if config.experiment != "attack":
        raise ValueError("belief planner evaluation requires experiment='attack'")
    if config.scenario != topology.name:
        raise ValueError("RunConfig scenario must match the supplied topology")
    if config.environment_version != ENVIRONMENT_VERSION:
        raise ValueError("belief planner evaluation requires the attack environment")
    if config.split == "test":
        raise ValueError("belief planner pilot preserves the blind test split")
    strategy = _strategy_for_policy(config.policy_id)
    planner = BeliefPlanner(
        strategy=strategy,
        sample_count=sample_count,
        max_restarts_per_sample=max_restarts_per_sample,
        max_nodes_per_sample=max_nodes_per_sample,
    )
    outcomes = tuple(
        _run_episode(
            topology,
            planner,
            config.run_id,
            seed,
            episode_index,
        )
        for seed in config.seeds
        for episode_index in range(config.episodes_per_seed)
    )
    results = tuple(result for result, _ in outcomes)
    diagnostics = tuple(diagnostic for _, diagnostic in outcomes)
    manifest = RunManifest(
        config=config,
        git_commit=git_commit,
        uv_lock_sha256=sha256_file(uv_lock_path),
        software=SoftwareMetadata.current() if software is None else software,
        hardware=HardwareMetadata.current() if hardware is None else hardware,
        episodes=EpisodeManifest(
            run_id=config.run_id,
            episode_ids=tuple(result.episode_id for result in results),
        ),
    )
    summary = dict(summarize_attack_results(results))
    summary["belief_sampler"] = {
        "sampler_id": "constrained-backtracking-v1",
        "posterior_exact": False,
        "sample_count_per_decision": sample_count,
        "max_restarts_per_sample": max_restarts_per_sample,
        "max_nodes_per_sample": max_nodes_per_sample,
        "decision_count": sum(item[2] for item in diagnostics),
        "mean_backtracks_per_decision": float(
            sum(item[0] for item in diagnostics) / sum(item[2] for item in diagnostics)
        ),
        "mean_restarts_per_decision": float(
            sum(item[1] for item in diagnostics) / sum(item[2] for item in diagnostics)
        ),
    }
    summary_path = write_json_atomic(Path(run_directory) / "summary.json", summary)
    persisted = persist_run(run_directory, manifest, results)
    return BeliefPlannerRun(manifest, results, summary, persisted, summary_path)


def _run_episode(
    topology: Topology,
    planner: BeliefPlanner,
    run_id: str,
    seed: int,
    episode_index: int,
) -> tuple[EpisodeResult, tuple[int, int, int]]:
    env = AttackEnv(topology)
    observation, _ = env.reset(seed=seed)
    policy_rng = np.random.default_rng(np.random.SeedSequence((seed, episode_index)))
    hit_segments = discovery_area = 0
    sunk_ship_lengths: list[int] = []
    first_hit_shot: int | None = None
    first_sunk_shot: int | None = None
    aggregate_backtracks = aggregate_restarts = 0
    decision_count = 0
    terminated = truncated = False
    info: dict[str, int | bool] = {}
    while not (terminated or truncated):
        state = PublicAttackState.from_observation(topology, observation)
        action, diagnostics = planner.select_action(state, env.action_masks(), policy_rng)
        aggregate_backtracks += diagnostics.backtrack_count
        aggregate_restarts += diagnostics.restart_count
        decision_count += 1
        observation, _, terminated, truncated, info = env.step(action)
        if bool(info["is_hit"]):
            hit_segments += 1
            if first_hit_shot is None:
                first_hit_shot = int(info["valid_shots"])
        sunk_ship_length = int(info["sunk_ship_length"])
        if sunk_ship_length:
            sunk_ship_lengths.append(sunk_ship_length)
            if first_sunk_shot is None:
                first_sunk_shot = int(info["valid_shots"])
        discovery_area += hit_segments
    valid_shots = int(info["valid_shots"])
    discovery_area += (topology.valid_cell_count - valid_shots) * hit_segments
    return (
        EpisodeResult(
            episode_id=f"{run_id}-seed-{seed}-episode-{episode_index:03d}",
            run_id=run_id,
            seed=seed,
            scenario=topology.name,
            valid_cells=topology.valid_cell_count,
            valid_shots=valid_shots,
            invalid_attempts=int(info["invalid_attempts"]),
            hit_segments=hit_segments,
            sunk_ship_lengths=tuple(sunk_ship_lengths),
            won=terminated,
            truncated=truncated,
            auc_discovery=discovery_area / (17 * topology.valid_cell_count),
            first_hit_shot=first_hit_shot,
            first_sunk_shot=first_sunk_shot,
        ),
        (aggregate_backtracks, aggregate_restarts, decision_count),
    )


def _strategy_for_policy(policy_id: str) -> str:
    strategies = {
        BELIEF_PROBABILITY_POLICY_ID: "probability",
        BELIEF_INFORMATION_POLICY_ID: "information",
        BELIEF_HORIZON_POLICY_ID: "horizon-2",
    }
    try:
        return strategies[policy_id]
    except KeyError as error:
        raise ValueError(f"unsupported belief policy: {policy_id!r}") from error
