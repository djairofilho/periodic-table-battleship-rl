"""Deterministic benchmarks for the public attack-policy baselines.

This module evaluates one policy per :class:`~.RunConfig`, so a persisted run
has one scenario, one policy and one fixed seed schedule.  The convenience
runner evaluates both initial policies on both benchmark topologies while
retaining that one-run-per-comparison-unit layout.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, Callable

import numpy as np

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
from periodic_table_battleship_rl.evaluation.storage import (
    PersistedRun,
    persist_run,
    write_json_atomic,
)
from periodic_table_battleship_rl.policies import (
    hunt_target_action,
    random_masked_action,
)
from periodic_table_battleship_rl.topology import (
    BATTLESHIP,
    PERIODIC_TABLE_BATTLESHIP,
    Topology,
)


ENVIRONMENT_VERSION = "attack-env-v1"
RANDOM_MASKED_POLICY_ID = "random_masked-v1"
HUNT_TARGET_POLICY_ID = "hunt_target-v1"
INITIAL_POLICY_IDS = (RANDOM_MASKED_POLICY_ID, HUNT_TARGET_POLICY_ID)
INITIAL_TOPOLOGIES = (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP)


@dataclass(frozen=True, slots=True)
class AttackBaselineRun:
    """Public artifacts emitted by a deterministic baseline evaluation."""

    manifest: RunManifest
    results: tuple[EpisodeResult, ...]
    summary: Mapping[str, Any]
    persisted: PersistedRun
    summary_path: Path


def run_attack_baseline(
    config: RunConfig,
    topology: Topology,
    run_directory: str | Path,
    *,
    git_commit: str,
    uv_lock_path: str | Path,
    software: SoftwareMetadata | None = None,
    hardware: HardwareMetadata | None = None,
) -> AttackBaselineRun:
    """Evaluate and persist one masked attack baseline.

    Each configured seed resets the environment to the same legal fleet for
    all policies.  Repeated episodes use independent, deterministic policy
    streams derived from ``(seed, episode_index)``.  This keeps the opponent
    collection paired between policies while still sampling their tie breaks.
    """

    _validate_config(config, topology)
    policy = _get_policy(config.policy_id)
    results = tuple(
        _run_episode(
            topology=topology,
            policy=policy,
            run_id=config.run_id,
            seed=seed,
            episode_index=episode_index,
        )
        for seed in config.seeds
        for episode_index in range(config.episodes_per_seed)
    )
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
    summary = summarize_attack_results(results)
    summary_path = write_json_atomic(Path(run_directory) / "summary.json", summary)
    persisted = persist_run(run_directory, manifest, results)
    return AttackBaselineRun(
        manifest=manifest,
        results=results,
        summary=summary,
        persisted=persisted,
        summary_path=summary_path,
    )


def run_initial_attack_baselines(
    output_directory: str | Path,
    *,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    git_commit: str,
    uv_lock_path: str | Path,
    split: str = "test",
    run_prefix: str = "initial-baselines",
    topologies: Iterable[Topology] = INITIAL_TOPOLOGIES,
    policy_ids: Iterable[str] = INITIAL_POLICY_IDS,
    software: SoftwareMetadata | None = None,
    hardware: HardwareMetadata | None = None,
) -> tuple[AttackBaselineRun, ...]:
    """Persist the random and hunt-target benchmarks on both topologies."""

    topologies = tuple(topologies)
    policy_ids = tuple(policy_ids)
    runs: list[AttackBaselineRun] = []
    for topology in topologies:
        for policy_id in policy_ids:
            run_id = f"{run_prefix}-{topology.name}-{policy_id}"
            config = RunConfig(
                run_id=run_id,
                experiment="attack",
                scenario=topology.name,
                environment_version=ENVIRONMENT_VERSION,
                policy_id=policy_id,
                split=split,
                seeds=seeds,
                episodes_per_seed=episodes_per_seed,
                parameters={
                    "fleet_sampler": "random_legal-v1",
                    "policy_rng": "numpy-seed-sequence-v1",
                    "reward_version": "efficiency-v0",
                },
            )
            runs.append(
                run_attack_baseline(
                    config,
                    topology,
                    Path(output_directory) / run_id,
                    git_commit=git_commit,
                    uv_lock_path=uv_lock_path,
                    software=software,
                    hardware=hardware,
                )
            )
    return tuple(runs)


def summarize_attack_results(
    results: Sequence[EpisodeResult],
) -> dict[str, dict[str, dict[str, float]] | int]:
    """Return JSON-native metrics, aggregating episode means by seed first."""

    if not results:
        raise ValueError("results must contain at least one episode")

    by_seed: dict[int, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        by_seed[result.seed].append(result)

    per_seed = {
        str(seed): _seed_metrics(seed_results)
        for seed, seed_results in sorted(by_seed.items())
    }
    metric_names = tuple(next(iter(per_seed.values())).keys())
    aggregate = {
        metric_name: _distribution(
            [metrics[metric_name] for metrics in per_seed.values()]
        )
        for metric_name in metric_names
    }
    return {
        "episode_count": len(results),
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def _run_episode(
    *,
    topology: Topology,
    policy: Callable[[Topology, np.ndarray, np.ndarray, np.random.Generator], int],
    run_id: str,
    seed: int,
    episode_index: int,
) -> EpisodeResult:
    env = AttackEnv(topology)
    observation, _ = env.reset(seed=seed)
    policy_rng = np.random.default_rng(np.random.SeedSequence((seed, episode_index)))
    hit_segments = 0
    discovery_area = 0
    sunk_ship_lengths: list[int] = []
    first_hit_shot: int | None = None
    first_sunk_shot: int | None = None
    terminated = truncated = False
    info: dict[str, int | bool] = {}

    while not (terminated or truncated):
        action = policy(topology, env.action_masks(), observation[1], policy_rng)
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
    return EpisodeResult(
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
    )


def _get_policy(
    policy_id: str,
) -> Callable[[Topology, np.ndarray, np.ndarray, np.random.Generator], int]:
    if policy_id == RANDOM_MASKED_POLICY_ID:
        return _random_policy
    if policy_id == HUNT_TARGET_POLICY_ID:
        return _hunt_target_policy
    raise ValueError(f"unsupported attack baseline: {policy_id!r}")


def _random_policy(
    topology: Topology,
    action_mask: np.ndarray,
    active_hits: np.ndarray,
    rng: np.random.Generator,
) -> int:
    del topology, active_hits
    return random_masked_action(action_mask, rng)


def _hunt_target_policy(
    topology: Topology,
    action_mask: np.ndarray,
    active_hits: np.ndarray,
    rng: np.random.Generator,
) -> int:
    return hunt_target_action(topology, action_mask, np.flatnonzero(active_hits), rng)


def _validate_config(config: RunConfig, topology: Topology) -> None:
    if config.experiment != "attack":
        raise ValueError("attack baseline requires an attack RunConfig")
    if config.scenario != topology.name:
        raise ValueError("RunConfig scenario must match the supplied topology")
    if config.environment_version != ENVIRONMENT_VERSION:
        raise ValueError(
            f"attack baseline requires environment version {ENVIRONMENT_VERSION!r}"
        )
    _get_policy(config.policy_id)


def _seed_metrics(results: Sequence[EpisodeResult]) -> dict[str, float]:
    valid_shots = [float(result.valid_shots) for result in results]
    return {
        "valid_shots": fmean(valid_shots),
        "valid_shots_normalized": fmean(
            result.valid_shots / result.valid_cells for result in results
        ),
        "shots_excess": fmean(result.valid_shots - 17 for result in results),
        "win_rate": fmean(float(result.won) for result in results),
        "truncation_rate": fmean(float(result.truncated) for result in results),
        "hit_rate": fmean(
            result.hit_segments / result.valid_shots if result.valid_shots else 0.0
            for result in results
        ),
        "invalid_attempts": fmean(float(result.invalid_attempts) for result in results),
        "auc_discovery": fmean(result.auc_discovery for result in results),
        "first_hit_shot": _mean_optional(result.first_hit_shot for result in results),
        "first_sunk_shot": _mean_optional(result.first_sunk_shot for result in results),
    }


def _mean_optional(values: Iterable[int | None]) -> float:
    present = [float(value) for value in values if value is not None]
    return fmean(present) if present else 0.0


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": float(median(values)),
    }
