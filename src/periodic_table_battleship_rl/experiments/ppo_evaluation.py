"""Blind, reproducible evaluation for a frozen MaskablePPO attacker.

This module never inspects environment internals.  The policy receives only
the public Gymnasium observation and ``AttackEnv.action_masks()`` output.  A
checkpoint's public training metadata is verified before evaluation so a
model trained for one topology cannot be reported against another by mistake.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from periodic_table_battleship_rl.envs import AttackEnvironmentConfig, AttackEnv
from periodic_table_battleship_rl.evaluation.schemas import (
    EpisodeManifest,
    EpisodeResult,
    HardwareMetadata,
    RunConfig,
    RunManifest,
    SoftwareMetadata,
    sha256_file,
)
from periodic_table_battleship_rl.evaluation.storage import PersistedRun, persist_run
from periodic_table_battleship_rl.topology import Topology
from periodic_table_battleship_rl.training.attack import (
    ATTACK_POLICY_ID,
    TRAINING_SCHEMA_VERSION,
    MaskableAttackPolicy,
    load_training_metadata,
)

from .attack_baselines import ENVIRONMENT_VERSION, summarize_attack_results


@dataclass(frozen=True, slots=True)
class PpoAttackEvaluation:
    """Public artifacts from one fixed-seed frozen-PPO evaluation."""

    manifest: RunManifest
    results: tuple[EpisodeResult, ...]
    summary: Mapping[str, Any]
    persisted: PersistedRun


def run_ppo_attack_evaluation(
    config: RunConfig,
    topology: Topology,
    policy: MaskableAttackPolicy,
    run_directory: str | Path,
    *,
    checkpoint_path: str | Path,
    training_metadata_path: str | Path,
    git_commit: str,
    uv_lock_path: str | Path,
    software: SoftwareMetadata | None = None,
    hardware: HardwareMetadata | None = None,
) -> PpoAttackEvaluation:
    """Evaluate a frozen PPO attacker using only public environment state.

    ``config.seeds`` is the complete, fixed episode schedule.  This function
    does not train, select, or mutate the policy, which lets callers reserve a
    held-out seed list for a future blind test.  Checkpoint and metadata hashes
    are added to the persisted run configuration; supplied conflicting hashes
    are rejected instead of silently overwriting provenance.
    """

    checkpoint = Path(checkpoint_path)
    metadata_path = Path(training_metadata_path)
    metadata = validate_ppo_checkpoint(
        topology,
        policy,
        checkpoint_path=checkpoint,
        training_metadata_path=metadata_path,
    )
    _validate_config(config, topology, policy)
    evaluated_config = _with_checkpoint_provenance(config, checkpoint, metadata_path)
    results = tuple(
        _run_episode(
            topology=topology,
            policy=policy,
            run_id=evaluated_config.run_id,
            seed=seed,
            episode_index=episode_index,
            environment_config=_environment_config(metadata),
        )
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
    del metadata  # Metadata was validated before any environment was reset.
    summary = summarize_attack_results(results)
    persisted = persist_run(run_directory, manifest, results)
    return PpoAttackEvaluation(
        manifest=manifest,
        results=results,
        summary=summary,
        persisted=persisted,
    )


def validate_ppo_checkpoint(
    topology: Topology,
    policy: MaskableAttackPolicy,
    *,
    checkpoint_path: str | Path,
    training_metadata_path: str | Path,
) -> dict[str, Any]:
    """Ensure a frozen model's public artifact metadata fits ``topology``."""

    checkpoint = Path(checkpoint_path)
    metadata_path = Path(training_metadata_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"PPO training metadata does not exist: {metadata_path}")

    metadata = load_training_metadata(metadata_path)
    if metadata.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported PPO training metadata schema version")
    if metadata.get("scenario") != topology.name:
        raise ValueError("PPO checkpoint scenario does not match the supplied topology")
    if metadata.get("policy_id") != policy.policy_id:
        raise ValueError("PPO checkpoint policy_id does not match the supplied policy")

    environment = metadata.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("PPO training metadata must contain an environment object")
    expected_environment = {
        "class": "AttackEnv",
        "action_mask_method": "action_masks",
        "action_count": topology.action_count,
        "valid_cells": topology.valid_cell_count,
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise ValueError(f"PPO checkpoint environment {name!r} does not match topology")
    return metadata


def _validate_config(
    config: RunConfig, topology: Topology, policy: MaskableAttackPolicy
) -> None:
    if config.experiment != "attack":
        raise ValueError("PPO attack evaluation requires an attack RunConfig")
    if config.scenario != topology.name:
        raise ValueError("RunConfig scenario must match the supplied topology")
    if config.environment_version != ENVIRONMENT_VERSION:
        raise ValueError(
            f"PPO attack evaluation requires environment version {ENVIRONMENT_VERSION!r}"
        )
    if config.policy_id != policy.policy_id:
        raise ValueError("RunConfig policy_id must match the supplied policy")
    if config.policy_id != ATTACK_POLICY_ID:
        raise ValueError(f"PPO attack evaluation requires policy {ATTACK_POLICY_ID!r}")


def _environment_config(metadata: Mapping[str, Any]) -> AttackEnvironmentConfig:
    """Read the model's public environment choice, defaulting v0.3 metadata."""
    environment = metadata["environment"]
    assert isinstance(environment, dict)
    configuration = environment.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("PPO training environment configuration must be an object")
    return AttackEnvironmentConfig.from_public_dict(configuration)


def _with_checkpoint_provenance(
    config: RunConfig, checkpoint_path: Path, metadata_path: Path
) -> RunConfig:
    parameters = dict(config.parameters)
    hashes = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_metadata_sha256": sha256_file(metadata_path),
        "evaluation_protocol": "blind-public-observation-v1",
    }
    for name, value in hashes.items():
        if name in parameters and parameters[name] != value:
            raise ValueError(f"RunConfig parameter {name!r} conflicts with evaluated artifact")
        parameters[name] = value
    return replace(config, parameters=parameters)


def _run_episode(
    *,
    topology: Topology,
    policy: MaskableAttackPolicy,
    run_id: str,
    seed: int,
    episode_index: int,
    environment_config: AttackEnvironmentConfig,
) -> EpisodeResult:
    """Play one episode without reading any non-public ``AttackEnv`` state."""

    environment = AttackEnv(topology, config=environment_config)
    observation, _ = environment.reset(seed=seed)
    hit_segments = 0
    discovery_area = 0
    sunk_ship_lengths: list[int] = []
    first_hit_shot: int | None = None
    first_sunk_shot: int | None = None
    terminated = truncated = False
    info: Mapping[str, int | bool] = {}

    while not (terminated or truncated):
        action = policy.select_action(
            observation,
            environment.action_masks(),
            deterministic=True,
        )
        observation, _, terminated, truncated, info = environment.step(action)
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
