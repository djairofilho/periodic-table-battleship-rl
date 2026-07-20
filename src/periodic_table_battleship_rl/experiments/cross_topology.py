"""Explicit, auditable PPO evaluation across board topologies.

The standard :func:`run_ppo_attack_evaluation` intentionally rejects a
checkpoint trained for a topology other than its target.  This module is the
only opt-in path for a topology-transfer experiment.  It validates the
checkpoint against its *source* topology, validates the run against its
*target* topology, and persists both identities and artifact hashes.

As with the standard evaluator, policy decisions receive only the public
Gymnasium observation and action mask.  No fleet placement or other hidden
environment state is exposed to the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
from periodic_table_battleship_rl.training.attack import MaskableAttackPolicy

from .attack_baselines import summarize_attack_results
from .ppo_evaluation import (
    _environment_config,
    _run_episode,
    _validate_config,
    validate_ppo_checkpoint,
)


CROSS_TOPOLOGY_PROTOCOL = "cross-topology-public-observation-v1"


@dataclass(frozen=True, slots=True)
class CrossTopologyPpoSource:
    """One frozen PPO artifact, identified by the topology that trained it."""

    topology: Topology
    policy: MaskableAttackPolicy
    checkpoint_path: Path
    training_metadata_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "training_metadata_path", Path(self.training_metadata_path))


@dataclass(frozen=True, slots=True)
class CrossTopologyPpoAttackEvaluation:
    """Public artifacts from one source-topology to target-topology evaluation."""

    source_topology: str
    target_topology: str
    manifest: RunManifest
    results: tuple[EpisodeResult, ...]
    summary: Mapping[str, Any]
    persisted: PersistedRun


@dataclass(frozen=True, slots=True)
class CrossTopologyMatrix:
    """A complete ordered train-by-test matrix of frozen PPO evaluations."""

    evaluations: tuple[CrossTopologyPpoAttackEvaluation, ...]

    def by_pair(self) -> dict[tuple[str, str], CrossTopologyPpoAttackEvaluation]:
        """Return matrix cells keyed by ``(source_scenario, target_scenario)``."""
        return {
            (evaluation.source_topology, evaluation.target_topology): evaluation
            for evaluation in self.evaluations
        }


def run_cross_topology_ppo_attack_evaluation(
    config: RunConfig,
    source: CrossTopologyPpoSource,
    target_topology: Topology,
    run_directory: str | Path,
    *,
    git_commit: str,
    uv_lock_path: str | Path,
    software: SoftwareMetadata | None = None,
    hardware: HardwareMetadata | None = None,
) -> CrossTopologyPpoAttackEvaluation:
    """Evaluate a frozen source policy on a possibly different target board.

    The function permits the diagonal ``source == target`` control cell so a
    caller can construct one complete train-by-test matrix with a single,
    clearly labelled protocol.  Different action-space sizes are rejected:
    a policy output index cannot be safely transferred in that case.
    """

    if source.topology.action_count != target_topology.action_count:
        raise ValueError(
            "cross-topology evaluation requires matching source and target action counts"
        )
    metadata = validate_ppo_checkpoint(
        source.topology,
        source.policy,
        checkpoint_path=source.checkpoint_path,
        training_metadata_path=source.training_metadata_path,
    )
    _validate_config(config, target_topology, source.policy)
    evaluated_config = _with_cross_topology_provenance(
        config,
        source_topology=source.topology,
        target_topology=target_topology,
        checkpoint_path=source.checkpoint_path,
        metadata_path=source.training_metadata_path,
    )
    results = tuple(
        _run_episode(
            topology=target_topology,
            policy=source.policy,
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
    del metadata  # Validate before any target environment is reset.
    persisted = persist_run(run_directory, manifest, results)
    return CrossTopologyPpoAttackEvaluation(
        source_topology=source.topology.name,
        target_topology=target_topology.name,
        manifest=manifest,
        results=results,
        summary=summarize_attack_results(results),
        persisted=persisted,
    )


def run_cross_topology_matrix(
    sources: Iterable[CrossTopologyPpoSource],
    target_topologies: Iterable[Topology],
    run_directory: str | Path,
    config_factory: Callable[[Topology, Topology], RunConfig],
    *,
    git_commit: str,
    uv_lock_path: str | Path,
    software: SoftwareMetadata | None = None,
    hardware: HardwareMetadata | None = None,
) -> CrossTopologyMatrix:
    """Run an ordered, complete source-by-target matrix with fixed configs.

    ``config_factory`` must create a test configuration for every ordered
    pair.  Its target scenario is checked by the evaluator, preventing a
    matrix cell from being accidentally persisted under the source scenario.
    """

    frozen_sources = tuple(sources)
    targets = tuple(target_topologies)
    _validate_matrix_inputs(frozen_sources, targets)
    base_directory = Path(run_directory)
    evaluations = tuple(
        run_cross_topology_ppo_attack_evaluation(
            config_factory(source.topology, target),
            source,
            target,
            base_directory / source.topology.name / target.name,
            git_commit=git_commit,
            uv_lock_path=uv_lock_path,
            software=software,
            hardware=hardware,
        )
        for source in frozen_sources
        for target in targets
    )
    return CrossTopologyMatrix(evaluations=evaluations)


def _validate_matrix_inputs(
    sources: tuple[CrossTopologyPpoSource, ...], targets: tuple[Topology, ...]
) -> None:
    if not sources:
        raise ValueError("cross-topology matrix requires at least one source")
    if not targets:
        raise ValueError("cross-topology matrix requires at least one target")
    source_names = [source.topology.name for source in sources]
    target_names = [topology.name for topology in targets]
    if len(set(source_names)) != len(source_names):
        raise ValueError("cross-topology matrix source scenarios must be unique")
    if len(set(target_names)) != len(target_names):
        raise ValueError("cross-topology matrix target scenarios must be unique")


def _with_cross_topology_provenance(
    config: RunConfig,
    *,
    source_topology: Topology,
    target_topology: Topology,
    checkpoint_path: Path,
    metadata_path: Path,
) -> RunConfig:
    parameters = dict(config.parameters)
    provenance: dict[str, str | int] = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_metadata_sha256": sha256_file(metadata_path),
        "evaluation_protocol": CROSS_TOPOLOGY_PROTOCOL,
        "source_scenario": source_topology.name,
        "source_valid_cells": source_topology.valid_cell_count,
        "source_action_count": source_topology.action_count,
        "target_scenario": target_topology.name,
        "target_valid_cells": target_topology.valid_cell_count,
        "target_action_count": target_topology.action_count,
    }
    for name, value in provenance.items():
        if name in parameters and parameters[name] != value:
            raise ValueError(f"RunConfig parameter {name!r} conflicts with evaluated artifact")
        parameters[name] = value
    return replace(config, parameters=parameters)
