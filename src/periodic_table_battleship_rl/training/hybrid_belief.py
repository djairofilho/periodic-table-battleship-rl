"""Validation-only MaskablePPO candidate augmented with public belief maps.

This candidate intentionally has a distinct policy identifier from the CNN
control.  It uses the same spatial extractor, training budget and action mask;
the ablation changes only the two public maps produced by
``BeliefAugmentedAttackEnv``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from periodic_table_battleship_rl.belief.features import (
    BELIEF_FEATURE_SCHEMA_VERSION,
    BeliefAugmentedAttackEnv,
    BeliefFeatureConfig,
)
from periodic_table_battleship_rl.envs.attack import AttackEnvironmentConfig, AttackEnv
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.topology import Topology

from .attack import (
    AttackCheckpointArtifact,
    AttackTrainingArtifact,
    AttackValidationConfig,
    AttackValidationResult,
    MaskableAttackPolicy,
    TrainingDependencyError,
    _learn_until,
)
from .cnn import cnn_policy_kwargs, spatial_features_extractor


HYBRID_BELIEF_PPO_POLICY_ID = "maskable-ppo-cnn-public-belief-v1"
HYBRID_BELIEF_TRAINING_SCHEMA_VERSION = "attack-hybrid-belief-training-v1"


def _require_maskable_ppo() -> type[Any]:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise TrainingDependencyError(
            "Hybrid belief MaskablePPO requires `uv sync --extra train`."
        ) from error
    return MaskablePPO


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridBeliefAttackTrainingConfig:
    """Configuration of a public-belief PPO candidate training run."""

    run_id: str
    seed: int
    total_timesteps: int
    checkpoint_directory: Path
    n_steps: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    features_dim: int = 128
    device: str = "auto"
    environment_config: AttackEnvironmentConfig = field(default_factory=AttackEnvironmentConfig)
    belief_config: BeliefFeatureConfig = field(default_factory=BeliefFeatureConfig)
    policy_id: str = HYBRID_BELIEF_PPO_POLICY_ID

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.seed < 0 or self.total_timesteps <= 0:
            raise ValueError("seed must be non-negative and total_timesteps positive")
        if self.n_steps <= 0 or self.batch_size <= 0 or self.batch_size > self.n_steps:
            raise ValueError("batch_size must be positive and no greater than n_steps")
        if self.learning_rate <= 0 or self.features_dim <= 0:
            raise ValueError("learning_rate and features_dim must be positive")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        if self.policy_id != HYBRID_BELIEF_PPO_POLICY_ID:
            raise ValueError(f"policy_id must be {HYBRID_BELIEF_PPO_POLICY_ID!r}")
        object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        values["environment_config"] = self.environment_config.public_dict()
        values["belief_config"] = self.belief_config.public_dict()
        return values


def _metadata(config: HybridBeliefAttackTrainingConfig, topology: Topology) -> dict[str, object]:
    return {
        "schema_version": HYBRID_BELIEF_TRAINING_SCHEMA_VERSION,
        "algorithm": "MaskablePPO",
        "policy_id": HYBRID_BELIEF_PPO_POLICY_ID,
        "architecture": {
            "features_extractor": "spatial-cnn-adaptive-pool-v1",
            "features_dim": config.features_dim,
            "actor_critic_hidden": [64],
            "belief_feature_schema": BELIEF_FEATURE_SCHEMA_VERSION,
            "extra_public_channels": ["occupancy_probability", "outcome_entropy"],
        },
        "run_id": config.run_id,
        "seed": config.seed,
        "scenario": topology.name,
        "environment": {
            "class": "BeliefAugmentedAttackEnv(AttackEnv)",
            "action_mask_method": "action_masks",
            "action_count": topology.action_count,
            "valid_cells": topology.valid_cell_count,
            "configuration": config.environment_config.public_dict(),
            "belief_features": config.belief_config.public_dict(),
        },
        "config": config.public_dict(),
        "dependencies": {
            "sb3-contrib": _package_version("sb3-contrib"),
            "stable-baselines3": _package_version("stable-baselines3"),
        },
        "split_policy": "validation-only; blind test is prohibited by this training API",
    }


def evaluate_hybrid_belief_validation(
    topology: Topology,
    policy: MaskableAttackPolicy,
    validation: AttackValidationConfig,
    *,
    training_step: int,
    environment_config: AttackEnvironmentConfig,
    belief_config: BeliefFeatureConfig,
) -> tuple[AttackValidationResult, ...]:
    """Evaluate only fixed validation seeds with belief maps rebuilt publicly."""
    if training_step <= 0:
        raise ValueError("training_step must be positive")
    results: list[AttackValidationResult] = []
    for seed in validation.seeds:
        for episode_index in range(validation.episodes_per_seed):
            environment = BeliefAugmentedAttackEnv(
                AttackEnv(topology, config=environment_config), belief_config
            )
            observation, _ = environment.reset(seed=seed)
            hit_segments = discovery_area = 0
            terminated = truncated = False
            info: dict[str, int | bool] = {}
            while not (terminated or truncated):
                action = policy.select_action(
                    observation, environment.action_masks(), deterministic=True
                )
                observation, _, terminated, truncated, raw_info = environment.step(action)
                info = dict(raw_info)
                if bool(info["is_hit"]):
                    hit_segments += 1
                discovery_area += hit_segments
            valid_shots = int(info["valid_shots"])
            discovery_area += (topology.valid_cell_count - valid_shots) * hit_segments
            results.append(
                AttackValidationResult(
                    training_step=training_step,
                    seed=seed,
                    episode_index=episode_index,
                    valid_shots=valid_shots,
                    hit_segments=hit_segments,
                    won=terminated,
                    truncated=truncated,
                    auc_discovery=discovery_area / (17 * topology.valid_cell_count),
                )
            )
    return tuple(results)


def train_hybrid_belief_attack_policy(
    topology: Topology,
    config: HybridBeliefAttackTrainingConfig,
    *,
    validation: AttackValidationConfig | None = None,
) -> AttackTrainingArtifact:
    """Train the candidate, selecting checkpoints only on validation seeds."""
    if validation is not None and validation.checkpoint_steps[-1] > config.total_timesteps:
        raise ValueError("checkpoint_steps cannot exceed total_timesteps")
    model_type = _require_maskable_ppo()
    environment = BeliefAugmentedAttackEnv(
        AttackEnv(topology, config=config.environment_config), config.belief_config
    )
    environment.reset(seed=config.seed)
    model = model_type(
        "MlpPolicy",
        environment,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=config.device,
        policy_kwargs=cnn_policy_kwargs(config.features_dim),
        verbose=0,
    )
    output = config.checkpoint_directory / config.run_id
    output.mkdir(parents=True, exist_ok=True)
    captured: list[AttackCheckpointArtifact] = []
    completed = 0
    if validation is not None:
        for step in validation.checkpoint_steps:
            _learn_until(model, completed_steps=completed, target_step=step)
            path = output / "checkpoints" / f"step-{step:09d}" / "model"
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(path))
            captured.append(
                AttackCheckpointArtifact(
                    step,
                    path.with_suffix(".zip"),
                    evaluate_hybrid_belief_validation(
                        topology,
                        MaskableAttackPolicy(model, HYBRID_BELIEF_PPO_POLICY_ID),
                        validation,
                        training_step=step,
                        environment_config=config.environment_config,
                        belief_config=config.belief_config,
                    ),
                )
            )
            completed = step
    if completed < config.total_timesteps:
        _learn_until(model, completed_steps=completed, target_step=config.total_timesteps)
    checkpoint = output / "model"
    model.save(str(checkpoint))
    metadata_path = write_json_atomic(output / "training.json", _metadata(config, topology))
    return AttackTrainingArtifact(
        checkpoint_path=checkpoint.with_suffix(".zip"),
        metadata_path=metadata_path,
        policy_id=HYBRID_BELIEF_PPO_POLICY_ID,
        scenario=topology.name,
        seed=config.seed,
        checkpoints=tuple(captured),
    )


def load_hybrid_belief_training_metadata(path: str | Path) -> dict[str, object]:
    """Load provenance and reject a checkpoint from another architecture."""
    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("training metadata must contain a JSON object")
    if metadata.get("schema_version") != HYBRID_BELIEF_TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported hybrid belief training metadata schema version")
    if metadata.get("policy_id") != HYBRID_BELIEF_PPO_POLICY_ID:
        raise ValueError("metadata does not describe a hybrid belief PPO policy")
    return metadata


def load_hybrid_belief_attack_policy(
    checkpoint_path: str | Path,
    *,
    features_dim: int = 128,
    device: str = "auto",
) -> MaskableAttackPolicy:
    """Load a hybrid policy after registering its shared CNN extractor."""
    model_type = _require_maskable_ppo()
    model = model_type.load(
        str(checkpoint_path),
        device=device,
        custom_objects={"features_extractor_class": spatial_features_extractor(features_dim)},
    )
    return MaskableAttackPolicy(model=model, policy_id=HYBRID_BELIEF_PPO_POLICY_ID)
