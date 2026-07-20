"""Spatial MaskablePPO attack policy kept separate from the MLP control.

The v0.3/v0.4 MLP training pipeline intentionally remains unchanged.  This
module owns the CNN candidate's identity, architecture and provenance so it
cannot accidentally be evaluated or reported as that control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.metadata
from pathlib import Path
from typing import Any

from periodic_table_battleship_rl.envs.attack import AttackEnvironmentConfig, AttackEnv
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.topology import Topology

from .attack import (
    AttackCheckpointArtifact,
    AttackTrainingArtifact,
    AttackValidationConfig,
    MaskableAttackPolicy,
    TrainingDependencyError,
    _learn_until,
    evaluate_attack_validation,
)


CNN_ATTACK_POLICY_ID = "maskable-ppo-cnn-v1"
CNN_TRAINING_SCHEMA_VERSION = "attack-cnn-training-v1"


def _require_maskable_ppo() -> type[Any]:
    """Import the optional learner only when CNN training is requested."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise TrainingDependencyError(
            "CNN MaskablePPO requires `uv sync --extra train`."
        ) from error
    return MaskablePPO


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class CnnAttackTrainingConfig:
    """Public configuration of one spatial PPO training run."""

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
    policy_id: str = CNN_ATTACK_POLICY_ID

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
        if self.policy_id != CNN_ATTACK_POLICY_ID:
            raise ValueError(f"policy_id must be {CNN_ATTACK_POLICY_ID!r}")
        object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        values["environment_config"] = self.environment_config.public_dict()
        return values


def spatial_features_extractor(features_dim: int) -> type[Any]:
    """Build the small CNN lazily, without importing PyTorch at package import.

    Adaptive pooling makes the feature extractor valid for all fixed 10x18
    scenarios, including layouts with periodic-table gaps represented by the
    public validity plane.
    """
    if features_dim <= 0:
        raise ValueError("features_dim must be positive")
    try:
        import torch.nn as nn
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    except ImportError as error:
        raise TrainingDependencyError(
            "CNN MaskablePPO requires `uv sync --extra train`."
        ) from error

    class SpatialAttackFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: Any) -> None:
            super().__init__(observation_space, features_dim)
            channels = int(observation_space.shape[0])
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 3)),
                nn.Flatten(),
            )
            self.linear = nn.Sequential(nn.Linear(64 * 2 * 3, features_dim), nn.ReLU())

        def forward(self, observations: Any) -> Any:
            return self.linear(self.cnn(observations.float()))

    return SpatialAttackFeaturesExtractor


def cnn_policy_kwargs(features_dim: int) -> dict[str, object]:
    """Return the exact network description persisted in training metadata."""
    return {
        "features_extractor_class": spatial_features_extractor(features_dim),
        "net_arch": {"pi": [64], "vf": [64]},
        "normalize_images": False,
    }


def _metadata(config: CnnAttackTrainingConfig, topology: Topology) -> dict[str, object]:
    return {
        "schema_version": CNN_TRAINING_SCHEMA_VERSION,
        "algorithm": "MaskablePPO",
        "policy_id": CNN_ATTACK_POLICY_ID,
        "architecture": {
            "features_extractor": "spatial-cnn-adaptive-pool-v1",
            "features_dim": config.features_dim,
            "convolutions": [
                {"in_channels": "observation_channels", "out_channels": 32, "kernel": 3},
                {"in_channels": 32, "out_channels": 64, "kernel": 3},
            ],
            "actor_critic_hidden": [64],
        },
        "run_id": config.run_id,
        "seed": config.seed,
        "scenario": topology.name,
        "environment": {
            "class": "AttackEnv",
            "action_mask_method": "action_masks",
            "action_count": topology.action_count,
            "valid_cells": topology.valid_cell_count,
            "configuration": config.environment_config.public_dict(),
        },
        "config": config.public_dict(),
        "dependencies": {
            "sb3-contrib": _package_version("sb3-contrib"),
            "stable-baselines3": _package_version("stable-baselines3"),
        },
    }


def train_cnn_attack_policy(
    topology: Topology,
    config: CnnAttackTrainingConfig,
    *,
    validation: AttackValidationConfig | None = None,
) -> AttackTrainingArtifact:
    """Train a CNN PPO candidate, selecting no checkpoint on held-out data."""
    if validation is not None and validation.checkpoint_steps[-1] > config.total_timesteps:
        raise ValueError("checkpoint_steps cannot exceed total_timesteps")
    model_type = _require_maskable_ppo()
    environment = AttackEnv(topology, config=config.environment_config)
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
            saved = path.with_suffix(".zip")
            results = evaluate_attack_validation(
                topology,
                MaskableAttackPolicy(model=model, policy_id=CNN_ATTACK_POLICY_ID),
                validation,
                training_step=step,
                environment_config=config.environment_config,
            )
            captured.append(AttackCheckpointArtifact(step, saved, results))
            completed = step
    if completed < config.total_timesteps:
        _learn_until(model, completed_steps=completed, target_step=config.total_timesteps)
    checkpoint = output / "model"
    model.save(str(checkpoint))
    metadata_path = write_json_atomic(output / "training.json", _metadata(config, topology))
    return AttackTrainingArtifact(
        checkpoint_path=checkpoint.with_suffix(".zip"),
        metadata_path=metadata_path,
        policy_id=CNN_ATTACK_POLICY_ID,
        scenario=topology.name,
        seed=config.seed,
        checkpoints=tuple(captured),
    )


def load_cnn_attack_policy(
    checkpoint_path: str | Path,
    *,
    features_dim: int = 128,
    device: str = "auto",
) -> MaskableAttackPolicy:
    """Load a CNN policy with its custom extractor registered before deserializing."""
    model_type = _require_maskable_ppo()
    model = model_type.load(
        str(checkpoint_path),
        device=device,
        custom_objects={"features_extractor_class": spatial_features_extractor(features_dim)},
    )
    return MaskableAttackPolicy(model=model, policy_id=CNN_ATTACK_POLICY_ID)


def load_cnn_training_metadata(path: str | Path) -> dict[str, object]:
    """Load CNN provenance and reject MLP or incompatible artifacts."""
    import json

    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("training metadata must contain a JSON object")
    if metadata.get("schema_version") != CNN_TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported CNN training metadata schema version")
    if metadata.get("policy_id") != CNN_ATTACK_POLICY_ID:
        raise ValueError("metadata does not describe a CNN PPO attack policy")
    return metadata
