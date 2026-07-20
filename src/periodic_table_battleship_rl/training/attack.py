"""Reproducible MaskablePPO training for the random-fleet attack task.

This module deliberately imports :mod:`sb3_contrib` only inside the functions
that construct or load a model.  A base installation can therefore inspect
benchmark code and run non-neural baselines without PyTorch.

The pipeline is intentionally a single-environment reference implementation.
It records enough public configuration to reproduce an A3 checkpoint and to
load it later for A4 blind evaluation or P3's frozen-attacker evaluator.  It
does not claim bitwise reproducibility across hardware, PyTorch versions, or
different vectorisation strategies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from periodic_table_battleship_rl.envs.attack import AttackEnv
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.topology import Topology


ATTACK_POLICY_ID = "maskable-ppo-v1"
TRAINING_SCHEMA_VERSION = "attack-training-v1"


class TrainingDependencyError(RuntimeError):
    """Raised when a training-only dependency was not installed."""


def _require_maskable_ppo() -> type[Any]:
    """Load sb3-contrib lazily and explain how to enable training."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise TrainingDependencyError(
            "MaskablePPO requires the optional training dependencies. "
            "Install them with `uv sync --extra train`."
        ) from error
    return MaskablePPO


def _package_version(distribution: str) -> str:
    """Return an installed distribution version without importing it eagerly."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackTrainingConfig:
    """Explicit hyperparameters and output identity of one A3 training run."""

    run_id: str
    seed: int
    total_timesteps: int
    checkpoint_directory: Path
    n_steps: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    device: str = "auto"
    policy_id: str = ATTACK_POLICY_ID

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size > self.n_steps:
            raise ValueError("batch_size cannot exceed n_steps")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        if self.policy_id != ATTACK_POLICY_ID:
            raise ValueError(f"policy_id must be {ATTACK_POLICY_ID!r}")
        object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))

    def public_dict(self) -> dict[str, str | int | float]:
        """Return JSON-native public configuration, excluding path internals."""
        values = asdict(self)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        return values


@dataclass(frozen=True, slots=True)
class AttackTrainingArtifact:
    """Paths and public identity of a completed checkpoint write."""

    checkpoint_path: Path
    metadata_path: Path
    policy_id: str
    scenario: str
    seed: int


@dataclass(frozen=True, slots=True)
class MaskableAttackPolicy:
    """Frozen attack policy adapter for blind attack and placement evaluation."""

    model: Any
    policy_id: str = ATTACK_POLICY_ID

    def select_action(
        self,
        observation: Any,
        action_mask: Any,
        *,
        deterministic: bool = True,
    ) -> int:
        """Choose one valid shot using only the public observation and mask."""
        action, _state = self.model.predict(
            observation,
            action_masks=action_mask,
            deterministic=deterministic,
        )
        return int(action)


def _metadata(config: AttackTrainingConfig, topology: Topology) -> dict[str, Any]:
    """Build public provenance needed by A4 and P3 consumers."""
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "algorithm": "MaskablePPO",
        "policy_id": config.policy_id,
        "run_id": config.run_id,
        "seed": config.seed,
        "scenario": topology.name,
        "environment": {
            "class": "AttackEnv",
            "action_mask_method": "action_masks",
            "action_count": topology.action_count,
            "valid_cells": topology.valid_cell_count,
        },
        "config": config.public_dict(),
        "dependencies": {
            "sb3-contrib": _package_version("sb3-contrib"),
            "stable-baselines3": _package_version("stable-baselines3"),
        },
    }


def _write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    """Atomically persist public UTF-8 metadata after the checkpoint is complete."""
    return write_json_atomic(path, metadata)


def train_attack_policy(
    topology: Topology,
    config: AttackTrainingConfig,
) -> AttackTrainingArtifact:
    """Train and persist a masked PPO attacker against random legal fleets.

    The output consists of ``model.zip`` and ``training.json`` under
    ``checkpoint_directory / run_id``.  This function is purposely unsuitable
    for a final benchmark run without an external evaluation protocol: it
    trains one environment and performs no model selection or validation.
    """
    maskable_ppo = _require_maskable_ppo()
    environment = AttackEnv(topology)
    environment.reset(seed=config.seed)
    model = maskable_ppo(
        "MlpPolicy",
        environment,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=config.device,
        verbose=0,
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)

    output_directory = config.checkpoint_directory / config.run_id
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "model"
    model.save(str(checkpoint_path))
    saved_checkpoint_path = checkpoint_path.with_suffix(".zip")
    metadata_path = _write_metadata(output_directory / "training.json", _metadata(config, topology))
    return AttackTrainingArtifact(
        checkpoint_path=saved_checkpoint_path,
        metadata_path=metadata_path,
        policy_id=config.policy_id,
        scenario=topology.name,
        seed=config.seed,
    )


def load_training_metadata(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate public provenance for an A3 checkpoint."""
    with Path(path).open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError("training metadata must contain a JSON object")
    if metadata.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported training metadata schema version")
    if metadata.get("policy_id") != ATTACK_POLICY_ID:
        raise ValueError("training metadata does not describe a MaskablePPO attack policy")
    return metadata


def load_attack_policy(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
) -> MaskableAttackPolicy:
    """Load a frozen MaskablePPO attacker for A4 or P3 evaluation.

    Callers retain ownership of the environment and must pass its current
    ``action_masks()`` output to :meth:`MaskableAttackPolicy.select_action`.
    This keeps evaluation explicit and prevents the policy wrapper from
    accessing hidden fleet state.
    """
    maskable_ppo = _require_maskable_ppo()
    model = maskable_ppo.load(str(checkpoint_path), device=device)
    return MaskableAttackPolicy(model=model)
