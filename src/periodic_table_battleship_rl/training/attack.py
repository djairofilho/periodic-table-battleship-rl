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
from statistics import fmean
from typing import Any

from periodic_table_battleship_rl.envs.attack import AttackEnv
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.topology import Topology


ATTACK_POLICY_ID = "maskable-ppo-v1"
TRAINING_SCHEMA_VERSION = "attack-training-v1"
VALIDATION_CURVE_SCHEMA_VERSION = "attack-validation-curve-v1"


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


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackValidationConfig:
    """Fixed validation schedule for one training seed.

    This type deliberately has no ``split`` field and accepts only validation
    seeds.  The held-out test schedule belongs to the experiment runner, not
    the training API, so it cannot accidentally drive checkpoint selection.
    """

    seeds: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    episodes_per_seed: int = 1

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("validation seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("validation seeds must not contain duplicates")
        if any(not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("validation seeds must be non-negative integers")
        if not self.checkpoint_steps:
            raise ValueError("checkpoint_steps must not be empty")
        if any(step <= 0 for step in self.checkpoint_steps):
            raise ValueError("checkpoint_steps must be positive")
        if tuple(sorted(self.checkpoint_steps)) != self.checkpoint_steps:
            raise ValueError("checkpoint_steps must be strictly increasing")
        if len(set(self.checkpoint_steps)) != len(self.checkpoint_steps):
            raise ValueError("checkpoint_steps must not contain duplicates")
        if self.episodes_per_seed <= 0:
            raise ValueError("episodes_per_seed must be positive")

    def public_dict(self) -> dict[str, object]:
        """Return a JSON-native validation-only schedule."""
        return {
            "split": "validation",
            "seeds": list(self.seeds),
            "checkpoint_steps": list(self.checkpoint_steps),
            "episodes_per_seed": self.episodes_per_seed,
        }


@dataclass(frozen=True, slots=True)
class AttackValidationResult:
    """Public metrics for one validation episode at one training checkpoint."""

    training_step: int
    seed: int
    episode_index: int
    valid_shots: int
    hit_segments: int
    won: bool
    truncated: bool
    auc_discovery: float

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-native record suitable for a learning-curve file."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AttackCheckpointArtifact:
    """One persisted intermediate model and its validation observations."""

    training_step: int
    checkpoint_path: Path
    validation_results: tuple[AttackValidationResult, ...]

    @property
    def mean_valid_shots(self) -> float:
        """Mean shots across this checkpoint's fixed validation schedule."""
        return fmean(result.valid_shots for result in self.validation_results)

    def to_dict(self) -> dict[str, object]:
        """Return the versioned curve representation of this checkpoint."""
        return {
            "training_step": self.training_step,
            "checkpoint_path": str(self.checkpoint_path),
            "mean_valid_shots": self.mean_valid_shots,
            "validation_results": [
                result.to_dict() for result in self.validation_results
            ],
        }


@dataclass(frozen=True, slots=True)
class AttackTrainingArtifact:
    """Paths and public identity of a completed checkpoint write."""

    checkpoint_path: Path
    metadata_path: Path
    policy_id: str
    scenario: str
    seed: int
    validation_curve_path: Path | None = None
    checkpoints: tuple[AttackCheckpointArtifact, ...] = ()


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


def _metadata(
    config: AttackTrainingConfig,
    topology: Topology,
    validation: AttackValidationConfig | None = None,
) -> dict[str, Any]:
    """Build public provenance needed by A4 and P3 consumers."""
    metadata: dict[str, Any] = {
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
    if validation is not None:
        metadata["validation_curve"] = {
            "schema_version": VALIDATION_CURVE_SCHEMA_VERSION,
            "path": "validation-curve.json",
            "schedule": validation.public_dict(),
        }
    return metadata


def _validation_curve(
    config: AttackTrainingConfig,
    topology: Topology,
    validation: AttackValidationConfig,
    checkpoints: tuple[AttackCheckpointArtifact, ...],
) -> dict[str, object]:
    """Build a public, validation-only learning-curve artifact."""
    return {
        "schema_version": VALIDATION_CURVE_SCHEMA_VERSION,
        "algorithm": "MaskablePPO",
        "policy_id": config.policy_id,
        "run_id": config.run_id,
        "training_seed": config.seed,
        "scenario": topology.name,
        "validation": validation.public_dict(),
        "checkpoints": [
            {
                **checkpoint.to_dict(),
                "checkpoint_path": str(
                    checkpoint.checkpoint_path.relative_to(
                        config.checkpoint_directory / config.run_id
                    )
                ),
            }
            for checkpoint in checkpoints
        ],
    }


def _write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    """Atomically persist public UTF-8 metadata after the checkpoint is complete."""
    return write_json_atomic(path, metadata)


def _validate_schedule(
    config: AttackTrainingConfig, validation: AttackValidationConfig
) -> None:
    """Reject curve schedules that would require training beyond the budget."""
    if validation.checkpoint_steps[-1] > config.total_timesteps:
        raise ValueError("checkpoint_steps cannot exceed total_timesteps")


def evaluate_attack_validation(
    topology: Topology,
    policy: MaskableAttackPolicy,
    validation: AttackValidationConfig,
    *,
    training_step: int,
) -> tuple[AttackValidationResult, ...]:
    """Evaluate a model on fixed public validation episodes only.

    The policy receives only the Gymnasium observation and action mask.  This
    lightweight evaluator intentionally does not persist a benchmark run and
    cannot accept a test split; experiment runners reserve held-out test seeds
    for final blind evaluation.
    """
    if training_step <= 0:
        raise ValueError("training_step must be positive")

    results: list[AttackValidationResult] = []
    for seed in validation.seeds:
        for episode_index in range(validation.episodes_per_seed):
            environment = AttackEnv(topology)
            observation, _ = environment.reset(seed=seed)
            hit_segments = 0
            discovery_area = 0
            terminated = truncated = False
            info: dict[str, int | bool] = {}

            while not (terminated or truncated):
                action = policy.select_action(
                    observation,
                    environment.action_masks(),
                    deterministic=True,
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


def _learn_until(
    model: Any, *, completed_steps: int, target_step: int
) -> None:
    """Continue PPO training without resetting schedules after the first chunk."""
    timesteps = target_step - completed_steps
    if completed_steps == 0:
        model.learn(total_timesteps=timesteps, progress_bar=False)
        return
    model.learn(
        total_timesteps=timesteps,
        progress_bar=False,
        reset_num_timesteps=False,
    )


def train_attack_policy(
    topology: Topology,
    config: AttackTrainingConfig,
    *,
    validation: AttackValidationConfig | None = None,
) -> AttackTrainingArtifact:
    """Train and persist a masked PPO attacker against random legal fleets.

    The output consists of ``model.zip`` and ``training.json`` under
    ``checkpoint_directory / run_id``.  This function is purposely unsuitable
    for a final benchmark run without an external evaluation protocol: it
    trains one environment and performs no model selection unless a fixed
    validation schedule is explicitly supplied.  The optional schedule emits
    intermediate model checkpoints and a versioned validation-only curve; it
    has no access to held-out test seeds.
    """
    if validation is not None:
        _validate_schedule(config, validation)
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
    output_directory = config.checkpoint_directory / config.run_id
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints: tuple[AttackCheckpointArtifact, ...] = ()
    if validation is None:
        _learn_until(model, completed_steps=0, target_step=config.total_timesteps)
    else:
        completed_steps = 0
        captured: list[AttackCheckpointArtifact] = []
        for training_step in validation.checkpoint_steps:
            _learn_until(
                model,
                completed_steps=completed_steps,
                target_step=training_step,
            )
            checkpoint_path = output_directory / "checkpoints" / f"step-{training_step:09d}" / "model"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_path))
            saved_checkpoint_path = checkpoint_path.with_suffix(".zip")
            results = evaluate_attack_validation(
                topology,
                MaskableAttackPolicy(model=model),
                validation,
                training_step=training_step,
            )
            captured.append(
                AttackCheckpointArtifact(
                    training_step=training_step,
                    checkpoint_path=saved_checkpoint_path,
                    validation_results=results,
                )
            )
            completed_steps = training_step
        if completed_steps < config.total_timesteps:
            _learn_until(
                model,
                completed_steps=completed_steps,
                target_step=config.total_timesteps,
            )
        checkpoints = tuple(captured)

    checkpoint_path = output_directory / "model"
    model.save(str(checkpoint_path))
    saved_checkpoint_path = checkpoint_path.with_suffix(".zip")
    metadata_path = _write_metadata(
        output_directory / "training.json", _metadata(config, topology, validation)
    )
    validation_curve_path = None
    if validation is not None:
        validation_curve_path = _write_metadata(
            output_directory / "validation-curve.json",
            _validation_curve(config, topology, validation, checkpoints),
        )
    return AttackTrainingArtifact(
        checkpoint_path=saved_checkpoint_path,
        metadata_path=metadata_path,
        policy_id=config.policy_id,
        scenario=topology.name,
        seed=config.seed,
        validation_curve_path=validation_curve_path,
        checkpoints=checkpoints,
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
