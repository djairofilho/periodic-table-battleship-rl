"""Public behavior cloning of hunt-target followed by optional PPO fine-tuning.

The dataset deliberately records only what an attacker could have observed at
decision time: public board planes, the public action mask and the teacher's
chosen legal action.  It never serializes a fleet, occupied cell or ship id.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from periodic_table_battleship_rl.envs.attack import AttackEnvironmentConfig, AttackEnv
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.policies import hunt_target_action
from periodic_table_battleship_rl.topology import Topology

from .attack import (
    AttackCheckpointArtifact,
    AttackValidationConfig,
    MaskableAttackPolicy,
    TrainingDependencyError,
    _learn_until,
    evaluate_attack_validation,
)


HUNT_TARGET_DATASET_SCHEMA_VERSION = "hunt-target-public-dataset-v1"
IMITATION_PPO_POLICY_ID = "hunt-target-imitation-maskable-ppo-v1"
IMITATION_TRAINING_SCHEMA_VERSION = "hunt-target-imitation-training-v1"


def _require_training_stack() -> tuple[type[Any], Any, Any]:
    """Import SB3 and PyTorch lazily, preserving the base installation."""
    try:
        import torch
        import torch.nn.functional as functional
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise TrainingDependencyError(
            "Imitation training requires `uv sync --extra train`."
        ) from error
    return MaskablePPO, torch, functional


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class HuntTargetDatasetConfig:
    """Fixed, deterministic public demonstration schedule."""

    dataset_id: str
    seeds: tuple[int, ...]
    output_directory: Path
    environment_config: AttackEnvironmentConfig = field(default_factory=AttackEnvironmentConfig)

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty sequence without duplicates")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        object.__setattr__(self, "output_directory", Path(self.output_directory))

    def public_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "seeds": list(self.seeds),
            "environment_config": self.environment_config.public_dict(),
        }


@dataclass(frozen=True, slots=True)
class HuntTargetDatasetArtifact:
    """Paths and public dimensions of an immutable demonstration dataset."""

    data_path: Path
    metadata_path: Path
    sample_count: int
    scenario: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ImitationTrainingConfig:
    """Behavior-cloning and validation-only PPO fine-tuning configuration."""

    run_id: str
    seed: int
    dataset_path: Path
    checkpoint_directory: Path
    cloning_epochs: int = 8
    cloning_batch_size: int = 128
    cloning_learning_rate: float = 3e-4
    fine_tune_timesteps: int = 0
    fine_tune_checkpoint_steps: tuple[int, ...] = ()
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    ppo_learning_rate: float = 3e-4
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.seed < 0:
            raise ValueError("run_id must not be empty and seed must be non-negative")
        if self.cloning_epochs <= 0 or self.cloning_batch_size <= 0:
            raise ValueError("cloning epochs and batch size must be positive")
        if self.cloning_learning_rate <= 0 or self.ppo_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.fine_tune_timesteps < 0:
            raise ValueError("fine_tune_timesteps must be non-negative")
        if self.ppo_n_steps <= 0 or not 0 < self.ppo_batch_size <= self.ppo_n_steps:
            raise ValueError("ppo batch size must be positive and no greater than ppo_n_steps")
        if tuple(sorted(self.fine_tune_checkpoint_steps)) != self.fine_tune_checkpoint_steps:
            raise ValueError("fine_tune_checkpoint_steps must be strictly increasing")
        if len(set(self.fine_tune_checkpoint_steps)) != len(self.fine_tune_checkpoint_steps):
            raise ValueError("fine_tune_checkpoint_steps must not contain duplicates")
        if any(step <= 0 or step > self.fine_tune_timesteps for step in self.fine_tune_checkpoint_steps):
            raise ValueError("fine-tune checkpoints must be within the fine-tune budget")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["dataset_path"] = str(self.dataset_path)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        values["fine_tune_checkpoint_steps"] = list(self.fine_tune_checkpoint_steps)
        return values


@dataclass(frozen=True, slots=True)
class ImitationTrainingArtifact:
    """Clone, fine-tune and validation artifacts from one immutable run."""

    behavior_clone_path: Path
    final_checkpoint_path: Path
    metadata_path: Path
    cloning_loss: tuple[float, ...]
    behavior_clone_validation: AttackCheckpointArtifact
    fine_tune_checkpoints: tuple[AttackCheckpointArtifact, ...]
    selected_checkpoint_path: Path
    policy_id: str = IMITATION_PPO_POLICY_ID


def generate_hunt_target_dataset(
    topology: Topology, config: HuntTargetDatasetConfig
) -> HuntTargetDatasetArtifact:
    """Generate deterministic teacher decisions from public state alone."""
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    for seed in config.seeds:
        environment = AttackEnv(topology, config=config.environment_config)
        observation, _ = environment.reset(seed=seed)
        teacher_rng = np.random.default_rng(seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action_mask = environment.action_masks()
            active_hits = np.flatnonzero(observation[1].reshape(-1)).tolist()
            action = hunt_target_action(topology, action_mask, active_hits, teacher_rng)
            observations.append(observation.copy())
            masks.append(action_mask.copy())
            actions.append(action)
            observation, _, terminated, truncated, _ = environment.step(action)
    if not observations:
        raise RuntimeError("teacher schedule did not produce any public decisions")
    output = config.output_directory / config.dataset_id
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "demonstrations.npz"
    np.savez_compressed(
        data_path,
        observations=np.stack(observations),
        action_masks=np.stack(masks),
        actions=np.asarray(actions, dtype=np.int64),
    )
    metadata = {
        "schema_version": HUNT_TARGET_DATASET_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "teacher_policy": "hunt-target-v1",
        "scenario": topology.name,
        "sample_count": len(actions),
        "observation_shape": list(observations[0].shape),
        "action_count": topology.action_count,
        "schedule": config.public_dict(),
        "public_fields": ["observations", "action_masks", "actions"],
        "excluded_hidden_fields": [
            "fleet",
            "occupied_cells",
            "ship_ids",
            "ship_placements",
        ],
        "generation": {
            "active_hits": "derived only from public observation channel 1",
            "tie_breaking": "numpy Generator seeded with episode seed",
        },
    }
    metadata_path = write_json_atomic(output / "dataset.json", metadata)
    return HuntTargetDatasetArtifact(data_path, metadata_path, len(actions), topology.name)


def load_hunt_target_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the three public fields used for cloning."""
    with np.load(Path(path), allow_pickle=False) as archive:
        expected = {"observations", "action_masks", "actions"}
        if set(archive.files) != expected:
            raise ValueError("demonstration dataset must contain exactly public fields")
        observations = archive["observations"]
        masks = archive["action_masks"]
        actions = archive["actions"]
    if observations.ndim != 4 or masks.ndim != 2 or actions.ndim != 1:
        raise ValueError("demonstration dataset has invalid array dimensions")
    if not (len(observations) == len(masks) == len(actions)):
        raise ValueError("demonstration dataset arrays must have matching sample counts")
    if masks.dtype != np.bool_ or actions.dtype != np.int64:
        raise ValueError("demonstration masks must be bool and actions int64")
    if actions.size and int(actions.max()) >= masks.shape[1]:
        raise ValueError("demonstration action is outside the action mask")
    if actions.size and not np.all(masks[np.arange(len(actions)), actions]):
        raise ValueError("demonstration action must be legal under its public mask")
    return observations, masks, actions


def _validate_dataset_metadata(
    dataset_path: Path, topology: Topology
) -> tuple[dict[str, object], Path]:
    metadata_path = dataset_path.with_name("dataset.json")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("dataset metadata must contain a JSON object")
    if metadata.get("schema_version") != HUNT_TARGET_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported hunt-target dataset schema version")
    if metadata.get("scenario") != topology.name:
        raise ValueError("dataset scenario does not match topology")
    if metadata.get("public_fields") != ["observations", "action_masks", "actions"]:
        raise ValueError("dataset metadata does not certify public fields")
    return metadata, metadata_path


def _clone_epoch(
    model: Any,
    observations: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    *,
    batch_size: int,
    order: np.ndarray,
    torch: Any,
    functional: Any,
) -> float:
    """Optimise the actual SB3 policy with cross-entropy over legal actions."""
    losses: list[float] = []
    model.policy.set_training_mode(True)
    for start in range(0, len(order), batch_size):
        indexes = order[start : start + batch_size]
        batch_observations, _ = model.policy.obs_to_tensor(observations[indexes])
        batch_masks = torch.as_tensor(masks[indexes], device=model.device)
        batch_actions = torch.as_tensor(actions[indexes], device=model.device, dtype=torch.long)
        distribution = model.policy.get_distribution(
            batch_observations, action_masks=batch_masks
        )
        loss = functional.cross_entropy(distribution.distribution.logits, batch_actions)
        model.policy.optimizer.zero_grad()
        loss.backward()
        model.policy.optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return fmean(losses)


def train_hunt_target_imitation(
    topology: Topology,
    config: ImitationTrainingConfig,
    *,
    validation: AttackValidationConfig,
) -> ImitationTrainingArtifact:
    """Clone hunt-target on public decisions, then optionally fine-tune PPO.

    ``validation`` is the only evaluation input.  This API has no test-seed
    parameter, so the blind-test gate stays outside model selection.
    """
    metadata, dataset_metadata_path = _validate_dataset_metadata(config.dataset_path, topology)
    observations, masks, actions = load_hunt_target_dataset(config.dataset_path)
    model_type, torch, functional = _require_training_stack()
    schedule = metadata.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError("dataset metadata must contain a public schedule")
    environment_values = schedule.get("environment_config")
    if not isinstance(environment_values, dict):
        raise ValueError("dataset schedule must contain environment_config")
    environment_config = AttackEnvironmentConfig.from_public_dict(environment_values)
    environment = AttackEnv(topology, config=environment_config)
    environment.reset(seed=config.seed)
    model = model_type(
        "MlpPolicy",
        environment,
        seed=config.seed,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        learning_rate=config.ppo_learning_rate,
        device=config.device,
        verbose=0,
    )
    rng = np.random.default_rng(config.seed)
    losses: list[float] = []
    for _ in range(config.cloning_epochs):
        losses.append(
            _clone_epoch(
                model,
                observations,
                masks,
                actions,
                batch_size=config.cloning_batch_size,
                order=rng.permutation(len(actions)),
                torch=torch,
                functional=functional,
            )
        )
    output = config.checkpoint_directory / config.run_id
    output.mkdir(parents=True, exist_ok=True)
    clone_path = output / "behavior-clone"
    model.save(str(clone_path))
    clone_validation = AttackCheckpointArtifact(
        training_step=config.cloning_epochs,
        checkpoint_path=clone_path.with_suffix(".zip"),
        validation_results=evaluate_attack_validation(
            topology,
            MaskableAttackPolicy(model=model, policy_id=IMITATION_PPO_POLICY_ID),
            validation,
            training_step=config.cloning_epochs,
            environment_config=environment_config,
        ),
    )
    captured: list[AttackCheckpointArtifact] = []
    completed = 0
    for step in config.fine_tune_checkpoint_steps:
        _learn_until(model, completed_steps=completed, target_step=step)
        path = output / "fine-tune" / f"step-{step:09d}" / "model"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        results = evaluate_attack_validation(
            topology,
            MaskableAttackPolicy(model=model, policy_id=IMITATION_PPO_POLICY_ID),
            validation,
            training_step=step,
            environment_config=environment_config,
        )
        captured.append(AttackCheckpointArtifact(step, path.with_suffix(".zip"), results))
        completed = step
    if completed < config.fine_tune_timesteps:
        _learn_until(
            model, completed_steps=completed, target_step=config.fine_tune_timesteps
        )
    final_path = output / "model"
    model.save(str(final_path))
    selected = min(
        captured,
        key=lambda item: (item.mean_valid_shots, item.training_step),
        default=None,
    )
    selected_path = selected.checkpoint_path if selected is not None else final_path.with_suffix(".zip")
    training_metadata = {
        "schema_version": IMITATION_TRAINING_SCHEMA_VERSION,
        "algorithm": "behavior-cloning-then-MaskablePPO",
        "policy_id": IMITATION_PPO_POLICY_ID,
        "scenario": topology.name,
        "run_id": config.run_id,
        "seed": config.seed,
        "environment": {
            "class": "AttackEnv",
            "action_mask_method": "action_masks",
            "configuration": environment_config.public_dict(),
        },
        "dataset": {
            "path": str(config.dataset_path),
            "sha256": _sha256(config.dataset_path),
            "metadata_sha256": _sha256(dataset_metadata_path),
            "schema_version": metadata["schema_version"],
            "public_only": True,
        },
        "config": config.public_dict(),
        "validation": validation.public_dict(),
        "cloning_loss": losses,
        "behavior_clone_validation": clone_validation.to_dict(),
        "fine_tune_checkpoints": [checkpoint.to_dict() for checkpoint in captured],
        "selected_checkpoint": str(selected_path.relative_to(output)),
        "dependencies": {
            "sb3-contrib": _package_version("sb3-contrib"),
            "stable-baselines3": _package_version("stable-baselines3"),
        },
    }
    metadata_path = write_json_atomic(output / "training.json", training_metadata)
    return ImitationTrainingArtifact(
        behavior_clone_path=clone_path.with_suffix(".zip"),
        final_checkpoint_path=final_path.with_suffix(".zip"),
        metadata_path=metadata_path,
        cloning_loss=tuple(losses),
        behavior_clone_validation=clone_validation,
        fine_tune_checkpoints=tuple(captured),
        selected_checkpoint_path=selected_path,
    )


def load_imitation_training_metadata(path: str | Path) -> dict[str, object]:
    """Load provenance for an imitation artifact without touching PyTorch."""
    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("training metadata must contain a JSON object")
    if metadata.get("schema_version") != IMITATION_TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported imitation training metadata schema version")
    if metadata.get("policy_id") != IMITATION_PPO_POLICY_ID:
        raise ValueError("metadata does not describe a hunt-target imitation policy")
    return metadata


def load_hunt_target_imitation_policy(
    checkpoint_path: str | Path, *, device: str = "auto"
) -> MaskableAttackPolicy:
    """Load a frozen cloned/fine-tuned policy through the standard adapter."""
    model_type, _torch, _functional = _require_training_stack()
    return MaskableAttackPolicy(
        model=model_type.load(str(checkpoint_path), device=device),
        policy_id=IMITATION_PPO_POLICY_ID,
    )
