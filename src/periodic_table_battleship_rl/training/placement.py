"""Reproducible MaskablePPO training for sequential fleet placement.

The training dependency is deliberately optional.  Importing this module only
uses the standard library and benchmark packages; :mod:`sb3_contrib` is loaded
when a caller actually starts training or loads a checkpoint.

The initial P4 pipeline trains against the frozen, versioned random and
hunt-target mixture.  Integrating a frozen PPO attacker is P3's separate
responsibility, so its checkpoint is never silently substituted here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from periodic_table_battleship_rl.envs.placement import PlacementEnv
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.placement.defensive import (
    FrozenDefensiveMixture,
    default_defensive_mixture,
)
from periodic_table_battleship_rl.topology import Topology


PLACEMENT_POLICY_ID = "maskable-ppo-placement-v1"
"""Stable identity for the initial learned fleet-placement policy."""

PLACEMENT_TRAINING_SCHEMA_VERSION = "placement-training-v1"
"""Schema version for public placement-training provenance."""


class PlacementTrainingDependencyError(RuntimeError):
    """Raised when the optional MaskablePPO dependency is unavailable."""


def _require_maskable_ppo() -> type[Any]:
    """Load sb3-contrib only for training and checkpoint-loading operations."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise PlacementTrainingDependencyError(
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
class PlacementTrainingConfig:
    """Explicit hyperparameters and output identity for one P4 training run."""

    run_id: str
    seed: int
    total_timesteps: int
    checkpoint_directory: Path
    n_steps: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    device: str = "auto"
    policy_id: str = PLACEMENT_POLICY_ID

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
        if self.policy_id != PLACEMENT_POLICY_ID:
            raise ValueError(f"policy_id must be {PLACEMENT_POLICY_ID!r}")
        object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))

    def public_dict(self) -> dict[str, str | int | float]:
        """Return JSON-native configuration without hidden runtime state."""
        values = asdict(self)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        return values


@dataclass(frozen=True, slots=True)
class PlacementTrainingArtifact:
    """Paths and public identity of one persisted placement checkpoint."""

    checkpoint_path: Path
    metadata_path: Path
    policy_id: str
    scenario: str
    evaluator_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class MaskablePlacementPolicy:
    """Frozen placement-policy adapter requiring an explicit legal-action mask."""

    model: Any
    policy_id: str = PLACEMENT_POLICY_ID

    def select_action(
        self,
        observation: Any,
        action_mask: Any,
        *,
        deterministic: bool = True,
    ) -> int:
        """Choose one legal placement without reading the environment internals."""
        action, _state = self.model.predict(
            observation,
            action_masks=action_mask,
            deterministic=deterministic,
        )
        return int(action)


def _mixture_metadata(mixture: FrozenDefensiveMixture) -> dict[str, object]:
    """Record exactly the evaluator suite that produced the training reward."""
    return {
        "evaluator_id": mixture.evaluator_id,
        "component_ids": list(mixture.component_ids),
        "weights": list(mixture.weights),
    }


def _metadata(
    config: PlacementTrainingConfig,
    topology: Topology,
    mixture: FrozenDefensiveMixture,
) -> dict[str, Any]:
    """Build public provenance needed for later P5 evaluation."""
    return {
        "schema_version": PLACEMENT_TRAINING_SCHEMA_VERSION,
        "algorithm": "MaskablePPO",
        "policy_id": config.policy_id,
        "run_id": config.run_id,
        "seed": config.seed,
        "scenario": topology.name,
        "environment": {
            "class": "PlacementEnv",
            "action_mask_method": "action_masks",
            "action_count": 360,
            "valid_cells": topology.valid_cell_count,
            "fleet_order": [5, 4, 3, 3, 2],
        },
        "defensive_mixture": _mixture_metadata(mixture),
        "config": config.public_dict(),
        "dependencies": {
            "sb3-contrib": _package_version("sb3-contrib"),
            "stable-baselines3": _package_version("stable-baselines3"),
        },
    }


def train_placement_policy(
    topology: Topology,
    config: PlacementTrainingConfig,
) -> PlacementTrainingArtifact:
    """Train and persist a masked PPO placer against the default frozen suite.

    The checkpoint and its ``training.json`` provenance are written below
    ``checkpoint_directory / run_id``.  This is a single-environment training
    reference, not a final performance claim: P5 must evaluate it against each
    fixed attacker independently using held-out episode seeds.
    """
    maskable_ppo = _require_maskable_ppo()
    mixture = default_defensive_mixture(topology)
    environment = PlacementEnv(topology, evaluator=mixture)
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
    metadata_path = write_json_atomic(
        output_directory / "training.json", _metadata(config, topology, mixture)
    )
    return PlacementTrainingArtifact(
        checkpoint_path=saved_checkpoint_path,
        metadata_path=metadata_path,
        policy_id=config.policy_id,
        scenario=topology.name,
        evaluator_id=mixture.evaluator_id,
        seed=config.seed,
    )


def load_placement_training_metadata(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate public provenance for a P4 checkpoint."""
    with Path(path).open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError("training metadata must contain a JSON object")
    if metadata.get("schema_version") != PLACEMENT_TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported placement training metadata schema version")
    if metadata.get("policy_id") != PLACEMENT_POLICY_ID:
        raise ValueError("training metadata does not describe a MaskablePPO placement policy")
    return metadata


def load_placement_policy(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
) -> MaskablePlacementPolicy:
    """Load a frozen policy for a caller-owned :class:`PlacementEnv` instance."""
    maskable_ppo = _require_maskable_ppo()
    model = maskable_ppo.load(str(checkpoint_path), device=device)
    return MaskablePlacementPolicy(model=model)
