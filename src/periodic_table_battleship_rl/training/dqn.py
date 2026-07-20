"""Masked DQN training for the public Battleship attack task.

The implementation keeps action legality in three places: behaviour selection,
replay records, and the target-network bootstrap.  In particular, an illegal
canvas cell can never become a greedy next action merely because its raw Q
value is large.  PyTorch remains an optional, lazy dependency of the ``train``
extra.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from periodic_table_battleship_rl.envs import AttackEnvironmentConfig, AttackEnv
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.topology import Topology


DQN_ATTACK_POLICY_ID = "masked-dqn-v1"
DQN_TRAINING_SCHEMA_VERSION = "dqn-attack-training-v1"


class DqnTrainingDependencyError(RuntimeError):
    """Raised when the optional PyTorch training dependency is unavailable."""


def _require_torch() -> Any:
    """Load PyTorch lazily so base installs can inspect this module."""
    try:
        import torch
    except ImportError as error:
        raise DqnTrainingDependencyError(
            "Masked DQN requires the optional training dependencies. "
            "Install them with `uv sync --extra train`."
        ) from error
    return torch


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class DqnAttackTrainingConfig:
    """Explicit, reproducible hyperparameters for one masked-DQN run."""

    run_id: str
    seed: int
    total_steps: int
    checkpoint_directory: Path
    replay_capacity: int = 20_000
    batch_size: int = 64
    warmup_steps: int = 512
    train_frequency: int = 1
    target_update_interval: int = 500
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    device: str = "cpu"
    environment_config: AttackEnvironmentConfig = AttackEnvironmentConfig()
    policy_id: str = DQN_ATTACK_POLICY_ID

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        for name in (
            "total_steps",
            "replay_capacity",
            "batch_size",
            "warmup_steps",
            "train_frequency",
            "target_update_interval",
            "hidden_dim",
            "epsilon_decay_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        if self.policy_id != DQN_ATTACK_POLICY_ID:
            raise ValueError(f"policy_id must be {DQN_ATTACK_POLICY_ID!r}")
        object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        values["environment_config"] = self.environment_config.public_dict()
        return values


@dataclass(frozen=True, slots=True)
class DqnAttackTrainingArtifact:
    """Paths and public identity of a completed DQN checkpoint."""

    checkpoint_path: Path
    metadata_path: Path
    policy_id: str
    scenario: str
    seed: int
    final_epsilon: float
    replay_size: int


@dataclass(frozen=True, slots=True)
class DqnTransition:
    """One public transition, including the *next* legal-action mask."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    next_action_mask: np.ndarray


class ReplayBuffer:
    """Bounded FIFO replay buffer with deterministic sample RNG ownership."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._records: deque[DqnTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def append(self, transition: DqnTransition) -> None:
        if transition.action < 0:
            raise ValueError("transition action must be non-negative")
        if transition.next_action_mask.dtype != np.bool_:
            raise TypeError("next_action_mask must have dtype bool")
        self._records.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> tuple[DqnTransition, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._records):
            raise ValueError("batch_size cannot exceed replay size")
        return tuple(rng.sample(list(self._records), batch_size))


def masked_argmax(values: Any, action_masks: Any) -> Any:
    """Return greedy actions after excluding every false action-mask entry.

    ``values`` has shape ``(batch, actions)`` and ``action_masks`` has the
    same shape.  A row without legal actions is a caller error: terminal rows
    are handled separately by :func:`masked_bootstrap_values`.
    """
    torch = _require_torch()
    if values.ndim != 2 or action_masks.shape != values.shape:
        raise ValueError("values and action_masks must be matching rank-2 tensors")
    if not bool(torch.all(action_masks.any(dim=1))):
        raise ValueError("masked_argmax requires at least one legal action per row")
    return values.masked_fill(~action_masks, -torch.inf).argmax(dim=1)


def masked_bootstrap_values(next_q_values: Any, next_action_masks: Any) -> Any:
    """Return masked target-network maxima, using zero for terminal rows."""
    torch = _require_torch()
    if next_q_values.ndim != 2 or next_action_masks.shape != next_q_values.shape:
        raise ValueError("Q values and masks must be matching rank-2 tensors")
    legal = next_action_masks.any(dim=1)
    masked = next_q_values.masked_fill(~next_action_masks, -torch.inf)
    return torch.where(legal, masked.max(dim=1).values, torch.zeros_like(legal, dtype=next_q_values.dtype))


def dqn_targets(
    rewards: Any,
    terminated_or_truncated: Any,
    next_q_values: Any,
    next_action_masks: Any,
    *,
    gamma: float,
) -> Any:
    """Build Bellman targets whose bootstrap obeys the next-state action mask."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    bootstrap = masked_bootstrap_values(next_q_values, next_action_masks)
    return rewards + gamma * (~terminated_or_truncated).to(rewards.dtype) * bootstrap


class _MlpQNetworkFactory:
    """Lazy factory to keep importing the module independent of PyTorch."""

    @staticmethod
    def create(observation_shape: tuple[int, ...], action_count: int, hidden_dim: int) -> Any:
        torch = _require_torch()
        input_size = int(np.prod(observation_shape))
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_count),
        )


@dataclass(frozen=True, slots=True)
class MaskedDqnPolicy:
    """Frozen DQN adapter that permits only public observation and masks."""

    network: Any
    device: str = "cpu"
    policy_id: str = DQN_ATTACK_POLICY_ID

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        del deterministic
        torch = _require_torch()
        if action_mask.dtype != np.bool_:
            raise TypeError("action_mask must have dtype bool")
        if not action_mask.any():
            raise ValueError("cannot select action from an empty mask")
        self.network.eval()
        with torch.no_grad():
            batch = torch.as_tensor(observation, device=self.device).unsqueeze(0).float()
            values = self.network(batch)
            mask = torch.as_tensor(action_mask, device=self.device).unsqueeze(0)
            return int(masked_argmax(values, mask).item())


def _epsilon(config: DqnAttackTrainingConfig, step: int) -> float:
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def _metadata(config: DqnAttackTrainingConfig, topology: Topology) -> dict[str, object]:
    return {
        "schema_version": DQN_TRAINING_SCHEMA_VERSION,
        "algorithm": "MaskedDQN",
        "policy_id": config.policy_id,
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
        "dependencies": {"torch": _package_version("torch")},
    }


def _batch_loss(network: Any, target_network: Any, batch: Iterable[DqnTransition], gamma: float, device: str) -> Any:
    torch = _require_torch()
    records = tuple(batch)
    observations = torch.as_tensor(np.stack([item.observation for item in records]), device=device).float()
    actions = torch.as_tensor([item.action for item in records], device=device).long()
    rewards = torch.as_tensor([item.reward for item in records], device=device).float()
    next_observations = torch.as_tensor(np.stack([item.next_observation for item in records]), device=device).float()
    done = torch.as_tensor(
        [item.terminated or item.truncated for item in records], device=device, dtype=torch.bool
    )
    next_masks = torch.as_tensor(
        np.stack([item.next_action_mask for item in records]), device=device, dtype=torch.bool
    )
    predicted = network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        targets = dqn_targets(rewards, done, target_network(next_observations), next_masks, gamma=gamma)
    return torch.nn.functional.smooth_l1_loss(predicted, targets)


def train_masked_dqn_attack_policy(
    topology: Topology, config: DqnAttackTrainingConfig
) -> DqnAttackTrainingArtifact:
    """Train a replay-buffer DQN without allowing invalid bootstrap actions.

    This is a single-environment reference runner.  Callers should select
    checkpoints on validation seeds and run blind tests through a separate
    protocol before treating a model as a benchmark candidate.
    """
    torch = _require_torch()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    environment = AttackEnv(topology, config=config.environment_config)
    observation, _ = environment.reset(seed=config.seed)
    network = _MlpQNetworkFactory.create(
        environment.observation_space.shape, topology.action_count, config.hidden_dim
    ).to(config.device)
    target_network = _MlpQNetworkFactory.create(
        environment.observation_space.shape, topology.action_count, config.hidden_dim
    ).to(config.device)
    target_network.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.replay_capacity)
    sample_rng = random.Random(config.seed + 1)
    action_rng = np.random.default_rng(config.seed + 2)

    for step in range(1, config.total_steps + 1):
        mask = environment.action_masks()
        if action_rng.random() < _epsilon(config, step - 1):
            action = int(action_rng.choice(np.flatnonzero(mask)))
        else:
            action = MaskedDqnPolicy(network=network, device=config.device).select_action(
                observation, mask
            )
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        buffer.append(
            DqnTransition(
                observation=observation.copy(),
                action=action,
                reward=float(reward),
                next_observation=next_observation.copy(),
                terminated=terminated,
                truncated=truncated,
                next_action_mask=environment.action_masks(),
            )
        )
        observation = next_observation
        if terminated or truncated:
            observation, _ = environment.reset(seed=config.seed + step)
        if len(buffer) >= max(config.batch_size, config.warmup_steps) and step % config.train_frequency == 0:
            loss = _batch_loss(
                network,
                target_network,
                buffer.sample(config.batch_size, sample_rng),
                config.gamma,
                config.device,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            optimizer.step()
        if step % config.target_update_interval == 0:
            target_network.load_state_dict(network.state_dict())

    output_directory = config.checkpoint_directory / config.run_id
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "model.pt"
    torch.save(
        {
            "state_dict": network.state_dict(),
            "observation_shape": tuple(environment.observation_space.shape),
            "action_count": topology.action_count,
            "hidden_dim": config.hidden_dim,
        },
        checkpoint_path,
    )
    metadata_path = write_json_atomic(output_directory / "training.json", _metadata(config, topology))
    return DqnAttackTrainingArtifact(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        policy_id=config.policy_id,
        scenario=topology.name,
        seed=config.seed,
        final_epsilon=_epsilon(config, config.total_steps),
        replay_size=len(buffer),
    )


def load_masked_dqn_attack_policy(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> MaskedDqnPolicy:
    """Load a DQN model using only architecture fields saved alongside weights."""
    torch = _require_torch()
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("DQN checkpoint must contain a mapping")
    shape = payload.get("observation_shape")
    action_count = payload.get("action_count")
    hidden_dim = payload.get("hidden_dim")
    state_dict = payload.get("state_dict")
    if (
        not isinstance(shape, tuple)
        or not isinstance(action_count, int)
        or not isinstance(hidden_dim, int)
        or not isinstance(state_dict, dict)
    ):
        raise ValueError("DQN checkpoint has invalid architecture metadata")
    network = _MlpQNetworkFactory.create(shape, action_count, hidden_dim).to(device)
    network.load_state_dict(state_dict)
    return MaskedDqnPolicy(network=network, device=device)


def load_dqn_training_metadata(path: str | Path) -> dict[str, Any]:
    """Load public DQN provenance and reject a different schema immediately."""
    with Path(path).open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != DQN_TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported DQN training metadata schema version")
    if metadata.get("policy_id") != DQN_ATTACK_POLICY_ID:
        raise ValueError("training metadata does not describe a masked DQN attack policy")
    return metadata
