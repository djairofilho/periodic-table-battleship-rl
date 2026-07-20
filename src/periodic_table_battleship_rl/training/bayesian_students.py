"""Public neural students distilled from Bayesian Battleship demonstrations.

The models consume exactly the four documented attacker observation planes and
the legal-action mask.  They never receive a fleet, ship identity, reward, or
any other environment-private value.  This is supervised policy distillation,
not a replacement for the Bayesian teacher nor an opening of the blind test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import numpy as np

from periodic_table_battleship_rl.envs.attack import AttackEnv
from periodic_table_battleship_rl.topology import Topology

from .dqn import _require_torch, masked_argmax
from .gnn import TopologyGraphQNetwork
from .bayesian_distillation import (
    BayesianDemonstrations,
    load_bayesian_demonstration_metadata,
    load_bayesian_demonstrations,
)


BAYESIAN_STUDENT_SCHEMA_VERSION = "bayesian-public-student-v1"
"""Schema for a distillation checkpoint and validation report."""

StudentArchitecture = Literal["cnn", "gnn"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class BayesianStudentTrainingConfig:
    """Small deterministic training budget appropriate for a 4 GB GPU."""

    run_id: str
    architecture: StudentArchitecture
    seed: int
    dataset_path: Path
    checkpoint_directory: Path
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dim: int = 32
    soft_target_weight: float = 0.35
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.architecture not in {"cnn", "gnn"}:
            raise ValueError(
                "run_id must not be empty and architecture must be cnn or gnn"
            )
        if self.seed < 0 or min(self.epochs, self.batch_size, self.hidden_dim) <= 0:
            raise ValueError("seed, epochs, batch_size and hidden_dim must be positive")
        if self.learning_rate <= 0 or not 0.0 <= self.soft_target_weight <= 1.0:
            raise ValueError("learning rate and soft target weight are invalid")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(
            self, "checkpoint_directory", Path(self.checkpoint_directory)
        )

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["dataset_path"] = str(self.dataset_path)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        return values


@dataclass(frozen=True, slots=True)
class BayesianStudentArtifact:
    """One public distillation checkpoint plus train-set diagnostics."""

    checkpoint_path: Path
    metadata_path: Path
    architecture: StudentArchitecture
    losses: tuple[float, ...]
    training_action_agreement: float


@dataclass(frozen=True, slots=True)
class BayesianStudentPolicy:
    """Legal-action adapter around a public observation student."""

    network: Any
    device: str
    policy_id: str

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        del deterministic
        if action_mask.dtype != np.bool_:
            raise TypeError("action_mask must have dtype bool")
        if not action_mask.any():
            raise ValueError("cannot choose from an empty action mask")
        torch = _require_torch()
        self.network.eval()
        with torch.no_grad():
            logits = self.network(
                torch.as_tensor(
                    observation, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            )
            mask = torch.as_tensor(
                action_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            return int(masked_argmax(logits, mask).item())


def build_bayesian_student(
    topology: Topology,
    *,
    architecture: StudentArchitecture,
    observation_channels: int,
    hidden_dim: int,
) -> Any:
    """Create a compact CNN or topology-aware GNN without extra frameworks."""
    if observation_channels <= 0 or hidden_dim <= 0:
        raise ValueError("observation_channels and hidden_dim must be positive")
    torch = _require_torch()
    if architecture == "gnn":
        return TopologyGraphQNetwork.create(
            topology,
            observation_channels=observation_channels,
            hidden_dim=hidden_dim,
            message_passing_steps=2,
        )
    if architecture != "cnn":
        raise ValueError("architecture must be cnn or gnn")

    class _CnnStudent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(
                    observation_channels, hidden_dim, kernel_size=3, padding=1
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Conv2d(hidden_dim, 1, kernel_size=1)

        def forward(self, observations: Any) -> Any:
            if observations.ndim != 4 or observations.shape[1] != observation_channels:
                raise ValueError(
                    "observations must be shaped (batch, channels, rows, columns)"
                )
            return self.head(self.backbone(observations)).flatten(start_dim=1)

    return _CnnStudent()


def train_bayesian_student(
    topology: Topology,
    config: BayesianStudentTrainingConfig,
) -> BayesianStudentArtifact:
    """Distil an immutable public dataset into one student architecture."""
    metadata = load_bayesian_demonstration_metadata(config.dataset_path)
    if metadata.get("scenario") != topology.name:
        raise ValueError("dataset scenario does not match topology")
    demonstrations = load_bayesian_demonstrations(config.dataset_path)
    if demonstrations.sample_count == 0:
        raise ValueError("Bayesian demonstrations must not be empty")
    torch = _require_torch()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    resolved_device = _resolve_device(torch, config.device)
    network = build_bayesian_student(
        topology,
        architecture=config.architecture,
        observation_channels=int(demonstrations.observations.shape[1]),
        hidden_dim=config.hidden_dim,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    observations = torch.as_tensor(
        demonstrations.observations, dtype=torch.float32, device=resolved_device
    )
    masks = torch.as_tensor(
        demonstrations.action_masks, dtype=torch.bool, device=resolved_device
    )
    actions = torch.as_tensor(
        demonstrations.teacher_actions, dtype=torch.long, device=resolved_device
    )
    scores = torch.as_tensor(
        demonstrations.teacher_occupancy_probabilities,
        dtype=torch.float32,
        device=resolved_device,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    losses: list[float] = []
    for _ in range(config.epochs):
        indices = torch.randperm(len(actions), generator=generator)
        batch_losses: list[float] = []
        network.train()
        for start in range(0, len(indices), config.batch_size):
            batch = indices[start : start + config.batch_size].to(resolved_device)
            # Cross entropy accepts infinities, while KL with a zero target at
            # an impossible action can propagate ``0 * -inf`` as NaN.  A very
            # negative finite logit preserves the legal-action distribution
            # without introducing a non-finite training objective.
            logits = network(observations[batch]).masked_fill(~masks[batch], -1e9)
            hard_loss = torch.nn.functional.cross_entropy(logits, actions[batch])
            soft_targets = _normalise_public_scores(scores[batch], masks[batch], torch)
            soft_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits, dim=1),
                soft_targets,
                reduction="batchmean",
            )
            loss = (
                1.0 - config.soft_target_weight
            ) * hard_loss + config.soft_target_weight * soft_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(fmean(batch_losses))
    agreement = _action_agreement(network, demonstrations, device=resolved_device)
    output = config.checkpoint_directory / config.run_id
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "student.pt"
    torch.save(
        {
            "schema_version": BAYESIAN_STUDENT_SCHEMA_VERSION,
            "architecture": config.architecture,
            "observation_channels": int(demonstrations.observations.shape[1]),
            "hidden_dim": config.hidden_dim,
            "state_dict": network.state_dict(),
        },
        checkpoint_path,
    )
    training_metadata = {
        "schema_version": BAYESIAN_STUDENT_SCHEMA_VERSION,
        "policy_id": f"bayesian-{config.architecture}-student-v1",
        "scenario": topology.name,
        "public_only": True,
        "dataset": {
            "path": str(config.dataset_path),
            "sha256": _sha256(config.dataset_path),
            "schema_version": metadata["schema_version"],
        },
        "config": config.public_dict(),
        "resolved_device": resolved_device,
        "losses": losses,
        "training_action_agreement": agreement,
    }
    metadata_path = output / "training.json"
    metadata_path.write_text(
        json.dumps(training_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return BayesianStudentArtifact(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        architecture=config.architecture,
        losses=tuple(losses),
        training_action_agreement=agreement,
    )


def load_bayesian_student_policy(
    topology: Topology,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> BayesianStudentPolicy:
    """Load one checkpoint while retaining public topology and mask handling."""
    torch = _require_torch()
    resolved_device = _resolve_device(torch, device)
    payload = torch.load(
        Path(checkpoint_path), map_location=resolved_device, weights_only=True
    )
    if payload.get("schema_version") != BAYESIAN_STUDENT_SCHEMA_VERSION:
        raise ValueError("unsupported Bayesian student checkpoint")
    architecture = payload.get("architecture")
    if architecture not in {"cnn", "gnn"}:
        raise ValueError("checkpoint architecture is invalid")
    network = build_bayesian_student(
        topology,
        architecture=architecture,
        observation_channels=int(payload["observation_channels"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(resolved_device)
    network.load_state_dict(payload["state_dict"])
    return BayesianStudentPolicy(
        network=network,
        device=resolved_device,
        policy_id=f"bayesian-{architecture}-student-v1",
    )


def evaluate_bayesian_student(
    topology: Topology,
    policy: BayesianStudentPolicy,
    *,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    """Evaluate only on an explicit validation seed list, never a test split."""
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(seed < 0 for seed in seeds)
    ):
        raise ValueError("validation requires at least two unique non-negative seeds")
    episodes: list[dict[str, object]] = []
    for seed in seeds:
        env = AttackEnv(topology)
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        hit_segments = discovery_area = 0
        first_hit: int | None = None
        while not (terminated or truncated):
            action = policy.select_action(observation, env.action_masks())
            observation, _, terminated, truncated, info = env.step(action)
            if bool(info["is_hit"]):
                hit_segments += 1
                first_hit = int(info["valid_shots"]) if first_hit is None else first_hit
            discovery_area += hit_segments
        shots = int(info["valid_shots"])
        discovery_area += (topology.valid_cell_count - shots) * hit_segments
        episodes.append(
            {
                "seed": seed,
                "valid_shots": shots,
                "auc_discovery": discovery_area / (17 * topology.valid_cell_count),
                "first_hit_shot": first_hit,
                "won": bool(terminated),
            }
        )
    return {
        "split": "validation",
        "blind_test_used": False,
        "policy_id": policy.policy_id,
        "episodes": episodes,
        "mean_valid_shots": fmean(float(item["valid_shots"]) for item in episodes),
        "mean_auc_discovery": fmean(float(item["auc_discovery"]) for item in episodes),
    }


def teacher_action_agreement(
    policy: BayesianStudentPolicy,
    demonstrations: BayesianDemonstrations,
) -> float:
    """Report agreement against a held-out public teacher demonstration set."""
    return _action_agreement(policy.network, demonstrations, device=policy.device)


def _normalise_public_scores(scores: Any, masks: Any, torch: Any) -> Any:
    masked_scores = torch.where(masks, scores, torch.zeros_like(scores))
    totals = masked_scores.sum(dim=1, keepdim=True)
    uniform = masks.float() / masks.float().sum(dim=1, keepdim=True)
    return torch.where(totals > 0.0, masked_scores / totals.clamp_min(1e-12), uniform)


def _action_agreement(
    network: Any, demonstrations: BayesianDemonstrations, *, device: str
) -> float:
    torch = _require_torch()
    network.eval()
    with torch.no_grad():
        observations = torch.as_tensor(
            demonstrations.observations, dtype=torch.float32, device=device
        )
        masks = torch.as_tensor(
            demonstrations.action_masks, dtype=torch.bool, device=device
        )
        predicted = masked_argmax(network(observations), masks)
        expected = torch.as_tensor(demonstrations.teacher_actions, device=device)
        return float((predicted == expected).float().mean().cpu())


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if requested.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA requested for Bayesian student but is unavailable")
    return requested
