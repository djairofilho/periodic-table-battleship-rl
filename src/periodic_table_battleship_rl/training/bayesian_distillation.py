"""Public Bayesian demonstrations for neural policy distillation.

The generator deliberately treats :class:`AttackEnv` as an opaque public
interface.  Each teacher decision is reconstructed from an observation and an
action mask, then persisted without a fleet, ship identity, occupancy target,
or private environment field.  The saved occupancy scores are the teacher's
Monte Carlo estimate, not privileged labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path

import numpy as np

from periodic_table_battleship_rl.belief.model import (
    PublicAttackState,
    sample_compatible_fleets,
)
from periodic_table_battleship_rl.belief.planners import probability_action
from periodic_table_battleship_rl.envs.attack import AttackEnvironmentConfig, AttackEnv
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.topology import Topology


BAYESIAN_DEMONSTRATION_SCHEMA_VERSION = "bayesian-public-demonstrations-v1"
BAYESIAN_PROBABILITY_TEACHER_ID = "belief_probability_mc-v1"
_PUBLIC_DATA_FIELDS = (
    "observations",
    "action_masks",
    "teacher_actions",
    "teacher_occupancy_probabilities",
)
_EXCLUDED_HIDDEN_FIELDS = (
    "fleet",
    "occupied_cells",
    "ship_ids",
    "ship_placements",
    "private_rewards",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class BayesianDemonstrationConfig:
    """A deterministic, public schedule for the probability teacher."""

    dataset_id: str
    seeds: tuple[int, ...]
    output_directory: Path
    sample_count: int = 32
    sampler_seed: int = 0
    max_restarts_per_sample: int = 128
    max_nodes_per_sample: int = 8_192
    environment_config: AttackEnvironmentConfig = field(default_factory=AttackEnvironmentConfig)

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and contain no duplicates")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if min(
            self.sample_count,
            self.max_restarts_per_sample,
            self.max_nodes_per_sample,
        ) <= 0:
            raise ValueError("Bayesian sampler limits must be positive")
        if self.sampler_seed < 0:
            raise ValueError("sampler_seed must be non-negative")
        object.__setattr__(self, "output_directory", Path(self.output_directory))

    def public_dict(self) -> dict[str, object]:
        """Return only values needed to replay the public teacher schedule."""
        values = asdict(self)
        values["output_directory"] = str(self.output_directory)
        values["seeds"] = list(self.seeds)
        values["environment_config"] = self.environment_config.public_dict()
        return values


@dataclass(frozen=True, slots=True)
class BayesianDemonstrationArtifact:
    """Completed dataset paths and dimensions for provenance consumers."""

    data_path: Path
    metadata_path: Path
    sample_count: int
    scenario: str
    data_sha256: str


@dataclass(frozen=True, slots=True)
class BayesianDemonstrations:
    """Validated in-memory public arrays used by a distillation trainer."""

    observations: np.ndarray
    action_masks: np.ndarray
    teacher_actions: np.ndarray
    teacher_occupancy_probabilities: np.ndarray

    @property
    def sample_count(self) -> int:
        return len(self.teacher_actions)


def generate_bayesian_demonstrations(
    topology: Topology,
    config: BayesianDemonstrationConfig,
) -> BayesianDemonstrationArtifact:
    """Persist public Bayesian teacher decisions for a fixed seed schedule.

    The finite fleet population is generated from ``PublicAttackState`` and
    the current action mask.  The environment is only reset, stepped, and
    asked for its documented public mask; no hidden attribute is read.
    """
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    occupancy_scores: list[np.ndarray] = []
    total_restarts = total_backtracks = 0

    for episode_seed in config.seeds:
        environment = AttackEnv(topology, config=config.environment_config)
        observation, _ = environment.reset(seed=episode_seed)
        teacher_rng = np.random.default_rng(
            np.random.SeedSequence((episode_seed, config.sampler_seed))
        )
        terminated = truncated = False
        while not (terminated or truncated):
            action_mask = environment.action_masks()
            state = PublicAttackState.from_observation(topology, observation)
            belief, diagnostics = sample_compatible_fleets(
                state,
                sample_count=config.sample_count,
                rng=teacher_rng,
                max_restarts_per_sample=config.max_restarts_per_sample,
                max_nodes_per_sample=config.max_nodes_per_sample,
            )
            scores = belief.action_probabilities(action_mask)
            action = probability_action(belief, action_mask)
            observations.append(observation.copy())
            masks.append(action_mask.copy())
            actions.append(action)
            occupancy_scores.append(scores.astype(np.float32, copy=False))
            total_restarts += diagnostics.restart_count
            total_backtracks += diagnostics.backtrack_count
            observation, _, terminated, truncated, _ = environment.step(action)

    if not actions:
        raise RuntimeError("Bayesian teacher schedule did not produce any decisions")
    output = config.output_directory / config.dataset_id
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "demonstrations.npz"
    np.savez_compressed(
        data_path,
        observations=np.stack(observations),
        action_masks=np.stack(masks),
        teacher_actions=np.asarray(actions, dtype=np.int64),
        teacher_occupancy_probabilities=np.stack(occupancy_scores),
    )
    data_sha256 = _sha256(data_path)
    metadata = {
        "schema_version": BAYESIAN_DEMONSTRATION_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "teacher_policy": BAYESIAN_PROBABILITY_TEACHER_ID,
        "scenario": topology.name,
        "sample_count": len(actions),
        "observation_shape": list(observations[0].shape),
        "action_count": topology.action_count,
        "schedule": config.public_dict(),
        "public_fields": list(_PUBLIC_DATA_FIELDS),
        "excluded_hidden_fields": list(_EXCLUDED_HIDDEN_FIELDS),
        "teacher": {
            "sampler_id": "constrained-backtracking-v1",
            "posterior_exact": False,
            "action_rule": "stable-argmax of public occupancy probabilities",
            "aggregate_restart_count": total_restarts,
            "aggregate_backtrack_count": total_backtracks,
        },
        "data_sha256": data_sha256,
    }
    metadata_path = write_json_atomic(output / "dataset.json", metadata)
    return BayesianDemonstrationArtifact(
        data_path=data_path,
        metadata_path=metadata_path,
        sample_count=len(actions),
        scenario=topology.name,
        data_sha256=data_sha256,
    )


def load_bayesian_demonstrations(path: str | Path) -> BayesianDemonstrations:
    """Load a dataset and reject hidden, malformed, or illegal records."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != set(_PUBLIC_DATA_FIELDS):
            raise ValueError("Bayesian demonstrations must contain exactly public fields")
        demonstrations = BayesianDemonstrations(
            observations=archive["observations"],
            action_masks=archive["action_masks"],
            teacher_actions=archive["teacher_actions"],
            teacher_occupancy_probabilities=archive["teacher_occupancy_probabilities"],
        )
    _validate_demonstrations(demonstrations)
    return demonstrations


def load_bayesian_demonstration_metadata(path: str | Path) -> dict[str, object]:
    """Load and validate dataset metadata without opening a training stack."""
    import json

    metadata_path = Path(path).with_name("dataset.json")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("Bayesian dataset metadata must contain a JSON object")
    if metadata.get("schema_version") != BAYESIAN_DEMONSTRATION_SCHEMA_VERSION:
        raise ValueError("unsupported Bayesian demonstration schema version")
    if metadata.get("public_fields") != list(_PUBLIC_DATA_FIELDS):
        raise ValueError("Bayesian dataset metadata does not certify public fields")
    if metadata.get("excluded_hidden_fields") != list(_EXCLUDED_HIDDEN_FIELDS):
        raise ValueError("Bayesian dataset metadata lacks hidden-field exclusions")
    return metadata


def _validate_demonstrations(demonstrations: BayesianDemonstrations) -> None:
    observations = demonstrations.observations
    masks = demonstrations.action_masks
    actions = demonstrations.teacher_actions
    scores = demonstrations.teacher_occupancy_probabilities
    if observations.ndim != 4 or masks.ndim != 2 or actions.ndim != 1 or scores.ndim != 2:
        raise ValueError("Bayesian demonstration arrays have invalid dimensions")
    if not (len(observations) == len(masks) == len(actions) == len(scores)):
        raise ValueError("Bayesian demonstration arrays must have matching sample counts")
    if observations.dtype != np.uint8 or masks.dtype != np.bool_ or actions.dtype != np.int64:
        raise ValueError("observations, masks, and actions have unexpected dtypes")
    if scores.dtype not in (np.float32, np.float64):
        raise ValueError("teacher occupancy probabilities must be floating point")
    if masks.shape[1] != scores.shape[1]:
        raise ValueError("teacher occupancy probabilities must align with action masks")
    if actions.size and (int(actions.min()) < 0 or int(actions.max()) >= masks.shape[1]):
        raise ValueError("teacher action is outside the action mask")
    if actions.size and not np.all(masks[np.arange(len(actions)), actions]):
        raise ValueError("teacher action must be legal under its public mask")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("teacher occupancy probabilities must be finite values in [0, 1]")
    if np.any(scores[~masks] != 0.0):
        raise ValueError("teacher occupancy probabilities must be zero for masked actions")
    if actions.size:
        expected_actions = np.argmax(np.where(masks, scores, -np.inf), axis=1)
        if not np.array_equal(actions, expected_actions):
            raise ValueError(
                "teacher action must be the stable public occupancy argmax"
            )
