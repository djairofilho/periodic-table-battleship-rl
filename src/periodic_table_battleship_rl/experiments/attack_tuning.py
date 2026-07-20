"""Reproducible attack-PPO hyperparameter selection on validation data only.

The search has a deliberately narrow responsibility: train one candidate for
each configured training seed, evaluate it on a fixed validation schedule,
and select the candidate with the lowest mean ``valid_shots``.  Test seeds are
not represented by these schemas or APIs, so callers must perform their blind
evaluation in a separate step after selection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from types import MappingProxyType
from typing import Any

from periodic_table_battleship_rl.evaluation.schemas import (
    EpisodeResult,
    HardwareMetadata,
    RunConfig,
    SoftwareMetadata,
    canonical_json,
)
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.topology import Topology
from periodic_table_battleship_rl.training.attack import (
    ATTACK_POLICY_ID,
    AttackTrainingArtifact,
    AttackTrainingConfig,
    MaskableAttackPolicy,
    load_attack_policy,
    train_attack_policy,
)

from .attack_baselines import ENVIRONMENT_VERSION
from .ppo_evaluation import PpoAttackEvaluation, run_ppo_attack_evaluation


ATTACK_TUNING_SCHEMA_VERSION = "attack-hyperparameter-search-v1"


def _frozen_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make provenance immutable after a trial has been accepted."""
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackHyperparameterCandidate:
    """One public PPO configuration eligible for validation selection."""

    candidate_id: str
    total_timesteps: int
    n_steps: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.batch_size <= 0 or self.batch_size > self.n_steps:
            raise ValueError("batch_size must be positive and no greater than n_steps")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.device.strip():
            raise ValueError("device must not be empty")

    def make_training_config(
        self,
        *,
        run_id: str,
        seed: int,
        checkpoint_directory: str | Path,
    ) -> AttackTrainingConfig:
        """Create the concrete training configuration for one train seed."""
        return AttackTrainingConfig(
            run_id=run_id,
            seed=seed,
            total_timesteps=self.total_timesteps,
            checkpoint_directory=Path(checkpoint_directory),
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
        )

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a stable, JSON-native candidate description."""
        return {
            "candidate_id": self.candidate_id,
            "total_timesteps": self.total_timesteps,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackTuningConfig:
    """Fixed train/validation schedule for one attack tuning search.

    There is intentionally no test-seed field.  Its absence makes accidental
    selection against held-out test episodes impossible through this module.
    """

    search_id: str
    scenario: str
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    validation_episodes_per_seed: int
    selection_metric: str = "mean_valid_shots"
    schema_version: str = ATTACK_TUNING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.search_id.strip():
            raise ValueError("search_id must not be empty")
        if not self.scenario.strip():
            raise ValueError("scenario must not be empty")
        _validate_seeds(self.training_seeds, "training_seeds")
        _validate_seeds(self.validation_seeds, "validation_seeds")
        if set(self.training_seeds) & set(self.validation_seeds):
            raise ValueError("training and validation seeds must not overlap")
        if self.validation_episodes_per_seed <= 0:
            raise ValueError("validation_episodes_per_seed must be positive")
        if self.selection_metric != "mean_valid_shots":
            raise ValueError("selection_metric must be 'mean_valid_shots'")
        if self.schema_version != ATTACK_TUNING_SCHEMA_VERSION:
            raise ValueError("unsupported attack tuning schema version")

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted public schedule."""
        return {
            "schema_version": self.schema_version,
            "search_id": self.search_id,
            "scenario": self.scenario,
            "training_seeds": list(self.training_seeds),
            "validation_seeds": list(self.validation_seeds),
            "validation_episodes_per_seed": self.validation_episodes_per_seed,
            "selection_metric": self.selection_metric,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackTuningTrialRequest:
    """A single candidate/train-seed job supplied to an injectable executor."""

    config: AttackTuningConfig
    candidate: AttackHyperparameterCandidate
    training_seed: int


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackTuningTrial:
    """Public validation output for one candidate and training seed."""

    candidate_id: str
    training_seed: int
    training_run_id: str
    validation_run_id: str
    results: tuple[EpisodeResult, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    split: str = "validation"

    def __post_init__(self) -> None:
        for name in ("candidate_id", "training_run_id", "validation_run_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")
        if self.split != "validation":
            raise ValueError("attack tuning trials must use the validation split")
        if not self.results:
            raise ValueError("results must contain at least one validation episode")
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))

    @property
    def mean_valid_shots(self) -> float:
        """Primary selection metric, calculated only from public results."""
        return fmean(float(result.valid_shots) for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact, registry-ready validation trial record."""
        return {
            "candidate_id": self.candidate_id,
            "training_seed": self.training_seed,
            "training_run_id": self.training_run_id,
            "validation_run_id": self.validation_run_id,
            "split": self.split,
            "episode_ids": [result.episode_id for result in self.results],
            "episode_count": len(self.results),
            "mean_valid_shots": self.mean_valid_shots,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackCandidateScore:
    """Candidate-level validation aggregate used for deterministic ranking."""

    candidate_id: str
    trial_count: int
    episode_count: int
    mean_valid_shots: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-native candidate score."""
        return {
            "candidate_id": self.candidate_id,
            "trial_count": self.trial_count,
            "episode_count": self.episode_count,
            "mean_valid_shots": self.mean_valid_shots,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackTuningResult:
    """Complete, registrable train/validation search result."""

    config: AttackTuningConfig
    candidates: tuple[AttackHyperparameterCandidate, ...]
    trials: tuple[AttackTuningTrial, ...]
    ranking: tuple[AttackCandidateScore, ...]
    selected_candidate_id: str

    def __post_init__(self) -> None:
        if not self.selected_candidate_id.strip():
            raise ValueError("selected_candidate_id must not be empty")
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        if self.selected_candidate_id not in candidate_ids:
            raise ValueError("selected_candidate_id must identify a configured candidate")
        if not self.ranking or self.ranking[0].candidate_id != self.selected_candidate_id:
            raise ValueError("selected_candidate_id must be the first ranked candidate")

    @property
    def selected_candidate(self) -> AttackHyperparameterCandidate:
        """Return the selected public hyperparameter configuration."""
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selected_candidate_id
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete search ledger without hidden game state."""
        return {
            "schema_version": ATTACK_TUNING_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "trials": [trial.to_dict() for trial in self.trials],
            "ranking": [score.to_dict() for score in self.ranking],
            "selected_candidate_id": self.selected_candidate_id,
        }


AttackTuningExecutor = Callable[[AttackTuningTrialRequest], AttackTuningTrial]


def run_attack_hyperparameter_search(
    config: AttackTuningConfig,
    topology: Topology,
    candidates: Sequence[AttackHyperparameterCandidate],
    executor: AttackTuningExecutor,
) -> AttackTuningResult:
    """Execute and select an attack search using only train/validation data.

    Candidates are ordered by ID before execution.  This gives reproducible
    execution and a deterministic lexical tie break for identical validation
    scores, independent of the caller's input order.
    """
    if topology.name != config.scenario:
        raise ValueError("topology name must match the tuning config scenario")
    ordered_candidates = _ordered_candidates(candidates)
    trials: list[AttackTuningTrial] = []
    for candidate in ordered_candidates:
        for training_seed in config.training_seeds:
            request = AttackTuningTrialRequest(
                config=config,
                candidate=candidate,
                training_seed=training_seed,
            )
            trial = executor(request)
            _validate_trial(trial, request, topology)
            trials.append(trial)
    return select_attack_hyperparameters(config, ordered_candidates, tuple(trials))


def select_attack_hyperparameters(
    config: AttackTuningConfig,
    candidates: Sequence[AttackHyperparameterCandidate],
    trials: Sequence[AttackTuningTrial],
) -> AttackTuningResult:
    """Rank complete validation trials by lower mean number of valid shots."""
    ordered_candidates = _ordered_candidates(candidates)
    expected_keys = {
        (candidate.candidate_id, training_seed)
        for candidate in ordered_candidates
        for training_seed in config.training_seeds
    }
    actual_keys = {(trial.candidate_id, trial.training_seed) for trial in trials}
    if actual_keys != expected_keys or len(trials) != len(expected_keys):
        raise ValueError("trials must contain exactly one result per candidate and training seed")
    for trial in trials:
        _validate_trial_results(trial, config, config.scenario)

    candidate_scores = tuple(
        _candidate_score(candidate.candidate_id, trials) for candidate in ordered_candidates
    )
    ranking = tuple(
        sorted(candidate_scores, key=lambda score: (score.mean_valid_shots, score.candidate_id))
    )
    ordered_trials = tuple(sorted(trials, key=lambda trial: (trial.candidate_id, trial.training_seed)))
    return AttackTuningResult(
        config=config,
        candidates=ordered_candidates,
        trials=ordered_trials,
        ranking=ranking,
        selected_candidate_id=ranking[0].candidate_id,
    )


def persist_attack_tuning_result(
    directory: str | Path, result: AttackTuningResult
) -> Path:
    """Atomically write the public tuning ledger and return its path."""
    return write_json_atomic(Path(directory) / "attack-tuning.json", result)


@dataclass(frozen=True, slots=True, kw_only=True)
class PpoAttackTuningExecutor:
    """Production executor with injectable train/load/evaluate operations."""

    topology: Topology
    checkpoint_directory: Path
    validation_directory: Path
    git_commit: str
    uv_lock_path: Path
    software: SoftwareMetadata | None = None
    hardware: HardwareMetadata | None = None
    trainer: Callable[[Topology, AttackTrainingConfig], AttackTrainingArtifact] = (
        train_attack_policy
    )
    policy_loader: Callable[..., MaskableAttackPolicy] = load_attack_policy
    evaluator: Callable[..., PpoAttackEvaluation] = run_ppo_attack_evaluation

    def __call__(self, request: AttackTuningTrialRequest) -> AttackTuningTrial:
        """Train once, then evaluate exactly once on the validation schedule."""
        if request.config.scenario != self.topology.name:
            raise ValueError("executor topology does not match the tuning scenario")
        training_run_id = (
            f"{request.config.search_id}-{request.candidate.candidate_id}-"
            f"train-{request.training_seed}"
        )
        training_config = request.candidate.make_training_config(
            run_id=training_run_id,
            seed=request.training_seed,
            checkpoint_directory=self.checkpoint_directory,
        )
        artifact = self.trainer(self.topology, training_config)
        policy = self.policy_loader(
            artifact.checkpoint_path,
            device=request.candidate.device,
        )
        validation_run_id = f"{training_run_id}-validation"
        validation_config = RunConfig(
            run_id=validation_run_id,
            experiment="attack",
            scenario=request.config.scenario,
            environment_version=ENVIRONMENT_VERSION,
            policy_id=ATTACK_POLICY_ID,
            split="validation",
            seeds=request.config.validation_seeds,
            episodes_per_seed=request.config.validation_episodes_per_seed,
            parameters={
                "tuning_search_id": request.config.search_id,
                "candidate_id": request.candidate.candidate_id,
                "training_seed": request.training_seed,
            },
        )
        evaluation = self.evaluator(
            validation_config,
            self.topology,
            policy,
            self.validation_directory / validation_run_id,
            checkpoint_path=artifact.checkpoint_path,
            training_metadata_path=artifact.metadata_path,
            git_commit=self.git_commit,
            uv_lock_path=self.uv_lock_path,
            software=self.software,
            hardware=self.hardware,
        )
        return AttackTuningTrial(
            candidate_id=request.candidate.candidate_id,
            training_seed=request.training_seed,
            training_run_id=training_run_id,
            validation_run_id=validation_run_id,
            results=evaluation.results,
            provenance={
                "checkpoint_path": str(artifact.checkpoint_path),
                "training_metadata_path": str(artifact.metadata_path),
                "validation_manifest_path": str(evaluation.persisted.manifest_path),
                "validation_episodes_path": str(evaluation.persisted.episodes_path),
            },
        )


def _validate_seeds(seeds: tuple[int, ...], field_name: str) -> None:
    if not seeds:
        raise ValueError(f"{field_name} must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{field_name} must not contain duplicates")
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError(f"{field_name} must contain non-negative integers")


def _ordered_candidates(
    candidates: Sequence[AttackHyperparameterCandidate],
) -> tuple[AttackHyperparameterCandidate, ...]:
    if not candidates:
        raise ValueError("candidates must contain at least one configuration")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))


def _validate_trial(
    trial: AttackTuningTrial,
    request: AttackTuningTrialRequest,
    topology: Topology,
) -> None:
    if trial.candidate_id != request.candidate.candidate_id:
        raise ValueError("trial candidate_id does not match the requested candidate")
    if trial.training_seed != request.training_seed:
        raise ValueError("trial training_seed does not match the requested seed")
    _validate_trial_results(trial, request.config, topology.name)


def _validate_trial_results(
    trial: AttackTuningTrial,
    config: AttackTuningConfig,
    scenario: str,
) -> None:
    expected_count = (
        len(config.validation_seeds) * config.validation_episodes_per_seed
    )
    if len(trial.results) != expected_count:
        raise ValueError("trial results do not match the fixed validation schedule")
    expected_seed_counts = {
        seed: config.validation_episodes_per_seed for seed in config.validation_seeds
    }
    actual_seed_counts = {seed: 0 for seed in config.validation_seeds}
    episode_ids: set[str] = set()
    for result in trial.results:
        if result.scenario != scenario:
            raise ValueError("validation result scenario does not match the topology")
        if result.seed not in actual_seed_counts:
            raise ValueError("validation result uses a seed outside the fixed schedule")
        if result.episode_id in episode_ids:
            raise ValueError("validation result episode IDs must be unique")
        episode_ids.add(result.episode_id)
        actual_seed_counts[result.seed] += 1
    if actual_seed_counts != expected_seed_counts:
        raise ValueError("validation result seed counts do not match the fixed schedule")


def _candidate_score(
    candidate_id: str, trials: Sequence[AttackTuningTrial]
) -> AttackCandidateScore:
    candidate_trials = tuple(trial for trial in trials if trial.candidate_id == candidate_id)
    results = tuple(result for trial in candidate_trials for result in trial.results)
    return AttackCandidateScore(
        candidate_id=candidate_id,
        trial_count=len(candidate_trials),
        episode_count=len(results),
        mean_valid_shots=fmean(float(result.valid_shots) for result in results),
    )


def attack_tuning_canonical_json(result: AttackTuningResult) -> str:
    """Expose canonical serialization for registry or content-addressed storage."""
    return canonical_json(result)
