"""Immutable v0.5 contracts for comparable, leakage-free experiments.

The objects in this module are intentionally independent from a training
library.  They are small JSON-serializable records that a runner can persist
before the blind test starts.  Their validation makes two mistakes explicit:
using a test seed to select a candidate and emitting an artifact without the
information required to reproduce it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping


PROTOCOL_VERSION = "experiment-v0.5"
_DECISIONS = frozenset({"promoted", "rejected"})
_DIRECTIONS = frozenset({"minimize", "maximize"})


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_hex_commit(value: str) -> None:
    if not 7 <= len(value) <= 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("git_commit must be a 7- to 64-character lowercase hex ID")


def _validate_seed_group(seeds: tuple[int, ...], name: str) -> None:
    if not seeds:
        raise ValueError(f"{name} seeds must not be empty")
    if any(not isinstance(seed, int) for seed in seeds):
        raise TypeError(f"{name} seeds must contain integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} seeds must not contain duplicates")


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            definition.name: _json_ready(getattr(value, definition.name))
            for definition in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class SeedInventory:
    """Four mutually exclusive seed inventories with distinct responsibilities."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    demonstration: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "demonstration": self.demonstration,
        }
        for name, seeds in groups.items():
            _validate_seed_group(seeds, name)
        seen: set[int] = set()
        for name, seeds in groups.items():
            overlap = seen.intersection(seeds)
            if overlap:
                raise ValueError(
                    f"{name} seeds overlap an earlier split: {sorted(overlap)}"
                )
            seen.update(seeds)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointPlan:
    """A pre-registered checkpoint schedule and validation-only selector."""

    steps: tuple[int, ...]
    metric: str
    direction: Literal["minimize", "maximize"]
    selection_split: Literal["validation"] = "validation"

    def __post_init__(self) -> None:
        if not self.steps or any(step <= 0 for step in self.steps):
            raise ValueError("steps must contain positive checkpoint steps")
        if tuple(sorted(self.steps)) != self.steps or len(set(self.steps)) != len(self.steps):
            raise ValueError("steps must be unique and sorted in ascending order")
        _require_text(self.metric, "metric")
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}")
        if self.selection_split != "validation":
            raise ValueError("checkpoints may be selected only on validation")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateRegistration:
    """Selection evidence recorded before a candidate receives a blind test."""

    record_id: str
    candidate_id: str
    control_id: str
    selected_checkpoint_step: int
    metric: str
    selection_seeds: tuple[int, ...]
    selection_split: Literal["validation"] = "validation"

    def __post_init__(self) -> None:
        for name in ("record_id", "candidate_id", "control_id", "metric"):
            _require_text(getattr(self, name), name)
        if self.candidate_id == self.control_id:
            raise ValueError("candidate_id and control_id must differ")
        if self.selected_checkpoint_step <= 0:
            raise ValueError("selected_checkpoint_step must be positive")
        _validate_seed_group(self.selection_seeds, "selection")
        if self.selection_split != "validation":
            raise ValueError("candidate selection is allowed only on validation")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class TestConfirmation:
    """A blind test tied to the persisted validation-selection record."""

    __test__: ClassVar[bool] = False

    candidate_id: str
    selection_record_id: str
    test_seeds: tuple[int, ...]
    split: Literal["test"] = "test"

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.selection_record_id, "selection_record_id")
        _validate_seed_group(self.test_seeds, "test")
        if self.split != "test":
            raise ValueError("confirmation must use the blind test split")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PromotionDecision:
    """Final decision based on an authorized blind-test confirmation."""

    candidate_id: str
    decision: Literal["promoted", "rejected"]
    reason: str
    confirmation: TestConfirmation

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.reason, "reason")
        if self.decision not in _DECISIONS:
            raise ValueError(f"decision must be one of {sorted(_DECISIONS)}")
        if self.confirmation.candidate_id != self.candidate_id:
            raise ValueError("confirmation candidate_id must match the decision")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactProvenance:
    """Per-artifact provenance; no report relies on ambient machine state."""

    run_id: str
    git_commit: str
    uv_lock_sha256: str
    config_sha256: str
    hardware: Mapping[str, str]

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_hex_commit(self.git_commit)
        _require_sha256(self.uv_lock_sha256, "uv_lock_sha256")
        _require_sha256(self.config_sha256, "config_sha256")
        if not self.hardware or any(not key.strip() or not value.strip() for key, value in self.hardware.items()):
            raise ValueError("hardware must contain non-empty string keys and values")
        object.__setattr__(self, "hardware", _freeze_mapping(self.hardware))

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRecord:
    """A public artifact and the exact run that created it."""

    artifact_id: str
    kind: str
    relative_path: str
    sha256: str
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        for name in ("artifact_id", "kind", "relative_path"):
            _require_text(getattr(self, name), name)
        if self.relative_path.startswith(("/", "\\")) or ".." in self.relative_path.split("/"):
            raise ValueError("relative_path must stay below the experiment directory")
        _require_sha256(self.sha256, "sha256")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentProtocol:
    """The complete pre-test contract for one v0.5 algorithm comparison."""

    experiment_id: str
    algorithm: str
    architecture: Mapping[str, Any]
    observation: Mapping[str, Any]
    reward: Mapping[str, Any]
    seeds: SeedInventory
    checkpoints: CheckpointPlan
    registration: CandidateRegistration
    artifacts: tuple[ArtifactRecord, ...]
    confirmation: TestConfirmation | None = None
    decision: PromotionDecision | None = None
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for name in ("experiment_id", "algorithm"):
            _require_text(getattr(self, name), name)
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
        for name in ("architecture", "observation", "reward"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, _freeze_mapping(values))
        if self.registration.selected_checkpoint_step not in self.checkpoints.steps:
            raise ValueError("selected checkpoint must be listed in the checkpoint plan")
        if self.registration.metric != self.checkpoints.metric:
            raise ValueError("registration metric must match the checkpoint plan")
        if not set(self.registration.selection_seeds).issubset(self.seeds.validation):
            raise ValueError("candidate selection may use only validation seeds")
        if not self.artifacts:
            raise ValueError("at least one artifact with provenance is required")
        artifact_ids = tuple(artifact.artifact_id for artifact in self.artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifact IDs must be unique")
        if self.confirmation is not None:
            self._validate_confirmation(self.confirmation)
        if self.decision is not None:
            if self.confirmation is None:
                raise ValueError("a promotion decision requires blind-test confirmation")
            if self.decision.confirmation != self.confirmation:
                raise ValueError("decision must reference the supplied confirmation")

    def _validate_confirmation(self, confirmation: TestConfirmation) -> None:
        if confirmation.candidate_id != self.registration.candidate_id:
            raise ValueError("confirmation candidate must match the registration")
        if confirmation.selection_record_id != self.registration.record_id:
            raise ValueError("confirmation must reference the persisted selection record")
        if confirmation.test_seeds != self.seeds.test:
            raise ValueError("blind confirmation must use the complete fixed test inventory")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)
