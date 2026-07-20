"""Immutable, JSON-serializable records used by benchmark evaluations.

The schemas deliberately contain only public evaluation information. In
particular, an :class:`EpisodeResult` never contains an opponent's hidden
fleet. Placement experiments may record the evaluated placement because it is
the policy output being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = "evaluation-v1"
EXPERIMENTS = frozenset({"attack", "placement"})
SPLITS = frozenset({"train", "validation", "test"})


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy a mapping so later caller mutations cannot alter a record."""
    return MappingProxyType(dict(values))


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """The fixed public configuration of one train, validation, or test run."""

    run_id: str
    experiment: str
    scenario: str
    environment_version: str
    policy_id: str
    split: str
    seeds: tuple[int, ...]
    episodes_per_seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "experiment",
            "scenario",
            "environment_version",
            "policy_id",
            "split",
        ):
            _require_non_empty(getattr(self, name), name)
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f"experiment must be one of {sorted(EXPERIMENTS)}")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if any(not isinstance(seed, int) for seed in self.seeds):
            raise TypeError("seeds must contain integers")
        if self.episodes_per_seed <= 0:
            raise ValueError("episodes_per_seed must be positive")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeResult:
    """Public result of one attack-policy episode."""

    episode_id: str
    run_id: str
    seed: int
    scenario: str
    valid_cells: int
    valid_shots: int
    invalid_attempts: int
    hit_segments: int
    sunk_ship_lengths: tuple[int, ...]
    won: bool
    truncated: bool
    auc_discovery: float
    first_hit_shot: int | None = None
    first_sunk_shot: int | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "run_id", "scenario"):
            _require_non_empty(getattr(self, name), name)
        for name in (
            "valid_cells",
            "valid_shots",
            "invalid_attempts",
            "hit_segments",
        ):
            _require_non_negative(getattr(self, name), name)
        if self.valid_cells == 0:
            raise ValueError("valid_cells must be positive")
        if self.valid_shots > self.valid_cells:
            raise ValueError("valid_shots cannot exceed valid_cells")
        if self.hit_segments > 17:
            raise ValueError("hit_segments cannot exceed the 17 fleet segments")
        if not 0.0 <= self.auc_discovery <= 1.0:
            raise ValueError("auc_discovery must be in [0, 1]")
        for name in ("first_hit_shot", "first_sunk_shot"):
            value = getattr(self, name)
            if value is not None and not 1 <= value <= self.valid_shots:
                raise ValueError(f"{name} must be within the valid-shot range")
        if self.first_sunk_shot is not None and self.first_hit_shot is None:
            raise ValueError("first_sunk_shot requires first_hit_shot")
        if self.won and self.hit_segments != 17:
            raise ValueError("a won episode must hit all 17 fleet segments")
        if self.won and self.truncated:
            raise ValueError("an episode cannot be both won and truncated")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlacementResult:
    """Result of evaluating one placement-policy episode against one attacker."""

    episode_id: str
    run_id: str
    seed: int
    scenario: str
    attacker_id: str
    attacker_seed: int
    placement_actions: tuple[int, ...]
    valid_cells: int
    valid_shots_to_sink: int
    hit_segments: int
    sunk_ship_lengths: tuple[int, ...]
    auc_discovery: float
    first_hit_shot: int | None = None
    first_sunk_shot: int | None = None
    all_sunk_shot: int | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        for name in ("episode_id", "run_id", "scenario", "attacker_id"):
            _require_non_empty(getattr(self, name), name)
        if len(self.placement_actions) != 5:
            raise ValueError("placement_actions must contain one action per ship")
        if any(not 0 <= action < 360 for action in self.placement_actions):
            raise ValueError("placement_actions must be valid placement action IDs")
        for name in ("valid_cells", "valid_shots_to_sink", "hit_segments"):
            _require_non_negative(getattr(self, name), name)
        if self.valid_cells == 0:
            raise ValueError("valid_cells must be positive")
        if self.valid_shots_to_sink > self.valid_cells:
            raise ValueError("valid_shots_to_sink cannot exceed valid_cells")
        if self.hit_segments > 17:
            raise ValueError("hit_segments cannot exceed the 17 fleet segments")
        if not 0.0 <= self.auc_discovery <= 1.0:
            raise ValueError("auc_discovery must be in [0, 1]")
        for name in ("first_hit_shot", "first_sunk_shot", "all_sunk_shot"):
            value = getattr(self, name)
            if value is not None and not 1 <= value <= self.valid_shots_to_sink:
                raise ValueError(f"{name} must be within the valid-shot range")
        if self.first_sunk_shot is not None and self.first_hit_shot is None:
            raise ValueError("first_sunk_shot requires first_hit_shot")
        if self.all_sunk_shot is not None and self.hit_segments != 17:
            raise ValueError("all_sunk_shot requires all 17 fleet segments to be hit")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class SoftwareMetadata:
    """Software versions needed to reproduce a run."""

    python_version: str
    platform: str
    dependencies: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.python_version, "python_version")
        _require_non_empty(self.platform, "platform")
        object.__setattr__(self, "dependencies", _freeze_mapping(self.dependencies))

    @classmethod
    def current(cls, dependencies: Mapping[str, str] | None = None) -> SoftwareMetadata:
        return cls(
            python_version=platform.python_version(),
            platform=platform.platform(),
            dependencies={} if dependencies is None else dependencies,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class HardwareMetadata:
    """Machine information that may affect execution speed or determinism."""

    machine: str
    processor: str
    cpu_count: int | None
    accelerator: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.machine, "machine")
        if self.cpu_count is not None and self.cpu_count <= 0:
            raise ValueError("cpu_count must be positive when supplied")

    @classmethod
    def current(cls, accelerator: str | None = None) -> HardwareMetadata:
        return cls(
            machine=platform.machine() or "unknown",
            processor=platform.processor() or "unknown",
            cpu_count=os.cpu_count(),
            accelerator=accelerator,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeManifest:
    """Ordered list of episode IDs emitted by a single evaluation run."""

    run_id: str
    episode_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        if not self.episode_ids:
            raise ValueError("episode_ids must not be empty")
        if any(not episode_id.strip() for episode_id in self.episode_ids):
            raise ValueError("episode_ids must not contain empty IDs")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("episode_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunManifest:
    """Provenance and episode inventory required to reproduce an evaluation."""

    config: RunConfig
    git_commit: str
    uv_lock_sha256: str
    software: SoftwareMetadata
    hardware: HardwareMetadata
    episodes: EpisodeManifest
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.git_commit, "git_commit")
        if len(self.uv_lock_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.uv_lock_sha256
        ):
            raise ValueError("uv_lock_sha256 must be a lowercase SHA-256 digest")
        if self.episodes.run_id != self.config.run_id:
            raise ValueError("episode manifest run_id must match config run_id")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


def _json_ready(value: Any) -> Any:
    """Convert schema objects and tuples into JSON-native values recursively."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            definition.name: _json_ready(getattr(value, definition.name))
            for definition in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return stable UTF-8-safe JSON for manifests and result records."""
    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_file(path: str | Path) -> str:
    """Compute the content hash recorded for the resolved ``uv.lock`` file."""
    digest = sha256()
    with Path(path).open("rb") as lock_file:
        for block in iter(lambda: lock_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
