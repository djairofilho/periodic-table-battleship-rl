"""Atomic UTF-8 persistence for reproducible benchmark runs.

The manifest is written last by :func:`persist_run`.  Consumers can therefore
treat its presence as the completion marker for a run directory: an interrupted
write may leave an orphaned temporary file, but never a partial JSON artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from .schemas import (
    EpisodeResult,
    PlacementResult,
    RunManifest,
    canonical_json,
    sha256_file,
)


ResultRecord = EpisodeResult | PlacementResult


@dataclass(frozen=True, slots=True)
class PersistedRun:
    """Paths and content hashes for a completed run-directory write."""

    manifest_path: Path
    episodes_path: Path
    manifest_sha256: str
    episodes_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """JSON-native contents recovered from a persisted benchmark run."""

    manifest: Mapping[str, Any]
    episodes: tuple[Mapping[str, Any], ...]


def _atomic_write_bytes(path: Path, contents: bytes) -> Path:
    """Replace ``path`` only after its complete contents reach a sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def write_json_atomic(path: str | Path, value: Any) -> Path:
    """Write canonical JSON as UTF-8, atomically replacing an existing file."""
    destination = Path(path)
    payload = canonical_json(value).encode("utf-8") + b"\n"
    return _atomic_write_bytes(destination, payload)


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document emitted by :func:`write_json_atomic`."""
    with Path(path).open(encoding="utf-8") as json_file:
        return json.load(json_file)


def write_jsonl_atomic(path: str | Path, records: Sequence[Any]) -> Path:
    """Write canonical JSON Lines as UTF-8, atomically replacing an old file."""
    destination = Path(path)
    payload = b"".join(
        canonical_json(record).encode("utf-8") + b"\n" for record in records
    )
    return _atomic_write_bytes(destination, payload)


def read_jsonl(path: str | Path) -> tuple[Any, ...]:
    """Read a UTF-8 JSON Lines file emitted by :func:`write_jsonl_atomic`."""
    with Path(path).open(encoding="utf-8") as json_file:
        return tuple(json.loads(line) for line in json_file if line.strip())


def persist_run(
    run_directory: str | Path,
    manifest: RunManifest,
    results: Sequence[ResultRecord],
) -> PersistedRun:
    """Persist public episode records and their manifest in a run directory.

    The result IDs must match the manifest exactly and in order.  Episode
    records are committed first; the manifest is the final completion marker.
    """
    records = tuple(results)
    episode_ids = tuple(record.episode_id for record in records)
    if episode_ids != manifest.episodes.episode_ids:
        raise ValueError("result episode IDs must match the manifest order exactly")

    directory = Path(run_directory)
    episodes_path = write_jsonl_atomic(directory / "episodes.jsonl", records)
    manifest_path = write_json_atomic(directory / "manifest.json", manifest)
    return PersistedRun(
        manifest_path=manifest_path,
        episodes_path=episodes_path,
        manifest_sha256=sha256_file(manifest_path),
        episodes_sha256=sha256_file(episodes_path),
    )


def load_run(run_directory: str | Path) -> LoadedRun:
    """Load public JSON records from a completed run directory.

    This intentionally returns JSON-native mappings.  Schema constructors own
    domain validation, while storage stays forward-compatible with future
    schema versions.
    """
    directory = Path(run_directory)
    manifest = read_json(directory / "manifest.json")
    episodes = read_jsonl(directory / "episodes.jsonl")
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    if any(not isinstance(episode, dict) for episode in episodes):
        raise ValueError("episodes.jsonl records must be JSON objects")
    return LoadedRun(manifest=manifest, episodes=episodes)
