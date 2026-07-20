"""Seed-paired, validation-only protocol for the Bayesian attack planner.

The module deliberately has no path for a ``test`` split.  It is a reusable
pre-registration for selecting whether the public-history probability planner
has enough evidence to become a candidate; it must not be used to publish a
final blind result.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

import numpy as np

from periodic_table_battleship_rl.analysis.statistics import bootstrap_mean_interval
from periodic_table_battleship_rl.evaluation.schemas import EpisodeResult
from periodic_table_battleship_rl.topology import (
    BATTLESHIP,
    DENSE_118,
    PERIODIC_TABLE_BATTLESHIP,
    Topology,
)


BELIEF_VALIDATION_SCHEMA_VERSION = "bayes-cross-topology-validation-v1"
"""Versioned configuration used for candidate-selection evidence."""


@dataclass(frozen=True, slots=True)
class BayesianValidationProtocol:
    """Fixed public configuration for a paired validation campaign."""

    seed_start: int = 8_801
    seed_count: int = 10
    episodes_per_seed: int = 1
    sample_count: int = 16
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_720
    split: str = "validation"

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("Bayesian cross-topology runner is validation-only")
        if self.seed_start < 0:
            raise ValueError("seed_start must be non-negative")
        if self.seed_count < 2:
            raise ValueError("seed_count must be at least two")
        if self.episodes_per_seed <= 0 or self.sample_count <= 0:
            raise ValueError("episode and sample counts must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        return payload


CROSS_TOPOLOGY_BAYESIAN_VALIDATION = BayesianValidationProtocol()
"""Frozen default protocol.  New parameters require a new schema version."""


VALIDATION_TOPOLOGIES: tuple[Topology, ...] = (
    BATTLESHIP,
    DENSE_118,
    PERIODIC_TABLE_BATTLESHIP,
)


def paired_seed_comparison(
    candidate: Sequence[EpisodeResult],
    reference: Sequence[EpisodeResult],
    *,
    metric: str,
    direction: str,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Return a seed-paired percentile interval for public episode metrics.

    Runs intentionally have distinct IDs.  Pairing therefore uses the frozen
    common fleet seed inventory and checks that each seed has the same number
    of episodes before aggregating each arm within seed.
    """
    if direction not in {"lower", "higher"}:
        raise ValueError("direction must be 'lower' or 'higher'")
    candidate_by_seed = _values_by_seed(candidate, metric=metric)
    reference_by_seed = _values_by_seed(reference, metric=metric)
    if candidate_by_seed.keys() != reference_by_seed.keys():
        raise ValueError("candidate and reference must share an identical seed inventory")
    differences: list[float] = []
    by_seed: list[dict[str, float | int]] = []
    for seed in sorted(candidate_by_seed):
        candidate_values = candidate_by_seed[seed]
        reference_values = reference_by_seed[seed]
        if len(candidate_values) != len(reference_values):
            raise ValueError(f"seed {seed} has unpaired episode counts")
        candidate_mean = fmean(candidate_values)
        reference_mean = fmean(reference_values)
        difference = candidate_mean - reference_mean
        differences.append(difference)
        by_seed.append(
            {
                "seed": seed,
                "candidate_mean": candidate_mean,
                "reference_mean": reference_mean,
                "candidate_minus_reference": difference,
            }
        )
    interval = bootstrap_mean_interval(
        differences,
        rng=rng,
        resamples=bootstrap_resamples,
    )
    improvement = interval.upper < 0.0 if direction == "lower" else interval.lower > 0.0
    return {
        "metric": metric,
        "direction": direction,
        "candidate_minus_reference_mean": interval.mean,
        "bootstrap_95": {
            "lower": interval.lower,
            "upper": interval.upper,
            "resamples": interval.resamples,
            "confidence_level": interval.confidence_level,
        },
        "improves_reference_at_95": improvement,
        "per_seed": by_seed,
    }


def _values_by_seed(
    results: Sequence[EpisodeResult], *, metric: str
) -> dict[int, list[float]]:
    if not results:
        raise ValueError("results must not be empty")
    values: dict[int, list[float]] = defaultdict(list)
    ids: set[str] = set()
    for result in results:
        if result.episode_id in ids:
            raise ValueError(f"duplicate episode_id: {result.episode_id}")
        ids.add(result.episode_id)
        value = getattr(result, metric, None)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"metric {metric!r} must be numeric and non-boolean")
        values[result.seed].append(float(value))
    return dict(values)
