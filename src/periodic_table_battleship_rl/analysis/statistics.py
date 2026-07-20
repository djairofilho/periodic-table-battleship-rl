"""Deterministic, dependency-light statistics for blind RL evaluations.

The functions in this module deliberately aggregate only public result records.
They do not inspect fleets or policy internals, so the same code can be used for
attack and placement experiments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

import numpy as np

from periodic_table_battleship_rl.evaluation.schemas import (
    EpisodeResult,
    PlacementResult,
)


EvaluationResult: TypeAlias = EpisodeResult | PlacementResult
MetricDirection: TypeAlias = Literal["higher", "lower"]


@dataclass(frozen=True, slots=True)
class SeedMetricSummary:
    """Mean of one numeric metric for all episodes emitted by one seed."""

    seed: int
    episode_count: int
    mean: float


@dataclass(frozen=True, slots=True)
class BootstrapMeanInterval:
    """Percentile bootstrap interval for an arithmetic mean."""

    mean: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """Seed-paired candidate-minus-reference metric difference."""

    metric: str
    direction: MetricDirection
    by_seed: tuple[SeedMetricSummary, ...]

    @property
    def mean_difference(self) -> float:
        """Return the unweighted mean across evaluation seeds."""
        return float(np.mean([summary.mean for summary in self.by_seed]))

    @property
    def is_improvement(self) -> bool:
        """Whether the average difference improves according to metric direction."""
        difference = self.mean_difference
        return difference > 0.0 if self.direction == "higher" else difference < 0.0


def summarize_metric_by_seed(
    results: Sequence[EvaluationResult], metric: str
) -> tuple[SeedMetricSummary, ...]:
    """Aggregate a public numeric result field by seed in deterministic order.

    ``results`` must have unique episode IDs. This catches accidental duplicate
    persistence before it can give a seed excess influence over a result.
    """
    if not results:
        raise ValueError("results must not be empty")

    seen_episode_ids: set[str] = set()
    values_by_seed: dict[int, list[float]] = defaultdict(list)
    for result in results:
        if result.episode_id in seen_episode_ids:
            raise ValueError(f"duplicate episode_id: {result.episode_id}")
        seen_episode_ids.add(result.episode_id)
        values_by_seed[result.seed].append(_metric_value(result, metric))

    return tuple(
        SeedMetricSummary(
            seed=seed,
            episode_count=len(values),
            mean=float(np.mean(values)),
        )
        for seed, values in sorted(values_by_seed.items())
    )


def paired_difference_by_seed(
    candidate: Sequence[EvaluationResult],
    reference: Sequence[EvaluationResult],
    *,
    metric: str,
    direction: MetricDirection = "higher",
) -> PairedDifference:
    """Compare two evaluations after requiring identical episode/seed pairs.

    Pairing by episode ID avoids silently comparing different sampled boards or
    attacker seeds. The candidate and reference may have different ``run_id``
    values, as expected for two policies.
    """
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be either 'higher' or 'lower'")
    candidate_by_id = _results_by_episode_id(candidate, label="candidate")
    reference_by_id = _results_by_episode_id(reference, label="reference")
    if candidate_by_id.keys() != reference_by_id.keys():
        missing_candidate = sorted(reference_by_id.keys() - candidate_by_id.keys())
        missing_reference = sorted(candidate_by_id.keys() - reference_by_id.keys())
        raise ValueError(
            "candidate and reference must contain identical episode IDs; "
            f"missing candidate={missing_candidate}, missing reference={missing_reference}"
        )

    values_by_seed: dict[int, list[float]] = defaultdict(list)
    for episode_id in sorted(candidate_by_id):
        candidate_result = candidate_by_id[episode_id]
        reference_result = reference_by_id[episode_id]
        if candidate_result.seed != reference_result.seed:
            raise ValueError(
                f"episode_id {episode_id!r} has incompatible seeds: "
                f"{candidate_result.seed} != {reference_result.seed}"
            )
        values_by_seed[candidate_result.seed].append(
            _metric_value(candidate_result, metric)
            - _metric_value(reference_result, metric)
        )

    return PairedDifference(
        metric=metric,
        direction=direction,
        by_seed=tuple(
            SeedMetricSummary(
                seed=seed,
                episode_count=len(values),
                mean=float(np.mean(values)),
            )
            for seed, values in sorted(values_by_seed.items())
        ),
    )


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
) -> BootstrapMeanInterval:
    """Return a deterministic percentile bootstrap interval for a mean.

    Callers supply the NumPy generator explicitly, for example
    ``np.random.default_rng(20260720)``. This makes every stochastic decision
    part of the experiment's recorded seed protocol.
    """
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(sample).all():
        raise ValueError("values must be finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    bootstrap_indices = rng.integers(0, sample.size, size=(resamples, sample.size))
    bootstrap_means = sample[bootstrap_indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, (alpha, 1.0 - alpha))
    return BootstrapMeanInterval(
        mean=float(sample.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        resamples=resamples,
    )


def _results_by_episode_id(
    results: Sequence[EvaluationResult], *, label: str
) -> dict[str, EvaluationResult]:
    if not results:
        raise ValueError(f"{label} results must not be empty")
    by_id = {result.episode_id: result for result in results}
    if len(by_id) != len(results):
        raise ValueError(f"{label} results contain duplicate episode IDs")
    return by_id


def _metric_value(result: EvaluationResult, metric: str) -> float:
    try:
        value = getattr(result, metric)
    except AttributeError as error:
        raise ValueError(
            f"metric {metric!r} is not available on {type(result).__name__}"
        ) from error
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"metric {metric!r} must be numeric and non-boolean")
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"metric {metric!r} must be finite")
    return value_float
