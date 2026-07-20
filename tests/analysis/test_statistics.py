from __future__ import annotations

import numpy as np
import pytest

from periodic_table_battleship_rl.analysis.statistics import (
    bootstrap_mean_interval,
    paired_difference_by_seed,
    summarize_metric_by_seed,
)
from periodic_table_battleship_rl.evaluation.schemas import EpisodeResult


def _episode(episode_id: str, seed: int, shots: int) -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id,
        run_id="run",
        seed=seed,
        scenario="battleship",
        valid_cells=100,
        valid_shots=shots,
        invalid_attempts=0,
        hit_segments=17,
        sunk_ship_lengths=(2, 3, 3, 4, 5),
        won=True,
        truncated=False,
        auc_discovery=0.5,
    )


def test_summarize_metric_by_seed_orders_and_aggregates() -> None:
    summaries = summarize_metric_by_seed(
        [_episode("b", 9, 20), _episode("a", 3, 10), _episode("c", 9, 30)],
        "valid_shots",
    )

    assert [(item.seed, item.episode_count, item.mean) for item in summaries] == [
        (3, 1, 10.0),
        (9, 2, 25.0),
    ]


def test_paired_difference_requires_matching_ids_and_seeds() -> None:
    candidate = [_episode("episode-a", 1, 30)]

    with pytest.raises(ValueError, match="identical episode IDs"):
        paired_difference_by_seed(candidate, [_episode("episode-b", 1, 20)], metric="valid_shots")
    with pytest.raises(ValueError, match="incompatible seeds"):
        paired_difference_by_seed(candidate, [_episode("episode-a", 2, 20)], metric="valid_shots")


def test_paired_difference_is_seed_aggregated_and_directional() -> None:
    difference = paired_difference_by_seed(
        [_episode("a", 7, 20), _episode("b", 7, 30), _episode("c", 9, 50)],
        [_episode("a", 7, 25), _episode("b", 7, 40), _episode("c", 9, 55)],
        metric="valid_shots",
        direction="lower",
    )

    assert [(item.seed, item.mean) for item in difference.by_seed] == [(7, -7.5), (9, -5.0)]
    assert difference.mean_difference == -6.25
    assert difference.is_improvement


def test_bootstrap_interval_is_reproducible_with_explicit_rng() -> None:
    first = bootstrap_mean_interval(
        [1.0, 2.0, 4.0, 8.0], rng=np.random.default_rng(20260720), resamples=500
    )
    second = bootstrap_mean_interval(
        [1.0, 2.0, 4.0, 8.0], rng=np.random.default_rng(20260720), resamples=500
    )

    assert first == second
    assert first.mean == 3.75
    assert first.lower <= first.mean <= first.upper


def test_statistics_rejects_duplicate_ids_and_non_numeric_metric() -> None:
    duplicate = [_episode("same", 1, 10), _episode("same", 2, 20)]

    with pytest.raises(ValueError, match="duplicate episode_id"):
        summarize_metric_by_seed(duplicate, "valid_shots")
    with pytest.raises(ValueError, match="must be numeric"):
        summarize_metric_by_seed([_episode("one", 1, 10)], "won")
