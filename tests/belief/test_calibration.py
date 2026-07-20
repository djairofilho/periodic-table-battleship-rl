"""Regression tests for exact microboard Monte Carlo calibration."""

from __future__ import annotations

import pytest

from periodic_table_battleship_rl.belief.calibration import (
    calibrate_constrained_sampler,
    default_micro_calibration_cases,
    rectangular_micro_topology,
)


def test_default_calibration_spans_fixed_compatible_public_histories() -> None:
    cases = default_micro_calibration_cases()

    assert [case.name for case in cases] == [
        "prior",
        "active-hit-center",
        "miss-center",
        "hit-corner-miss-center",
        "sunk-top-edge",
        "two-ship-prior",
        "two-ship-active-hit-center",
    ]
    assert all(case.state.topology.valid_cell_count == 9 for case in cases)


def test_calibration_is_reproducible_and_samples_only_exact_support() -> None:
    first = calibrate_constrained_sampler(sample_count=32, repetitions=3, seed=87)
    second = calibrate_constrained_sampler(sample_count=32, repetitions=3, seed=87)

    assert first.to_dict() == second.to_dict()
    report = first.to_dict()
    assert report["posterior_exact"] is False
    assert report["blind_test_used"] is False
    for case in report["cases"]:
        assert case["exact_posterior"] is True
        assert case["mean_metrics"]["unexpected_sample_mass"] == 0.0
        assert case["mean_metrics"]["exact_support_coverage"] <= 1.0


def test_uniform_single_ship_prior_matches_oracle_fleet_count() -> None:
    result = calibrate_constrained_sampler(
        cases=(default_micro_calibration_cases()[0],),
        sample_count=64,
        repetitions=1,
        seed=1,
    )

    case = result.to_dict()["cases"][0]
    assert case["exact_fleet_count"] == 12
    assert case["mean_metrics"]["occupancy_mean_absolute_error"] >= 0.0
    assert case["mean_metrics"]["fleet_distribution_total_variation"] >= 0.0
    assert case["mean_metrics"]["ideal_iid_fleet_distribution_total_variation"] >= 0.0


@pytest.mark.parametrize("rows, columns", [(0, 3), (3, 0)])
def test_micro_topology_rejects_non_positive_dimensions(rows: int, columns: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        rectangular_micro_topology(rows, columns)
