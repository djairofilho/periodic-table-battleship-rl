"""Finite-microboard calibration for the constrained Monte Carlo sampler.

The production sampler intentionally makes no posterior-exactness claim.  On
small boards we can enumerate the same compatible-fleet space exactly and
measure the proposal's empirical discrepancy without inspecting a game's
private fleet.  This module is deliberately a diagnostic: it does not change
the planner's action-selection behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from periodic_table_battleship_rl.belief.model import (
    PublicAttackState,
    exact_belief,
    sample_compatible_fleets,
)
from periodic_table_battleship_rl.game import Fleet, ShipSpec
from periodic_table_battleship_rl.topology import Cell, Topology


CALIBRATION_SCHEMA_VERSION = "belief-sampler-calibration-v1"


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    """One compatible public history and its fixed micro-fleet specification."""

    name: str
    state: PublicAttackState
    specs: tuple[ShipSpec, ...]


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """One replicate's empirical proposal discrepancy from exact belief."""

    occupancy_mean_absolute_error: float
    occupancy_root_mean_squared_error: float
    occupancy_max_absolute_error: float
    fleet_distribution_total_variation: float
    ideal_iid_fleet_distribution_total_variation: float
    fleet_distribution_tv_excess_vs_iid: float
    exact_support_coverage: float
    unexpected_sample_mass: float
    restart_count: int
    backtrack_count: int


@dataclass(frozen=True, slots=True)
class CalibrationCaseResult:
    """Summary and replicate-level evidence for one public history."""

    name: str
    exact_fleet_count: int
    exact_posterior: bool
    sample_count: int
    repetitions: int
    metrics: tuple[CalibrationMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe, reproducible evidence without rounding metrics."""
        metric_dicts = [asdict(metric) for metric in self.metrics]
        summary = {
            key: float(np.mean([metric[key] for metric in metric_dicts]))
            for key in (
                "occupancy_mean_absolute_error",
                "occupancy_root_mean_squared_error",
                "occupancy_max_absolute_error",
                "fleet_distribution_total_variation",
                "ideal_iid_fleet_distribution_total_variation",
                "fleet_distribution_tv_excess_vs_iid",
                "exact_support_coverage",
                "unexpected_sample_mass",
                "restart_count",
                "backtrack_count",
            )
        }
        return {
            "name": self.name,
            "exact_fleet_count": self.exact_fleet_count,
            "exact_posterior": self.exact_posterior,
            "sample_count": self.sample_count,
            "repetitions": self.repetitions,
            "mean_metrics": summary,
            "replicates": metric_dicts,
        }


@dataclass(frozen=True, slots=True)
class SamplerCalibration:
    """Calibration result for a fixed, non-blind microboard protocol."""

    cases: tuple[CalibrationCaseResult, ...]
    sample_count: int
    repetitions: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        """Serialize an auditable protocol result for reports and artifacts."""
        case_dicts = [case.to_dict() for case in self.cases]
        aggregate_keys = (
            "occupancy_mean_absolute_error",
            "occupancy_root_mean_squared_error",
            "occupancy_max_absolute_error",
            "fleet_distribution_total_variation",
            "ideal_iid_fleet_distribution_total_variation",
            "fleet_distribution_tv_excess_vs_iid",
            "exact_support_coverage",
            "unexpected_sample_mass",
            "restart_count",
            "backtrack_count",
        )
        all_metrics = [
            metric
            for case in case_dicts
            for metric in case["replicates"]  # type: ignore[index]
        ]
        aggregate = {
            key: float(np.mean([metric[key] for metric in all_metrics]))
            for key in aggregate_keys
        }
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "purpose": "calibrate-constrained-backtracking-against-exact-micro-belief",
            "sampler_id": "constrained-backtracking-v1",
            "posterior_exact": False,
            "blind_test_used": False,
            "sample_count": self.sample_count,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "aggregate_mean_metrics": aggregate,
            "cases": case_dicts,
        }


def rectangular_micro_topology(rows: int = 3, columns: int = 3) -> Topology:
    """Build a compact full-grid topology without importing an environment."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    cells = {
        row * columns + column: Cell(
            action=row * columns + column,
            row=row,
            column=column,
        )
        for row in range(rows)
        for column in range(columns)
    }
    neighbors: dict[int, tuple[int, ...]] = {}
    for action, cell in cells.items():
        adjacent = tuple(
            candidate_row * columns + candidate_column
            for candidate_row, candidate_column in (
                (cell.row - 1, cell.column),
                (cell.row, cell.column - 1),
                (cell.row, cell.column + 1),
                (cell.row + 1, cell.column),
            )
            if 0 <= candidate_row < rows and 0 <= candidate_column < columns
        )
        neighbors[action] = tuple(sorted(adjacent))
    return Topology(
        name=f"calibration-grid-{rows}x{columns}",
        rows=rows,
        columns=columns,
        cells_by_action=cells,
        neighbors_by_action=neighbors,
    )


def default_micro_calibration_cases() -> tuple[CalibrationCase, ...]:
    """Return fixed public states spanning prior, hit, miss and sunk evidence.

    The single length-two ship makes the 3x3 fleet space identical to the
    dynamic-programming oracle's physical layouts: twelve uniform fleets.
    None of these fixed histories consume a validation or blind-test seed.
    """
    topology = rectangular_micro_topology()
    specs = (ShipSpec("micro-ship-2", 2),)
    single_ship_cases = (
        CalibrationCase("prior", PublicAttackState(topology, frozenset(), frozenset()), specs),
        CalibrationCase(
            "active-hit-center",
            PublicAttackState(topology, frozenset({4}), frozenset()),
            specs,
        ),
        CalibrationCase(
            "miss-center",
            PublicAttackState(topology, frozenset(), frozenset({4})),
            specs,
        ),
        CalibrationCase(
            "hit-corner-miss-center",
            PublicAttackState(topology, frozenset({0}), frozenset({4})),
            specs,
        ),
        CalibrationCase(
            "sunk-top-edge",
            PublicAttackState(
                topology,
                frozenset({0, 1}),
                frozenset(),
                frozenset({0, 1}),
            ),
            specs,
        ),
    )
    two_ship_specs = (ShipSpec("micro-alpha-2", 2), ShipSpec("micro-beta-2", 2))
    return single_ship_cases + (
        CalibrationCase(
            "two-ship-prior",
            PublicAttackState(topology, frozenset(), frozenset()),
            two_ship_specs,
        ),
        CalibrationCase(
            "two-ship-active-hit-center",
            PublicAttackState(topology, frozenset({4}), frozenset()),
            two_ship_specs,
        ),
    )


def calibrate_constrained_sampler(
    cases: Sequence[CalibrationCase] | None = None,
    *,
    sample_count: int = 1_024,
    repetitions: int = 32,
    seed: int = 7_201,
) -> SamplerCalibration:
    """Measure Monte Carlo occupancy and fleet-distribution error exactly.

    The exact reference is the *uniform* distribution over compatible fleets,
    matching ``exact_belief`` and the one-ship microboard oracle.  Total
    variation contains both finite-sample noise and proposal bias; it is a
    measured discrepancy, not a proof that the production posterior is exact.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    selected_cases = tuple(cases or default_micro_calibration_cases())
    if not selected_cases:
        raise ValueError("at least one calibration case is required")

    results: list[CalibrationCaseResult] = []
    for case_index, case in enumerate(selected_cases):
        exact = exact_belief(case.state, case.specs, max_fleets=100_000)
        exact_distribution = _fleet_distribution(exact.fleets)
        exact_occupancy = exact.occupancy_probabilities()
        metrics: list[CalibrationMetrics] = []
        for repetition in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence((seed, case_index, repetition)))
            sampled, diagnostics = sample_compatible_fleets(
                case.state,
                case.specs,
                sample_count=sample_count,
                rng=rng,
            )
            sampled_distribution = _fleet_distribution(sampled.fleets)
            ideal_iid_distribution = _fleet_distribution(
                tuple(
                    exact.fleets[index]
                    for index in np.random.default_rng(
                        np.random.SeedSequence((seed, case_index, repetition, 1))
                    ).integers(exact.size, size=sample_count)
                )
            )
            unexpected_mass = sum(
                probability
                for fleet_key, probability in sampled_distribution.items()
                if fleet_key not in exact_distribution
            )
            if unexpected_mass > 0.0:
                raise AssertionError("constrained sampler emitted a fleet outside exact support")
            occupancy_error = sampled.occupancy_probabilities() - exact_occupancy
            proposal_total_variation = _total_variation(
                exact_distribution, sampled_distribution
            )
            ideal_iid_total_variation = _total_variation(
                exact_distribution, ideal_iid_distribution
            )
            metrics.append(
                CalibrationMetrics(
                    occupancy_mean_absolute_error=float(np.mean(np.abs(occupancy_error))),
                    occupancy_root_mean_squared_error=float(
                        np.sqrt(np.mean(np.square(occupancy_error)))
                    ),
                    occupancy_max_absolute_error=float(np.max(np.abs(occupancy_error))),
                    fleet_distribution_total_variation=proposal_total_variation,
                    ideal_iid_fleet_distribution_total_variation=ideal_iid_total_variation,
                    fleet_distribution_tv_excess_vs_iid=(
                        proposal_total_variation - ideal_iid_total_variation
                    ),
                    exact_support_coverage=float(
                        len(set(sampled_distribution).intersection(exact_distribution))
                        / len(exact_distribution)
                    ),
                    unexpected_sample_mass=float(unexpected_mass),
                    restart_count=diagnostics.restart_count,
                    backtrack_count=diagnostics.backtrack_count,
                )
            )
        results.append(
            CalibrationCaseResult(
                name=case.name,
                exact_fleet_count=exact.size,
                exact_posterior=exact.exact,
                sample_count=sample_count,
                repetitions=repetitions,
                metrics=tuple(metrics),
            )
        )
    return SamplerCalibration(tuple(results), sample_count, repetitions, seed)


FleetKey = tuple[tuple[str, int, int, str, tuple[int, ...]], ...]


def _fleet_distribution(fleets: Sequence[Fleet]) -> dict[FleetKey, float]:
    counts: dict[FleetKey, int] = {}
    for fleet in fleets:
        key: FleetKey = tuple(
            (
                placement.ship_id,
                placement.length,
                placement.anchor,
                str(placement.orientation),
                placement.cells,
            )
            for placement in fleet.placements
        )
        counts[key] = counts.get(key, 0) + 1
    total = len(fleets)
    return {key: count / total for key, count in counts.items()}


def _total_variation(
    first: dict[FleetKey, float], second: dict[FleetKey, float]
) -> float:
    return 0.5 * sum(
        abs(first.get(key, 0.0) - second.get(key, 0.0))
        for key in first.keys() | second.keys()
    )
