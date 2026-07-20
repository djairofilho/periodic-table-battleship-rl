"""Calibrate constrained-backtracking against exact 3x3 public beliefs.

Example:

    uv run --extra visual python scripts/run_belief_sampler_calibration.py

This fixed microboard diagnostic never reads the blind test inventory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from periodic_table_battleship_rl.belief.calibration import (
    SamplerCalibration,
    calibrate_constrained_sampler,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "v0.7-bayes-sampler-calibration"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=1_024)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7_201)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Write deterministic JSON, CSV and a static calibration chart."""
    arguments = _arguments()
    result = calibrate_constrained_sampler(
        sample_count=arguments.sample_count,
        repetitions=arguments.repetitions,
        seed=arguments.seed,
    )
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict() | _provenance()
    (output / "belief-sampler-calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(result, output / "belief-sampler-calibration.csv")
    _plot(result, output / "belief-sampler-calibration.png")
    print(f"wrote microboard sampler calibration to {output}")


def _provenance() -> dict[str, str | bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    )
    return {"git_commit": f"{commit}-dirty" if dirty else commit, "working_tree_dirty": dirty}


def _write_csv(result: SamplerCalibration, destination: Path) -> None:
    rows = []
    for case in result.cases:
        summary = case.to_dict()["mean_metrics"]
        rows.append(
            {
                "case": case.name,
                "exact_fleet_count": case.exact_fleet_count,
                "sample_count": case.sample_count,
                "repetitions": case.repetitions,
                **summary,
            }
        )
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(result: SamplerCalibration, destination: Path) -> None:
    names = [case.name.replace("-", "\n") for case in result.cases]
    mae = [
        case.to_dict()["mean_metrics"]["occupancy_mean_absolute_error"]
        for case in result.cases
    ]
    total_variation = [
        case.to_dict()["mean_metrics"]["fleet_distribution_total_variation"]
        for case in result.cases
    ]
    ideal_total_variation = [
        case.to_dict()["mean_metrics"]["ideal_iid_fleet_distribution_total_variation"]
        for case in result.cases
    ]
    coverage = [
        case.to_dict()["mean_metrics"]["exact_support_coverage"]
        for case in result.cases
    ]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    axes[0].bar(x - 0.26, mae, width=0.26, label="MAE de ocupação", color="#0072B2")
    axes[0].bar(
        x,
        total_variation,
        width=0.26,
        label="TV de frotas proposta",
        color="#D55E00",
    )
    axes[0].bar(
        x + 0.26,
        ideal_total_variation,
        width=0.26,
        label="TV IID exata",
        color="#999999",
    )
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Discrepância da crença exata (menor é melhor)")
    axes[0].set_title("Amostrador vs. crença exata")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(x, coverage, color="#009E73")
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Fração do suporte exato visitada")
    axes[1].set_title("Cobertura do suporte")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"Calibração Monte Carlo 3×3: {result.sample_count} amostras × "
        f"{result.repetitions} repetições"
    )
    figure.savefig(destination, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
