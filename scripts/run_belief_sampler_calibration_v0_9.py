"""Calibrate a sampler variant on expanded micro-history sets for v0.9.

This runner emits reproducible evidence for the Bayesian sampler variant used in
the v0.9 campaign.  It never reads hidden fleet data and never touches blind
test seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from periodic_table_battleship_rl.belief.calibration import (
    SamplerCalibration,
    calibrate_constrained_sampler,
    default_micro_calibration_cases,
    extended_micro_calibration_cases,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "v0.9-bayes-sampler-calibration"
SUPPORTED_CASE_SETS = ("default", "extended")
SUPPORTED_SAMPLERS = (
    "constrained-backtracking-v1",
    "constrained-backtracking-short-v1",
    "importance-v1",
    "mcmc-v1",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=2_048)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7_201)
    parser.add_argument("--case-set", choices=SUPPORTED_CASE_SETS, default="extended")
    parser.add_argument("--sampler-id", choices=SUPPORTED_SAMPLERS, default="constrained-backtracking-v1")
    parser.add_argument("--importance-resamples", type=int, default=4)
    parser.add_argument("--mcmc-steps", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="small quick run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    sample_count = 64 if arguments.smoke else arguments.sample_count
    repetitions = 4 if arguments.smoke else arguments.repetitions
    if arguments.sampler_id == "mcmc-v1" and repetitions > 0:
        mcmc_steps = 16 if arguments.smoke else arguments.mcmc_steps
    else:
        mcmc_steps = arguments.mcmc_steps

    cases = (
        extended_micro_calibration_cases()
        if arguments.case_set == "extended"
        else default_micro_calibration_cases()
    )
    payload = {
        "campaign": "v0.9-bayes-sampler-calibration",
        "case_set": arguments.case_set,
        "sampler_id": arguments.sampler_id,
        "smoke": arguments.smoke,
        "sample_count": sample_count,
        "repetitions": repetitions,
        "seed": arguments.seed,
        "importance_resamples": arguments.importance_resamples,
        "mcmc_steps": mcmc_steps,
    }

    started = perf_counter()
    result = calibrate_constrained_sampler(
        cases=cases,
        sample_count=sample_count,
        repetitions=repetitions,
        seed=arguments.seed,
        sampler_id=arguments.sampler_id,
        importance_resamples=arguments.importance_resamples,
        mcmc_steps=mcmc_steps,
    )
    payload["elapsed_seconds"] = perf_counter() - started
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    report_payload = result.to_dict() | _provenance() | payload
    (output / "belief-sampler-calibration-v0.9.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(result, output / "belief-sampler-calibration-v0.9.csv")
    _plot(result, output / "belief-sampler-calibration-v0.9.png")
    _write_summary(result, output / "belief-sampler-calibration-v0.9.md")


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
    rows: list[dict[str, str | float]] = []
    for case in result.cases:
        summary = case.to_dict()["mean_metrics"]
        rows.append(
            {
                "case": case.name,
                "exact_fleet_count": case.exact_fleet_count,
                "sample_count": case.sample_count,
                "repetitions": case.repetitions,
                "occupancy_mae": summary["occupancy_mean_absolute_error"],
                "occupancy_rmse": summary["occupancy_root_mean_squared_error"],
                "fleet_tv": summary["fleet_distribution_total_variation"],
                "fleet_tv_excess_iid": summary["fleet_distribution_tv_excess_vs_iid"],
                "support_coverage": summary["exact_support_coverage"],
                "unexpected_mass": summary["unexpected_sample_mass"],
                "backtracks": summary["backtrack_count"],
            }
        )
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(result: SamplerCalibration, destination: Path) -> None:
    rows: list[str] = []
    for case in result.cases:
        summary = case.to_dict()["mean_metrics"]
        rows.append(
            f"| `{case.name}` | {case.exact_fleet_count} | "
            f"{summary['occupancy_mean_absolute_error']:.4f} | "
            f"{summary['fleet_distribution_tv_excess_vs_iid']:+.4f} | "
            f"{summary['exact_support_coverage']:.3f} |"
        )
    lines = [
        "# Calibração do amostrador Bayesiano (v0.9)",
        "",
        "- Campaign: `v0.9-bayes-sampler-calibration`",
        f"- Sampler: `{result.to_dict()['sampler_id']}`",
        f"- Case set: `{result.to_dict()['cases'][0]['name'] if result.cases else 'empty'}`",
        "",
        "| Caso | Frotas reais | MAE ocupação | Excesso TV vs IID | Cobertura do suporte |",
        "| --- | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        f"- Total de amostras por caso: `{result.sample_count}` x `{result.repetitions}` repetições",
        f"- Taxa média de cobertura: `{result.to_dict()['aggregate_mean_metrics']['exact_support_coverage']:.4f}`",
        f"- Módulo de completude: `{result.to_dict()['aggregate_mean_metrics']['exact_support_coverage']:.4f}`",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(result: SamplerCalibration, destination: Path) -> None:
    names = [case.name.replace("-", "\n") for case in result.cases]
    x = np.arange(len(names))
    metrics_by_case = [case.to_dict()["mean_metrics"] for case in result.cases]
    mae = [case_metric["occupancy_mean_absolute_error"] for case_metric in metrics_by_case]
    excess_tv = [
        case_metric["fleet_distribution_tv_excess_vs_iid"] for case_metric in metrics_by_case
    ]
    coverage = [
        case_metric["exact_support_coverage"] for case_metric in metrics_by_case
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].bar(x - 0.20, mae, width=0.40, color="#3b82f6", label="MAE de ocupação")
    axes[0].bar(
        x + 0.20,
        excess_tv,
        width=0.40,
        color="#9333ea",
        label="Excesso de TV vs IID",
    )
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Métrica (menor é melhor)")
    axes[0].set_title(f"Erros de calibração por estado ({result.to_dict()['sampler_id']})")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, coverage, color="#22c55e")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("Cobertura do suporte exato")
    axes[1].set_title("Cobertura por estado")
    axes[1].grid(axis="y", alpha=0.25)

    figure.savefig(destination, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
