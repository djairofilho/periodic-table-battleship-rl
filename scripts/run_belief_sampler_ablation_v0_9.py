"""Run sampler ablations for the v0.9 belief-probability pipeline."""

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
    calibrate_constrained_sampler,
    extended_micro_calibration_cases,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "v0.9-bayes-sampler-ablation"
VAR_SAMPLERS = (
    "constrained-backtracking-v1",
    "constrained-backtracking-short-v1",
    "importance-v1",
    "mcmc-v1",
)

CSV_FIELDNAMES = (
    "sampler_id",
    "status",
    "runtime_seconds",
    "occupancy_mae",
    "occupancy_rmse",
    "fleet_tv",
    "tv_excess_vs_iid",
    "support_coverage",
    "unexpected_sample_mass",
    "backtrack_count",
    "failure",
    "runtime_penalty_rank",
    "tv_excess_rank",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7_300)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--importance-resamples", type=int, default=4)
    parser.add_argument("--mcmc-steps", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help="small quick run")
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=list(VAR_SAMPLERS),
        choices=VAR_SAMPLERS,
        metavar="SAMPLER",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    sample_count = 64 if arguments.smoke else arguments.sample_count
    repetitions = 3 if arguments.smoke else arguments.repetitions
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    cases = extended_micro_calibration_cases()
    payload = {
        "schema_version": "v0.9-bayes-sampler-ablation-v1",
        "campaign": "v0.9-bayes-sampler-ablation",
        "seed": arguments.seed,
        "sample_count": sample_count,
        "repetitions": repetitions,
        "importance_resamples": arguments.importance_resamples,
        "mcmc_steps": arguments.mcmc_steps,
        "samplers": arguments.samplers,
        "case_count": len(cases),
        "smoke": arguments.smoke,
    }

    results: list[dict] = []
    baseline: dict | None = None
    for sampler_id in arguments.samplers:
        started = perf_counter()
        try:
            result = calibrate_constrained_sampler(
                cases,
                sample_count=sample_count,
                repetitions=repetitions,
                seed=arguments.seed,
                sampler_id=sampler_id,
                importance_resamples=arguments.importance_resamples,
                mcmc_steps=arguments.mcmc_steps,
            )
            runtime = perf_counter() - started
            metrics = result.to_dict()["aggregate_mean_metrics"]
            entry = {
                "sampler_id": sampler_id,
                "status": "ok",
                "runtime_seconds": runtime,
                "occupancy_mae": metrics["occupancy_mean_absolute_error"],
                "occupancy_rmse": metrics["occupancy_root_mean_squared_error"],
                "fleet_tv": metrics["fleet_distribution_total_variation"],
                "tv_excess_vs_iid": metrics["fleet_distribution_tv_excess_vs_iid"],
                "support_coverage": metrics["exact_support_coverage"],
                "unexpected_sample_mass": metrics["unexpected_sample_mass"],
                "backtrack_count": metrics["backtrack_count"],
            }
        except Exception as error:
            entry = {
                "sampler_id": sampler_id,
                "status": "failed",
                "runtime_seconds": perf_counter() - started,
                "failure": str(error),
            }

        results.append(entry)
        if entry["status"] == "ok" and baseline is None:
            baseline = entry

    payload["samplers"] = results

    if baseline is not None:
        for entry in results:
            if entry.get("status") != "ok":
                entry["runtime_penalty_rank"] = None
                entry["tv_excess_rank"] = None
                continue
            entry["runtime_penalty_rank"] = entry["runtime_seconds"] / max(
                baseline["runtime_seconds"], 1e-9
            )
            entry["tv_excess_rank"] = entry["tv_excess_vs_iid"] - baseline["tv_excess_vs_iid"]

    payload["recommendation"] = _recommend(results)
    report_file = output / "belief-sampler-ablation-v0.9.json"
    artifact = _with_git(payload)
    report_file.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(results, output / "belief-sampler-ablation-v0.9.csv")
    _write_summary(artifact, output / "belief-sampler-ablation-v0.9.md")
    _plot(results, output / "belief-sampler-ablation-v0.9.png")


def _with_git(payload: dict) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    )
    return payload | {"git_commit": f"{commit}-dirty" if dirty else commit, "working_tree_dirty": dirty}


def _recommend(entries: list[dict]) -> dict[str, object]:
    ok = [entry for entry in entries if entry.get("status") == "ok"]
    if not ok:
        reasons = "; ".join(entry.get("failure", "failed") for entry in entries)
        return {"status": "failed", "reason": reasons}
    ordered = sorted(ok, key=lambda item: (item["tv_excess_vs_iid"], item["runtime_seconds"]))
    best = ordered[0]
    return {
        "status": "ok",
        "best_sampler": best["sampler_id"],
        "best_tv_excess": best["tv_excess_vs_iid"],
        "best_runtime_seconds": best["runtime_seconds"],
    }


def _write_csv(results: list[dict], destination: Path) -> None:
    if not results:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in CSV_FIELDNAMES})


def _write_summary(payload: dict, destination: Path) -> None:
    rows: list[str] = []
    for sampler in payload["samplers"]:
        if sampler.get("status") != "ok":
            rows.append(
                f"| `{sampler['sampler_id']}` | falhou | {sampler['failure']} |"
            )
            continue
        rows.append(
            f"| `{sampler['sampler_id']}` | {sampler['occupancy_mae']:.4f} | "
            f"{sampler['fleet_tv']:.4f} | {sampler['tv_excess_vs_iid']:+.4f} | "
            f"{sampler['support_coverage']:.4f} | {sampler['runtime_seconds']:.2f} |"
        )
    recommendation = payload["recommendation"]
    lines = [
        "# Ablação de amostrador v0.9",
        "",
        f"- Campanha: `{payload['campaign']}`",
        f"- Repetições: `{payload['repetitions']}`",
        f"- Tamanho da amostra: `{payload['sample_count']}`",
        f"- Commit: `{payload['git_commit']}`",
        "",
    ]
    if recommendation["status"] == "ok":
        lines.append(f"- Recomendação: **{recommendation['best_sampler']}**")
        lines.extend(
            [
                "| Sampler | MAE | TV | Excesso TV vs IID | Cobertura | Tempo (s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *rows,
            ]
        )
    else:
        lines.append(f"- Recomendação: **não disponível** ({recommendation['reason']})")
        lines.append("| Sampler | Status | Falha |")
        lines.append("| --- | --- | --- |")
        lines.extend(rows)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(entries: list[dict], destination: Path) -> None:
    ok = [entry for entry in entries if entry.get("status") == "ok"]
    if not ok:
        destination.write_text("", encoding="utf-8")
        return
    names = [entry["sampler_id"] for entry in ok]
    mae = [entry["occupancy_mae"] for entry in ok]
    tv = [entry["tv_excess_vs_iid"] for entry in ok]
    runtime = [entry["runtime_seconds"] for entry in ok]
    figure, axis = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    axis[0].bar(range(len(names)), mae, color="#2563eb")
    axis[0].set_xticks(range(len(names)), names, rotation=15)
    axis[0].set_title("MAE por amostrador")
    axis[0].set_ylabel("MAE da ocupação")
    axis[0].grid(axis="y", alpha=0.25)

    axis[1].scatter(runtime, tv, c=np.arange(len(names)), s=90)
    axis[1].set_xlabel("Tempo total (s)")
    axis[1].set_ylabel("Excesso TV vs IID")
    axis[1].set_title("Custo x erro")
    axis[1].axhline(0.0, color="#111827", linewidth=1, linestyle="--", alpha=0.6)
    axis[1].grid(alpha=0.25)
    for x, y, label in zip(runtime, tv, names, strict=True):
        axis[1].text(x, y, label, fontsize=8, ha="left", va="bottom")

    figure.savefig(destination, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()

