"""v0.9 validation on three topologies with multi-seed paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from periodic_table_battleship_rl.experiments import (
    BELIEF_PROBABILITY_POLICY_ID,
    HUNT_TARGET_POLICY_ID,
    run_attack_baseline,
    run_belief_planner_evaluation,
)
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.experiments.belief_validation import (
    BayesianValidationProtocol,
    paired_seed_comparison,
)
from periodic_table_battleship_rl.topology import (
    BATTLESHIP,
    DENSE_118,
    PERIODIC_TABLE_BATTLESHIP,
    Topology,
)
from periodic_table_battleship_rl.evaluation import RunConfig


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "v0.9-bayes-cross-topology-validation"
ARTIFACT_ROOT = ROOT / "artifacts" / "v0.9-bayes-cross-topology-validation"


TOPOLOGIES: tuple[Topology, ...] = (BATTLESHIP, DENSE_118, PERIODIC_TABLE_BATTLESHIP)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=9_201)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    seed_count = 2 if args.smoke else args.seed_count
    bootstrap_resamples = 100 if args.smoke else args.bootstrap_resamples
    protocol = BayesianValidationProtocol(
        seed_start=args.seed_start,
        seed_count=seed_count,
        episodes_per_seed=args.episodes_per_seed,
        sample_count=args.sample_count,
        bootstrap_resamples=bootstrap_resamples,
        split="validation",
    )
    artifact_root = ARTIFACT_ROOT / ("smoke" if args.smoke else "full")
    run_root = RUN_ROOT / ("smoke" if args.smoke else "full")
    artifact_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "schema_version": "bayes-cross-topology-validation-v1",
        "campaign": "v0.9-bayes-cross-topology-validation",
        "protocol": protocol.as_dict(),
        "topologies": {},
        "blind_test_used": False,
        "candidate": BELIEF_PROBABILITY_POLICY_ID,
        "reference": HUNT_TARGET_POLICY_ID,
        "split": "validation",
    }
    git_commit, dirty = _git_state()
    report["git_commit"] = git_commit
    report["working_tree_dirty"] = dirty

    for topology_index, topology in enumerate(TOPOLOGIES):
        topology_root = run_root / topology.name
        candidate = run_belief_planner_evaluation(
            RunConfig(
                run_id=f"v09-validation-{topology.name}-{BELIEF_PROBABILITY_POLICY_ID}",
                experiment="attack",
                scenario=topology.name,
                environment_version=ENVIRONMENT_VERSION,
                policy_id=BELIEF_PROBABILITY_POLICY_ID,
                split="validation",
                seeds=protocol.seeds,
                episodes_per_seed=protocol.episodes_per_seed,
                parameters={"campaign": "v0.9-bayes-cross-topology-validation"},
            ),
            topology,
            topology_root / BELIEF_PROBABILITY_POLICY_ID,
            git_commit=git_commit,
            uv_lock_path=ROOT / "uv.lock",
            sample_count=protocol.sample_count,
        )

        reference = run_attack_baseline(
            RunConfig(
                run_id=f"v09-validation-{topology.name}-{HUNT_TARGET_POLICY_ID}",
                experiment="attack",
                scenario=topology.name,
                environment_version=ENVIRONMENT_VERSION,
                policy_id=HUNT_TARGET_POLICY_ID,
                split="validation",
                seeds=protocol.seeds,
                episodes_per_seed=protocol.episodes_per_seed,
                parameters={"campaign": "v0.9-bayes-cross-topology-validation"},
            ),
            topology,
            topology_root / HUNT_TARGET_POLICY_ID,
            git_commit=git_commit,
            uv_lock_path=ROOT / "uv.lock",
        )

        rng = np.random.default_rng(
            np.random.SeedSequence((protocol.bootstrap_seed, topology_index))
        )
        paired = paired_seed_comparison(
            candidate.results,
            reference.results,
            metric="valid_shots",
            direction="lower",
            bootstrap_resamples=protocol.bootstrap_resamples,
            rng=rng,
        )
        report["topologies"][topology.name] = {
            "valid_cells": topology.valid_cell_count,
            "candidate_summary": candidate.summary,
            "reference_summary": reference.summary,
            "paired_valid_shots": paired,
        }

    _write_report(report, artifact_root / "bayes-cross-topology-v0.9.json")
    _write_csv(report, artifact_root / "bayes-cross-topology-v0.9.csv")
    _write_summary(report, artifact_root / "bayes-cross-topology-v0.9.md")
    _plot(report["topologies"], artifact_root / "paired-valid-shots-v0.9.png")


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    )
    return (f"{commit}-dirty" if dirty else commit, dirty)


def _write_report(report: dict[str, object], destination: Path) -> None:
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(report: dict[str, object], destination: Path) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    rows: list[dict[str, str | float]] = []
    for topology_name, topology_report in topologies.items():
        assert isinstance(topology_report, dict)
        candidate_summary = topology_report["candidate_summary"]
        reference_summary = topology_report["reference_summary"]
        paired = topology_report["paired_valid_shots"]
        assert isinstance(candidate_summary, dict)
        assert isinstance(reference_summary, dict)
        assert isinstance(paired, dict)
        rows.append(
            {
                "topology": topology_name,
                "candidate_mean_valid_shots": candidate_summary["aggregate"]["valid_shots"]["mean"],
                "reference_mean_valid_shots": reference_summary["aggregate"]["valid_shots"]["mean"],
                "delta_mean": paired["candidate_minus_reference_mean"],
                "ci_lower": paired["bootstrap_95"]["lower"],
                "ci_upper": paired["bootstrap_95"]["upper"],
                "improves": paired["improves_reference_at_95"],
            }
        )
    fieldnames = (
        "topology",
        "candidate_mean_valid_shots",
        "reference_mean_valid_shots",
        "delta_mean",
        "ci_lower",
        "ci_upper",
        "improves",
    )
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(report: dict[str, object], destination: Path) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    protocol = report["protocol"]
    assert isinstance(protocol, dict)
    protocol_seed_start = protocol.get("seed_start")
    rows: list[str] = []
    for name, topology_report in topologies.items():
        paired = topology_report["paired_valid_shots"]
        candidate_summary = topology_report["candidate_summary"]
        reference_summary = topology_report["reference_summary"]
        assert isinstance(paired, dict)
        assert isinstance(candidate_summary, dict)
        assert isinstance(reference_summary, dict)
        rows.append(
            f"| `{name}` | {candidate_summary['aggregate']['valid_shots']['mean']:.2f} | "
            f"{reference_summary['aggregate']['valid_shots']['mean']:.2f} | "
            f"{paired['candidate_minus_reference_mean']:+.2f} "
            f"[{paired['bootstrap_95']['lower']:+.2f}; {paired['bootstrap_95']['upper']:+.2f}] |"
        )
    lines = [
        "# Validação Bayesiana v0.9 (topologias)",
        "",
        "Essa campanha usa somente split de validação pré-registrada.",
        f"- Seed inicial: `{protocol_seed_start}`",
        f"- Seeds: `{report['protocol']['seeds']}`",
        "",
        "| Topologia | Média Bayes | Média Hunt-target | Bayes - hunt, IC 95% |",
        "| --- | ---: | ---: | ---: |",
        *rows,
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(topologies: dict[str, object], destination: Path) -> None:
    names: list[str] = []
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for name, data in topologies.items():
        assert isinstance(data, dict)
        comparison = data["paired_valid_shots"]
        assert isinstance(comparison, dict)
        mean = float(comparison["candidate_minus_reference_mean"])
        interval = comparison["bootstrap_95"]
        assert isinstance(interval, dict)
        names.append(name)
        means.append(mean)
        lower.append(mean - float(interval["lower"]))
        upper.append(float(interval["upper"]) - mean)

    if not names:
        return
    figure, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
    axis.errorbar(names, means, yerr=np.array([lower, upper]), fmt="o", capsize=5, color="#2563eb")
    axis.axhline(0.0, color="#111827", linewidth=1, linestyle="--")
    axis.set_ylabel("Bayes - Hunt (tamanho válido)")
    axis.set_title("Validação v0.9: diferença pareada por topologia")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
