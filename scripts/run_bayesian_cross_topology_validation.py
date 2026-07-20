"""Run the frozen, validation-only Bayesian comparison on three topologies.

This program never accepts a test split.  It compares the public-history
probability planner with ``hunt_target-v1`` on the same legal fleets, writes
per-run manifests, paired bootstrap statistics and static report graphics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from periodic_table_battleship_rl.evaluation import RunConfig
from periodic_table_battleship_rl.experiments import (
    BELIEF_PROBABILITY_POLICY_ID,
    HUNT_TARGET_POLICY_ID,
    run_attack_baseline,
    run_belief_planner_evaluation,
)
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.experiments.belief_validation import (
    BELIEF_VALIDATION_SCHEMA_VERSION,
    BayesianValidationProtocol,
    VALIDATION_TOPOLOGIES,
    paired_seed_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "v0.7-bayes-cross-topology-validation"
ARTIFACT_ROOT = ROOT / "artifacts" / "v0.7-bayes-cross-topology-validation"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=8_801)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--smoke", action="store_true", help="use two validation seeds and 100 resamples"
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    protocol = BayesianValidationProtocol(
        seed_start=arguments.seed_start,
        seed_count=2 if arguments.smoke else arguments.seed_count,
        sample_count=arguments.sample_count,
        bootstrap_resamples=100 if arguments.smoke else arguments.bootstrap_resamples,
    )
    git_commit, dirty = _git_state()
    run_root = RUN_ROOT / ("smoke" if arguments.smoke else "full")
    artifact_root = ARTIFACT_ROOT / ("smoke" if arguments.smoke else "full")
    artifact_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": BELIEF_VALIDATION_SCHEMA_VERSION,
        "campaign": "v0.7-bayes-cross-topology-validation",
        "split": "validation",
        "blind_test_used": False,
        "protocol": protocol.as_dict(),
        "git_commit": git_commit,
        "working_tree_dirty": dirty,
        "candidate": BELIEF_PROBABILITY_POLICY_ID,
        "reference": HUNT_TARGET_POLICY_ID,
        "sampler": {"id": "constrained-backtracking-v1", "posterior_exact": False},
        "topologies": {},
    }
    for topology_index, topology in enumerate(VALIDATION_TOPOLOGIES):
        topology_root = run_root / topology.name
        candidate_config = RunConfig(
            run_id=f"v07-validation-{topology.name}-{BELIEF_PROBABILITY_POLICY_ID}",
            experiment="attack",
            scenario=topology.name,
            environment_version=ENVIRONMENT_VERSION,
            policy_id=BELIEF_PROBABILITY_POLICY_ID,
            split="validation",
            seeds=protocol.seeds,
            episodes_per_seed=protocol.episodes_per_seed,
            parameters={
                "campaign": "v0.7-bayes-cross-topology-validation",
                "schema_version": BELIEF_VALIDATION_SCHEMA_VERSION,
                "promotion_eligible": True,
                "selection_only": True,
            },
        )
        started = perf_counter()
        candidate = run_belief_planner_evaluation(
            candidate_config,
            topology,
            topology_root / BELIEF_PROBABILITY_POLICY_ID,
            git_commit=git_commit,
            uv_lock_path=ROOT / "uv.lock",
            sample_count=protocol.sample_count,
        )
        candidate_seconds = perf_counter() - started
        reference_config = RunConfig(
            run_id=f"v07-validation-{topology.name}-{HUNT_TARGET_POLICY_ID}",
            experiment="attack",
            scenario=topology.name,
            environment_version=ENVIRONMENT_VERSION,
            policy_id=HUNT_TARGET_POLICY_ID,
            split="validation",
            seeds=protocol.seeds,
            episodes_per_seed=protocol.episodes_per_seed,
            parameters={
                "campaign": "v0.7-bayes-cross-topology-validation",
                "schema_version": BELIEF_VALIDATION_SCHEMA_VERSION,
                "promotion_eligible": False,
                "selection_only": True,
            },
        )
        started = perf_counter()
        reference = run_attack_baseline(
            reference_config,
            topology,
            topology_root / HUNT_TARGET_POLICY_ID,
            git_commit=git_commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        reference_seconds = perf_counter() - started
        comparison_rng = np.random.default_rng(
            np.random.SeedSequence((protocol.bootstrap_seed, topology_index))
        )
        shots = paired_seed_comparison(
            candidate.results,
            reference.results,
            metric="valid_shots",
            direction="lower",
            bootstrap_resamples=protocol.bootstrap_resamples,
            rng=comparison_rng,
        )
        auc = paired_seed_comparison(
            candidate.results,
            reference.results,
            metric="auc_discovery",
            direction="higher",
            bootstrap_resamples=protocol.bootstrap_resamples,
            rng=np.random.default_rng(
                np.random.SeedSequence((protocol.bootstrap_seed, topology_index, 1))
            ),
        )
        report["topologies"][topology.name] = {
            "valid_cells": topology.valid_cell_count,
            "candidate_summary": candidate.summary,
            "reference_summary": reference.summary,
            "paired_valid_shots": shots,
            "paired_auc_discovery": auc,
            "elapsed_seconds": {
                "candidate": candidate_seconds,
                "reference": reference_seconds,
            },
        }
    _write_report(report, artifact_root)
    _plot(report, artifact_root / "paired-valid-shots.png")


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
    (destination / "bayes-cross-topology-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    rows: list[str] = []
    for name, value in topologies.items():
        assert isinstance(value, dict)
        candidate = value["candidate_summary"]
        reference = value["reference_summary"]
        paired = value["paired_valid_shots"]
        assert isinstance(candidate, dict) and isinstance(reference, dict) and isinstance(paired, dict)
        candidate_mean = candidate["aggregate"]["valid_shots"]["mean"]
        reference_mean = reference["aggregate"]["valid_shots"]["mean"]
        interval = paired["bootstrap_95"]
        rows.append(
            f"| `{name}` | {candidate_mean:.2f} | {reference_mean:.2f} | "
            f"{paired['candidate_minus_reference_mean']:+.2f} "
            f"[{interval['lower']:+.2f}; {interval['upper']:+.2f}] |"
        )
    lines = [
        "# Validação Bayesiana v0.7 entre topologias",
        "",
        "Esta campanha usa somente seeds de validação pré-registradas. Ela não",
        "abre, cria ou consome inventário de teste cego.",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Seeds: `{report['protocol']['seeds']}`",
        f"- Amostras Monte Carlo por decisão: `{report['protocol']['sample_count']}`",
        "- Estatística: bootstrap percentil pareado por seed, 95% bilateral.",
        "- O amostrador gera frotas compatíveis, mas não declara posterior exato.",
        "",
        "Menos tiros é melhor. Intervalo inteiramente abaixo de zero favorece a",
        "política Bayesiana sobre `hunt_target-v1` nesta validação.",
        "",
        "| Topologia | Bayes | Hunt-target | Bayes − hunt-target, IC 95% |",
        "| --- | ---: | ---: | ---: |",
        *rows,
        "",
        "![Diferenças pareadas de tiros](paired-valid-shots.png)",
    ]
    (destination / "bayes-cross-topology-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot(report: dict[str, object], destination: Path) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    labels = list(topologies)
    means: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for value in topologies.values():
        assert isinstance(value, dict)
        comparison = value["paired_valid_shots"]
        assert isinstance(comparison, dict)
        interval = comparison["bootstrap_95"]
        assert isinstance(interval, dict)
        mean = float(comparison["candidate_minus_reference_mean"])
        means.append(mean)
        lower_errors.append(mean - float(interval["lower"]))
        upper_errors.append(float(interval["upper"]) - mean)
    figure, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
    axis.errorbar(
        labels,
        means,
        yerr=np.array([lower_errors, upper_errors]),
        fmt="o",
        capsize=5,
        color="#2563eb",
    )
    axis.axhline(0.0, color="#111827", linewidth=1, linestyle="--")
    axis.set_ylabel("Bayes − hunt-target: tiros válidos")
    axis.set_title("Validação v0.7: diferença pareada por topologia")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
