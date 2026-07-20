"""Distil compact CNN/GNN public students and evaluate only validation seeds.

This is an audit-sized v0.7 pilot.  Training and held-out validation seed
inventories are explicit and disjoint.  It neither accepts nor consumes a
blind-test split, and its promotion decision is intentionally always deferred
to the pre-registered larger validation protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np

from periodic_table_battleship_rl.envs.attack import AttackEnv
from periodic_table_battleship_rl.policies import hunt_target_action
from periodic_table_battleship_rl.topology import (
    BATTLESHIP,
    DENSE_118,
    PERIODIC_TABLE_BATTLESHIP,
    Topology,
)
from periodic_table_battleship_rl.training.bayesian_distillation import (
    BayesianDemonstrationConfig,
    generate_bayesian_demonstrations,
    load_bayesian_demonstrations,
)
from periodic_table_battleship_rl.training.bayesian_students import (
    BayesianStudentTrainingConfig,
    evaluate_bayesian_student,
    load_bayesian_student_policy,
    teacher_action_agreement,
    train_bayesian_student,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "v0.7-bayesian-students"
CHECKPOINT_ROOT = ROOT / ".local-runs" / "v0.7-bayesian-students"
TOPOLOGIES: tuple[Topology, ...] = (BATTLESHIP, DENSE_118, PERIODIC_TABLE_BATTLESHIP)
TRAIN_SEEDS = (9_601, 9_602)
VALIDATION_SEEDS = (9_651, 9_652)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use two Monte Carlo samples and four epochs",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.sample_count <= 0 or args.epochs <= 0 or args.hidden_dim <= 0:
        raise ValueError("sample-count, epochs and hidden-dim must be positive")
    sample_count = 2 if args.smoke else args.sample_count
    epochs = 4 if args.smoke else args.epochs
    data_root = ARTIFACT_ROOT / "datasets"
    report: dict[str, object] = {
        "schema_version": "bayesian-public-student-pilot-v1",
        "campaign": "v0.7-bayesian-student-pilot",
        "split": "validation",
        "blind_test_used": False,
        "promotion_eligible": False,
        "promotion_blocker": (
            "Pilot with two held-out seeds only; promotion requires the separately "
            "pre-registered multi-seed validation gate."
        ),
        "teacher": "belief_probability_mc-v1",
        "training_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "sample_count": sample_count,
        "epochs": epochs,
        "topologies": {},
    }
    for topology_index, topology in enumerate(TOPOLOGIES):
        train_dataset = generate_bayesian_demonstrations(
            topology,
            BayesianDemonstrationConfig(
                dataset_id=f"{topology.name}-train",
                seeds=TRAIN_SEEDS,
                output_directory=data_root,
                sample_count=sample_count,
                sampler_seed=17,
            ),
        )
        validation_dataset = generate_bayesian_demonstrations(
            topology,
            BayesianDemonstrationConfig(
                dataset_id=f"{topology.name}-validation",
                seeds=VALIDATION_SEEDS,
                output_directory=data_root,
                sample_count=sample_count,
                sampler_seed=17,
            ),
        )
        held_out = load_bayesian_demonstrations(validation_dataset.data_path)
        topology_report: dict[str, object] = {
            "valid_cells": topology.valid_cell_count,
            "datasets": {
                "train": {
                    "path": str(train_dataset.data_path.relative_to(ROOT)),
                    "sha256": train_dataset.data_sha256,
                    "decisions": train_dataset.sample_count,
                },
                "validation": {
                    "path": str(validation_dataset.data_path.relative_to(ROOT)),
                    "sha256": validation_dataset.data_sha256,
                    "decisions": validation_dataset.sample_count,
                },
            },
            "students": {},
        }
        for architecture in ("cnn", "gnn"):
            artifact = train_bayesian_student(
                topology,
                BayesianStudentTrainingConfig(
                    run_id=f"{topology.name}-{architecture}",
                    architecture=architecture,  # type: ignore[arg-type]
                    seed=20_260_720 + topology_index,
                    dataset_path=train_dataset.data_path,
                    checkpoint_directory=CHECKPOINT_ROOT,
                    epochs=epochs,
                    hidden_dim=args.hidden_dim,
                    device=args.device,
                ),
            )
            policy = load_bayesian_student_policy(
                topology, artifact.checkpoint_path, device=args.device
            )
            topology_report["students"][architecture] = {
                "training_action_agreement": artifact.training_action_agreement,
                "validation_teacher_action_agreement": teacher_action_agreement(
                    policy, held_out
                ),
                "validation_game": evaluate_bayesian_student(
                    topology, policy, seeds=VALIDATION_SEEDS
                ),
                "checkpoint": str(artifact.checkpoint_path.relative_to(ROOT)),
                "losses": list(artifact.losses),
            }
        topology_report["hunt_target_validation"] = _evaluate_hunt_target(
            topology, VALIDATION_SEEDS
        )
        report["topologies"][topology.name] = topology_report
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "student-pilot-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary(report)
    _plot(report)


def _evaluate_hunt_target(
    topology: Topology, seeds: tuple[int, ...]
) -> dict[str, object]:
    episodes: list[dict[str, object]] = []
    for seed in seeds:
        environment = AttackEnv(topology)
        observation, _ = environment.reset(seed=seed)
        rng = np.random.default_rng(seed)
        terminated = truncated = False
        while not (terminated or truncated):
            active_hits = np.flatnonzero(observation[1].reshape(-1)).tolist()
            action = hunt_target_action(
                topology, environment.action_masks(), active_hits, rng
            )
            observation, _, terminated, truncated, info = environment.step(action)
        episodes.append({"seed": seed, "valid_shots": int(info["valid_shots"])})
    return {
        "policy_id": "hunt-target-v1",
        "episodes": episodes,
        "mean_valid_shots": fmean(float(item["valid_shots"]) for item in episodes),
    }


def _write_summary(report: dict[str, object]) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    rows: list[str] = []
    for name, topology_report in topologies.items():
        assert isinstance(topology_report, dict)
        students = topology_report["students"]
        hunt = topology_report["hunt_target_validation"]
        assert isinstance(students, dict) and isinstance(hunt, dict)
        for architecture, student in students.items():
            assert isinstance(student, dict)
            evaluation = student["validation_game"]
            assert isinstance(evaluation, dict)
            rows.append(
                f"| `{name}` | `{architecture}` | "
                f"{student['validation_teacher_action_agreement']:.3f} | "
                f"{evaluation['mean_valid_shots']:.2f} | {hunt['mean_valid_shots']:.2f} |"
            )
    lines = [
        "# Piloto de estudantes neurais Bayesianos v0.7",
        "",
        "CNN e GNN recebem exclusivamente observações públicas e máscara legal. "
        "As seeds de treino e validação são distintas; nenhum teste cego foi aberto.",
        "",
        f"- Seeds de treino: `{report['training_seeds']}`",
        f"- Seeds de validação: `{report['validation_seeds']}`",
        f"- Promoção: **rejeitada neste piloto**. {report['promotion_blocker']}",
        "",
        "| Topologia | Estudante | Acordo com professor | Tiros | Hunt-target |",
        "| --- | --- | ---: | ---: | ---: |",
        *rows,
        "",
        "![Comparação de tiros](student-valid-shots.png)",
    ]
    (ARTIFACT_ROOT / "student-pilot-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot(report: dict[str, object]) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for name, topology_report in topologies.items():
        assert isinstance(topology_report, dict)
        students = topology_report["students"]
        hunt = topology_report["hunt_target_validation"]
        assert isinstance(students, dict) and isinstance(hunt, dict)
        for architecture, student in students.items():
            assert isinstance(student, dict)
            evaluation = student["validation_game"]
            assert isinstance(evaluation, dict)
            labels.append(f"{name}\n{architecture}")
            values.append(float(evaluation["mean_valid_shots"]))
            colors.append("#2563eb" if architecture == "cnn" else "#7c3aed")
        labels.append(f"{name}\nhunt")
        values.append(float(hunt["mean_valid_shots"]))
        colors.append("#64748b")
    figure, axis = plt.subplots(figsize=(11, 4.8), layout="constrained")
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("Tiros válidos médios (menor é melhor)")
    axis.set_title("Piloto v0.7: estudantes públicos vs. hunt-target")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(ARTIFACT_ROOT / "student-valid-shots.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
