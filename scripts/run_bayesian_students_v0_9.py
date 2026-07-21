"""Run Bayesian student distillation campaigns in v0.9 with multi-seed ablations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np

from periodic_table_battleship_rl.envs.attack import AttackEnv
from periodic_table_battleship_rl.policies import hunt_target_action
from periodic_table_battleship_rl.topology import BATTLESHIP, DENSE_118, PERIODIC_TABLE_BATTLESHIP
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
ARTIFACT_ROOT = ROOT / "artifacts" / "v0.9-bayesian-students"
CHECKPOINT_ROOT = ROOT / ".local-runs" / "v0.9-bayesian-students"
SCENARIOS = (BATTLESHIP, DENSE_118, PERIODIC_TABLE_BATTLESHIP)


@dataclass(frozen=True)
class Ablation:
    architecture: str
    hidden_dim: int
    soft_target_weight: float

    @property
    def label(self) -> str:
        return (
            f"{self.architecture}-h{self.hidden_dim}-"
            f"s{self.soft_target_weight:.2f}"
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seed-start", type=int, default=15_001)
    parser.add_argument("--train-seed-count", type=int, default=3)
    parser.add_argument("--validation-seed-start", type=int, default=16_001)
    parser.add_argument("--validation-seed-count", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden-dim", nargs="+", type=int, default=(32, 48))
    parser.add_argument("--soft-weight", nargs="+", type=float, default=(0.0, 0.35, 0.70))
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=("cnn", "gnn"),
        choices=("cnn", "gnn"),
    )
    parser.add_argument("--sampler-seed", type=int, default=19_001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    train_seed_count = 1 if args.smoke else args.train_seed_count
    validation_seed_count = 2 if args.smoke else args.validation_seed_count
    sample_count = 2 if args.smoke else args.sample_count
    epochs = 2 if args.smoke else args.epochs
    hidden_dims = (16,) if args.smoke else args.hidden_dim
    soft_weights = (0.0, 0.5) if args.smoke else tuple(args.soft_weight)

    train_seeds = tuple(
        range(args.train_seed_start, args.train_seed_start + train_seed_count)
    )
    validation_seeds = tuple(
        range(
            args.validation_seed_start,
            args.validation_seed_start + validation_seed_count,
        )
    )

    report: dict[str, object] = {
        "schema_version": "v0.9-bayesian-students-v1",
        "campaign": "v0.9-bayesian-students",
        "split": "validation",
        "smoke": args.smoke,
        "train_seeds": list(train_seeds),
        "validation_seeds": list(validation_seeds),
        "sample_count": sample_count,
        "epochs": epochs,
        "architectures": list(args.architectures),
        "hidden_dims": list(hidden_dims),
        "soft_target_weights": list(soft_weights),
        "topologies": {},
    }

    ablations = [
        Ablation(architecture=arch, hidden_dim=hidden_dim, soft_target_weight=soft_weight)
        for arch in args.architectures
        for hidden_dim in hidden_dims
        for soft_weight in soft_weights
    ]

    for topology in SCENARIOS:
        topo_report = _run_topology(
            topology,
            train_seeds,
            validation_seeds,
            sample_count,
            args,
            epochs,
            soft_weights,
            ablations,
        )
        report["topologies"][topology.name] = topo_report

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_ROOT
    (out / "bayesian-student-v0.9-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(report, out / "bayesian-student-v0.9-results.csv")
    _write_summary(report, out / "bayesian-student-v0.9-summary.md")
    _plot(report, out / "bayesian-student-v0.9-valid-shots.png")


def _run_topology(
    topology,
    train_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    sample_count: int,
    args,
    epochs: int,
    soft_weights: tuple[float, ...],
    ablations: list[Ablation],
) -> dict[str, object]:
    del soft_weights
    datasets = _get_datasets(topology, train_seeds, validation_seeds, args, sample_count)
    held_out = load_bayesian_demonstrations(datasets.validation_data)
    hunt = _evaluate_hunt_target(topology, validation_seeds)

    students: list[dict[str, object]] = []
    for train_seed_index, train_seed in enumerate(train_seeds):
        for ablation in ablations:
            artifact = train_bayesian_student(
                topology,
                BayesianStudentTrainingConfig(
                    run_id=f"{topology.name}-{ablation.label}-s{train_seed}",
                    architecture=ablation.architecture,
                    seed=train_seed,
                    dataset_path=datasets.train_data,
                    checkpoint_directory=CHECKPOINT_ROOT,
                    epochs=epochs,
                    batch_size=32 if args.smoke else 64,
                    learning_rate=1e-3,
                    hidden_dim=ablation.hidden_dim,
                    soft_target_weight=ablation.soft_target_weight,
                    device=args.device,
                ),
            )
            policy = load_bayesian_student_policy(
                topology, artifact.checkpoint_path, device=args.device
            )
            game = evaluate_bayesian_student(topology, policy, seeds=validation_seeds)
            students.append(
                {
                    "seed": train_seed,
                    "ablation": ablation.label,
                    "architecture": ablation.architecture,
                    "hidden_dim": ablation.hidden_dim,
                    "soft_target_weight": ablation.soft_target_weight,
                    "train_decisions": datasets.train_decisions,
                    "train_seed": train_seed,
                    "mean_valid_shots": float(game["mean_valid_shots"]),
                    "mean_auc_discovery": float(game["mean_auc_discovery"]),
                    "validation_teacher_action_agreement": teacher_action_agreement(
                        policy, held_out
                    ),
                    "training_action_agreement": float(artifact.training_action_agreement),
                    "checkpoint": str(artifact.checkpoint_path.relative_to(ROOT)),
                    "loss_last": artifact.losses[-1] if artifact.losses else None,
                    "losses": list(artifact.losses),
                    "seed_offset": train_seed_index,
                }
            )

    return {
        "valid_cells": topology.valid_cell_count,
        "hunt_target": hunt,
        "students": students,
    }


@dataclass(frozen=True)
class _Datasets:
    train_data: Path
    validation_data: Path
    train_decisions: int


def _get_datasets(
    topology,
    train_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    args,
    sample_count: int,
) -> _Datasets:
    train = generate_bayesian_demonstrations(
        topology,
        BayesianDemonstrationConfig(
            dataset_id=f"{topology.name}-v0.9-students-train",
            seeds=train_seeds,
            output_directory=ARTIFACT_ROOT / "datasets",
            sample_count=sample_count,
            sampler_seed=args.sampler_seed,
        ),
    )
    validation = generate_bayesian_demonstrations(
        topology,
        BayesianDemonstrationConfig(
            dataset_id=f"{topology.name}-v0.9-students-validation",
            seeds=validation_seeds,
            output_directory=ARTIFACT_ROOT / "datasets",
            sample_count=sample_count,
            sampler_seed=args.sampler_seed + 1,
        ),
    )
    return _Datasets(
        train_data=train.data_path,
        validation_data=validation.data_path,
        train_decisions=train.sample_count,
    )


def _evaluate_hunt_target(topology, seeds: tuple[int, ...]) -> dict[str, object]:
    episodes = []
    for seed in seeds:
        environment = AttackEnv(topology)
        observation, _ = environment.reset(seed=seed)
        rng = np.random.default_rng(seed)
        terminated = truncated = False
        while not (terminated or truncated):
            active_hits = np.flatnonzero(observation[1].reshape(-1)).tolist()
            action = hunt_target_action(topology, environment.action_masks(), active_hits, rng)
            observation, _, terminated, truncated, info = environment.step(action)
        episodes.append({"seed": seed, "valid_shots": int(info["valid_shots"])})
    return {
        "policy_id": "hunt-target-v1",
        "mean_valid_shots": fmean(float(item["valid_shots"]) for item in episodes),
        "episodes": episodes,
    }


def _write_csv(report: dict[str, object], destination: Path) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    rows: list[dict[str, object]] = []
    for topology_name, topology_report in topologies.items():
        assert isinstance(topology_report, dict)
        for student in topology_report["students"]:
            assert isinstance(student, dict)
            rows.append(
                {
                    "topology": topology_name,
                    "seed": student["seed"],
                    "ablation": student["ablation"],
                    "architecture": student["architecture"],
                    "hidden_dim": student["hidden_dim"],
                    "soft_target_weight": student["soft_target_weight"],
                    "mean_valid_shots": student["mean_valid_shots"],
                    "mean_auc_discovery": student["mean_auc_discovery"],
                    "validation_teacher_action_agreement": student[
                        "validation_teacher_action_agreement"
                    ],
                    "training_action_agreement": student["training_action_agreement"],
                    "checkpoint": student["checkpoint"],
                    "loss_last": student["loss_last"],
                    "seed_offset": student["seed_offset"],
                }
            )
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(report: dict[str, object], destination: Path) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    rows: list[str] = []
    for name, topology_report in topologies.items():
        assert isinstance(topology_report, dict)
        hunt = topology_report["hunt_target"]
        assert isinstance(hunt, dict)
        best: dict | None = None
        for student in topology_report["students"]:
            assert isinstance(student, dict)
            if best is None or float(student["mean_valid_shots"]) < float(best["mean_valid_shots"]):
                best = student
        assert best is not None
        rows.append(
            f"| `{name}` | {hunt['mean_valid_shots']:.2f} | "
            f"{best['ablation']} | {best['mean_valid_shots']:.2f} | "
            f"{best['mean_auc_discovery']:.4f} | {best['training_action_agreement']:.3f} |"
        )

    lines = [
        "# Treinamentos CNN/GNN v0.9 (multi-seed)",
        "",
        f"- Arquiteturas: `{report['architectures']}`",
        f"- Seeds de treino: `{report['train_seeds']}`",
        f"- Seeds de validação: `{report['validation_seeds']}`",
        "",
        "| Cenário | Hunt-target | Melhor estudante | Tiros válidos | AUC | Acordo (treino) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
        *rows,
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(report: dict[str, object], destination: Path) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for name, topology_report in topologies.items():
        assert isinstance(topology_report, dict)
        hunt = topology_report["hunt_target"]
        assert isinstance(hunt, dict)
        labels.append(f"{name}\nHunt")
        values.append(float(hunt["mean_valid_shots"]))
        colors.append("#6b7280")
        for student in topology_report["students"]:
            assert isinstance(student, dict)
            labels.append(f"{name}\n{student['ablation']}")
            values.append(float(student["mean_valid_shots"]))
            colors.append("#0ea5e9" if student["architecture"] == "cnn" else "#8b5cf6")
    figure, axis = plt.subplots(figsize=(13, 5.0), layout="constrained")
    axis.bar(range(len(labels)), values, color=colors)
    axis.set_xticks(range(len(labels)), labels, rotation=60)
    axis.set_ylabel("Média de tiros válidos (menor é melhor)")
    axis.set_title("Estudantes Bayesianos v0.9: validação multi-seed")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
