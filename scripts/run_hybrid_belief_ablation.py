"""Run a validation-only CNN versus public-belief-CNN ablation for #64.

The blind test inventory is intentionally absent.  This runner compares the
same MaskablePPO/CNN budget; the candidate alone receives two maps computed
from public history by the bounded Monte Carlo belief sampler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, pstdev
import subprocess

import matplotlib.pyplot as plt

from periodic_table_battleship_rl.belief import BeliefFeatureConfig
from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training import AttackValidationConfig
from periodic_table_battleship_rl.training.cnn import (
    CnnAttackTrainingConfig,
    train_cnn_attack_policy,
)
from periodic_table_battleship_rl.training.hybrid_belief import (
    HybridBeliefAttackTrainingConfig,
    train_hybrid_belief_attack_policy,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SEEDS = (8801, 8802, 8803)
TRAINING_SEEDS = (8811, 8812, 8813)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--training-seed-count", type=int, default=len(TRAINING_SEEDS))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _mean_shots(artifact: object) -> float:
    checkpoints = getattr(artifact, "checkpoints")
    return fmean(result.valid_shots for result in checkpoints[-1].validation_results)


def main() -> None:
    arguments = _arguments()
    if (
        arguments.timesteps <= 0
        or arguments.sample_count <= 0
        or not 1 <= arguments.training_seed_count <= len(TRAINING_SEEDS)
    ):
        raise ValueError("timesteps, sample-count and training-seed-count must be positive")
    timesteps = 256 if arguments.smoke else arguments.timesteps
    training_seeds = TRAINING_SEEDS[: 1 if arguments.smoke else arguments.training_seed_count]
    n_steps = min(256, timesteps)
    schedule = AttackValidationConfig(
        seeds=VALIDATION_SEEDS,
        checkpoint_steps=(timesteps,),
    )
    output = ROOT / ".local-runs" / "v0.6-hybrid-belief"
    control_scores: list[float] = []
    candidate_scores: list[float] = []
    for seed in training_seeds:
        control = train_cnn_attack_policy(
            BATTLESHIP,
            CnnAttackTrainingConfig(
                run_id=f"v06-hybrid-control-s{seed}",
                seed=seed,
                total_timesteps=timesteps,
                checkpoint_directory=output,
                n_steps=n_steps,
                batch_size=min(64, n_steps),
                device="cpu",
            ),
            validation=schedule,
        )
        candidate = train_hybrid_belief_attack_policy(
            BATTLESHIP,
            HybridBeliefAttackTrainingConfig(
                run_id=f"v06-hybrid-belief-s{seed}",
                seed=seed,
                total_timesteps=timesteps,
                checkpoint_directory=output,
                n_steps=n_steps,
                batch_size=min(64, n_steps),
                device="cpu",
                belief_config=BeliefFeatureConfig(sample_count=arguments.sample_count),
            ),
            validation=schedule,
        )
        control_scores.append(_mean_shots(control))
        candidate_scores.append(_mean_shots(candidate))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    )
    report = {
        "schema_version": "hybrid-belief-ablation-v1",
        "issue": 64,
        "split": "validation",
        "blind_test_used": False,
        "scenario": BATTLESHIP.name,
        "training_seeds": training_seeds,
        "validation_seeds": VALIDATION_SEEDS,
        "timesteps": timesteps,
        "smoke": arguments.smoke,
        "control": {
            "policy": "maskable-ppo-cnn-v1",
            "mean_valid_shots": fmean(control_scores),
            "training_seed_std": pstdev(control_scores),
            "mean_valid_shots_by_training_seed": control_scores,
        },
        "candidate": {
            "policy": "maskable-ppo-cnn-public-belief-v1",
            "mean_valid_shots": fmean(candidate_scores),
            "training_seed_std": pstdev(candidate_scores),
            "mean_valid_shots_by_training_seed": candidate_scores,
            "belief_features": BeliefFeatureConfig(
                sample_count=arguments.sample_count
            ).public_dict(),
        },
        "candidate_minus_control": fmean(candidate_scores) - fmean(control_scores),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "promotion_eligible": False,
        "conclusion": (
            "Smoke runs validate the pipeline only; no promotion or blind test is allowed."
            if arguments.smoke
            else "Validation-only ablation; multi-seed gate remains required for promotion."
        ),
    }
    artifacts = ROOT / "artifacts" / "v0.6-hybrid-belief-pilot"
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "hybrid-belief-ablation-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_ablation(report, artifacts / "hybrid-belief-ablation.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _plot_ablation(report: dict[str, object], destination: Path) -> None:
    control = report["control"]
    candidate = report["candidate"]
    assert isinstance(control, dict) and isinstance(candidate, dict)
    means = [float(control["mean_valid_shots"]), float(candidate["mean_valid_shots"])]
    deviations = [float(control["training_seed_std"]), float(candidate["training_seed_std"])]
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    axis.bar(
        ("CNN\ncontrole", "CNN +\ncrença pública"),
        means,
        yerr=deviations,
        capsize=4,
        color=("#64748b", "#2563eb"),
    )
    axis.set_ylabel("Tiros válidos na validação (menor é melhor)")
    axis.set_title("Ablação da crença pública (sem teste cego)")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
