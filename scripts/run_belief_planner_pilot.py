"""Run validation-only Bayes-planner pilots with public, visual evidence.

The default seeds belong to validation and demonstration only.  This script
does not open the fixed blind test inventory and labels its Monte Carlo
proposal as approximate in every generated report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from periodic_table_battleship_rl.belief import BeliefPlanner, PublicAttackState
from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.evaluation import RunConfig
from periodic_table_battleship_rl.experiments import (
    BELIEF_HORIZON_POLICY_ID,
    BELIEF_INFORMATION_POLICY_ID,
    BELIEF_PROBABILITY_POLICY_ID,
    HUNT_TARGET_POLICY_ID,
    run_attack_baseline,
    run_belief_planner_evaluation,
)
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.topology import BATTLESHIP


ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = ROOT / "runs" / "v0.6-bayes-planner-validation"
ARTIFACT_DIRECTORY = ROOT / "artifacts" / "v0.6-bayes-planner-validation"
POLICIES = (
    BELIEF_PROBABILITY_POLICY_ID,
    BELIEF_INFORMATION_POLICY_ID,
    BELIEF_HORIZON_POLICY_ID,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--demo-seed", type=int, default=8701)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.sample_count <= 0 or args.seed_count <= 0 or args.demo_seed < 0:
        raise ValueError("sample-count and seed-count must be positive; demo-seed non-negative")
    validation_seeds = tuple(range(8601, 8601 + args.seed_count))
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    working_tree_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    )
    git_commit = f"{base_commit}-dirty" if working_tree_dirty else base_commit
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results = {}
    for policy_id in POLICIES:
        run_id = f"v06-{policy_id}"
        evaluation = run_belief_planner_evaluation(
            RunConfig(
                run_id=run_id,
                experiment="attack",
                scenario=BATTLESHIP.name,
                environment_version=ENVIRONMENT_VERSION,
                policy_id=policy_id,
                split="validation",
                seeds=validation_seeds,
                episodes_per_seed=1,
                parameters={
                    "campaign": "v0.6-bayes-planner-pilot",
                    "promotion_eligible": False,
                    "sampler": "constrained-backtracking-v1",
                    "posterior_exact": False,
                },
            ),
            BATTLESHIP,
            RUN_DIRECTORY / policy_id,
            git_commit=git_commit,
            uv_lock_path=ROOT / "uv.lock",
            sample_count=args.sample_count,
        )
        results[policy_id] = evaluation.summary
    baseline = run_attack_baseline(
        RunConfig(
            run_id="v06-hunt-target-validation",
            experiment="attack",
            scenario=BATTLESHIP.name,
            environment_version=ENVIRONMENT_VERSION,
            policy_id=HUNT_TARGET_POLICY_ID,
            split="validation",
            seeds=validation_seeds,
            episodes_per_seed=1,
            parameters={"campaign": "v0.6-bayes-planner-pilot", "promotion_eligible": False},
        ),
        BATTLESHIP,
        RUN_DIRECTORY / HUNT_TARGET_POLICY_ID,
        git_commit=git_commit,
        uv_lock_path=ROOT / "uv.lock",
    )
    results[HUNT_TARGET_POLICY_ID] = baseline.summary
    _write_report(
        results,
        validation_seeds,
        args.demo_seed,
        args.sample_count,
        git_commit,
        working_tree_dirty,
    )
    _plot_comparison(results)
    _plot_demonstration(args.demo_seed, args.sample_count)


def _write_report(
    results: dict[str, dict],
    validation_seeds: tuple[int, ...],
    demo_seed: int,
    sample_count: int,
    git_commit: str,
    working_tree_dirty: bool,
) -> None:
    summary = {
        "campaign": "v0.6-bayes-planner-pilot",
        "split": "validation",
        "validation_seeds": validation_seeds,
        "demonstration_seed": demo_seed,
        "sample_count": sample_count,
        "git_commit": git_commit,
        "working_tree_dirty": working_tree_dirty,
        "sampler": "constrained-backtracking-v1",
        "posterior_exact": False,
        "results": results,
        "blind_test_used": False,
    }
    (ARTIFACT_DIRECTORY / "belief-planner-report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for policy_id, result in results.items():
        metrics = result["aggregate"]["valid_shots"]
        rows.append((policy_id, metrics["mean"], metrics["std"]))
    lines = [
        "# Piloto Bayesiano v0.6: validação",
        "",
        "Este é um piloto de validação, sem acesso ao teste cego. As frotas do",
        "planejador Monte Carlo são compatíveis com o histórico público, mas a",
        "distribuição proposta por backtracking não é declarada como posterior exato.",
        "",
        f"- Seeds de validação: `{list(validation_seeds)}`",
        f"- Seed de demonstração: `{demo_seed}`",
        f"- Amostras por decisão: `{sample_count}`",
        f"- Revisão de origem: `{git_commit}`",
        "",
        "| Política | Tiros válidos médios | Desvio entre seeds |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| `{name}` | {mean:.2f} | {std:.2f} |" for name, mean, std in rows)
    lines.extend(
        [
            "",
            "![Comparação de políticas](belief-policy-comparison.png)",
            "",
            "![Distribuição de tiros de demonstração](belief-demo-heatmaps.png)",
            "",
            "A figura de demonstração usa seed separada e é ilustrativa, não",
            "evidência para promoção.",
        ]
    )
    (ARTIFACT_DIRECTORY / "belief-planner-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot_comparison(results: dict[str, dict]) -> None:
    names = list(results)
    means = [results[name]["aggregate"]["valid_shots"]["mean"] for name in names]
    stds = [results[name]["aggregate"]["valid_shots"]["std"] for name in names]
    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.bar(range(len(names)), means, yerr=stds, capsize=4, color="#3b82f6")
    axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names])
    axis.set_ylabel("Tiros válidos (menor é melhor)")
    axis.set_title("Piloto Bayesiano v0.6 — validação")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(ARTIFACT_DIRECTORY / "belief-policy-comparison.png", dpi=160)
    plt.close(figure)


def _plot_demonstration(demo_seed: int, sample_count: int) -> None:
    strategy_by_policy = {
        BELIEF_PROBABILITY_POLICY_ID: "probability",
        BELIEF_INFORMATION_POLICY_ID: "information",
        BELIEF_HORIZON_POLICY_ID: "horizon-2",
    }
    figure, axes = plt.subplots(1, len(strategy_by_policy), figsize=(13, 4), layout="constrained")
    for axis, (policy_id, strategy) in zip(axes, strategy_by_policy.items(), strict=True):
        env = AttackEnv(BATTLESHIP)
        observation, _ = env.reset(seed=demo_seed)
        planner = BeliefPlanner(strategy, sample_count=sample_count)
        rng = np.random.default_rng(np.random.SeedSequence((demo_seed, 0)))
        heatmap = np.zeros(BATTLESHIP.action_count, dtype=np.int64)
        terminated = truncated = False
        while not (terminated or truncated):
            state = PublicAttackState.from_observation(BATTLESHIP, observation)
            action, _ = planner.select_action(state, env.action_masks(), rng)
            heatmap[action] += 1
            observation, _, terminated, truncated, _ = env.step(action)
        image = heatmap.reshape(BATTLESHIP.rows, BATTLESHIP.columns)
        axis.imshow(image, cmap="viridis")
        axis.set_title(policy_id.replace("_mc-v1", "").replace("_", "\n"))
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Demonstração separada (seed {demo_seed})")
    figure.savefig(ARTIFACT_DIRECTORY / "belief-demo-heatmaps.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
