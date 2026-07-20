"""Train Q-learning and SARSA on the oracle microboard, then score exactly.

Example:

    uv run --extra visual python scripts/run_micro_rl_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from periodic_table_battleship_rl.algorithms import TabularTrainingConfig
from periodic_table_battleship_rl.experiments.micro_rl import run_micro_rl_comparison


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "v0.6-micro-rl"


def main() -> None:
    """Run paired training seeds and write machine- and human-readable evidence."""

    args = _arguments()
    if args.episodes <= 0 or args.seed_count <= 0:
        raise ValueError("episodes and seed-count must be positive")
    config = TabularTrainingConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.first_seed, args.first_seed + args.seed_count))
    comparison = run_micro_rl_comparison(config, seeds=seeds)
    provenance = _provenance()
    _write_json(comparison, seeds, provenance, output / "micro-rl-report.json")
    _write_csv(comparison, output / "micro-rl-comparison.csv")
    _plot(comparison, output / "micro-rl-comparison.png")
    print(f"wrote exact-oracle micro-RL comparison to {output}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=7101)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _provenance() -> dict[str, str | bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    )
    return {"git_commit": commit, "working_tree_dirty": dirty}


def _write_json(
    comparison: object,
    seeds: tuple[int, ...],
    provenance: dict[str, str | bool],
    destination: Path,
) -> None:
    from periodic_table_battleship_rl.experiments.micro_rl import MicroRLComparison

    if not isinstance(comparison, MicroRLComparison):
        raise TypeError("comparison must be a MicroRLComparison")
    payload = {
        "protocol": "micro-rl-exact-oracle-v1",
        "board": {"rows": 3, "columns": 3, "ship_lengths": [2]},
        "training": {
            "seeds": list(seeds),
            "episodes_per_seed": comparison.config.episodes,
            "alpha": comparison.config.alpha,
            "gamma": comparison.config.gamma,
            "epsilon_start": comparison.config.epsilon_start,
            "epsilon_end": comparison.config.epsilon_end,
        },
        "evaluation": {
            "kind": "exact-public-belief-enumeration",
            "hidden_fleets": 12,
            "oracle_expected_shots": comparison.oracle_expected_shots,
        },
        "rows": comparison.rows(),
        **provenance,
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(comparison: object, destination: Path) -> None:
    from periodic_table_battleship_rl.experiments.micro_rl import MicroRLComparison

    if not isinstance(comparison, MicroRLComparison):
        raise TypeError("comparison must be a MicroRLComparison")
    rows = comparison.rows()
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(comparison: object, destination: Path) -> None:
    from periodic_table_battleship_rl.experiments.micro_rl import MicroRLComparison

    if not isinstance(comparison, MicroRLComparison):
        raise TypeError("comparison must be a MicroRLComparison")
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    algorithm_data = {
        algorithm: [trial.expected_shots for trial in comparison.trials if trial.algorithm == algorithm]
        for algorithm in ("q_learning", "sarsa")
    }
    positions = np.arange(len(algorithm_data))
    means = [float(np.mean(values)) for values in algorithm_data.values()]
    stds = [float(np.std(values, ddof=1)) for values in algorithm_data.values()]
    bars = axis.bar(positions, means, yerr=stds, capsize=5, color=["#0072B2", "#E69F00"])
    axis.axhline(
        comparison.oracle_expected_shots,
        color="#009E73",
        linestyle="--",
        linewidth=2,
        label="oráculo DP (exato)",
    )
    axis.axhline(20 / 3, color="#777777", linestyle=":", label="aleatório mascarado (exato)")
    axis.set_xticks(positions, ["Q-learning", "SARSA"])
    axis.set_ylabel("tiros esperados (menor é melhor)")
    axis.set_title("Políticas tabulares avaliadas contra o oráculo exato")
    axis.set_ylim(0, max(means) + max(stds) + 1.0)
    axis.legend()
    for bar, mean in zip(bars, means, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, mean + 0.08, f"{mean:.3f}", ha="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
