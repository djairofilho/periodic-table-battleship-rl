"""Rebuild the exact microboard-oracle comparison and its visual artifact.

Example:

    uv run --extra visual python scripts/run_micro_oracle.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from periodic_table_battleship_rl.oracle import evaluate_baselines


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "v0.6-micro-oracle"


def main() -> None:
    """Write exact tables, provenance-free numerical result and one figure."""

    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    comparison = evaluate_baselines()
    _write_csv(comparison, output / "comparison.csv")
    _write_json(comparison, output / "oracle-report.json")
    _plot(comparison, output / "oracle-comparison.png")
    print(f"wrote exact micro-oracle results to {output}")


def _write_csv(comparison: object, destination: Path) -> None:
    from periodic_table_battleship_rl.oracle import OracleComparison

    if not isinstance(comparison, OracleComparison):
        raise TypeError("comparison must be an OracleComparison")
    rows = [
        {
            "policy": "dynamic-programming-oracle",
            "expected_shots": comparison.oracle.expected_shots,
            "regret_vs_oracle": 0.0,
            "status": "optimal exact",
        },
        *(
            {
                "policy": result.name,
                "expected_shots": result.expected_shots,
                "regret_vs_oracle": result.regret_vs_oracle,
                "status": "baseline exact evaluation",
            }
            for result in comparison.baselines
        ),
    ]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(comparison: object, destination: Path) -> None:
    from periodic_table_battleship_rl.oracle import OracleComparison

    if not isinstance(comparison, OracleComparison):
        raise TypeError("comparison must be an OracleComparison")
    payload = {
        "protocol": "exact-public-belief-dp-v1",
        "board": {
            "rows": comparison.config.rows,
            "columns": comparison.config.columns,
            "ship_lengths": list(comparison.config.ship_lengths),
        },
        "uniform_prior_fleet_count": comparison.fleet_count,
        "oracle": {
            "expected_shots": comparison.oracle.expected_shots,
            "optimal_first_actions": list(comparison.oracle.optimal_actions),
            "first_action_values": {
                str(action): value
                for action, value in comparison.oracle.action_values.items()
            },
            "memoized_belief_states": comparison.oracle.solved_states,
        },
        "baselines": [
            {
                "name": result.name,
                "expected_shots": result.expected_shots,
                "regret_vs_oracle": result.regret_vs_oracle,
            }
            for result in comparison.baselines
        ],
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _plot(comparison: object, destination: Path) -> None:
    from periodic_table_battleship_rl.oracle import ExactBattleshipOracle, OracleComparison

    if not isinstance(comparison, OracleComparison):
        raise TypeError("comparison must be an OracleComparison")
    oracle = ExactBattleshipOracle(comparison.config)
    probabilities = oracle.occupancy_probabilities(oracle.initial_state).reshape(
        comparison.config.rows, comparison.config.columns
    )
    labels = ["DP exact"] + [result.name for result in comparison.baselines]
    expected_shots = [comparison.oracle.expected_shots] + [
        result.expected_shots for result in comparison.baselines
    ]
    colors = ["#0072B2", "#999999", "#E69F00", "#009E73"]

    figure, (heatmap_axis, bar_axis) = plt.subplots(1, 2, figsize=(10, 4.2))
    image = heatmap_axis.imshow(probabilities, cmap="viridis", vmin=0, vmax=1)
    for row in range(comparison.config.rows):
        for column in range(comparison.config.columns):
            heatmap_axis.text(
                column,
                row,
                f"{probabilities[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if probabilities[row, column] < 0.28 else "black",
                fontsize=10,
            )
    heatmap_axis.set_title("Prior exato de ocupação")
    heatmap_axis.set_xlabel("coluna")
    heatmap_axis.set_ylabel("linha")
    heatmap_axis.set_xticks(range(comparison.config.columns))
    heatmap_axis.set_yticks(range(comparison.config.rows))
    figure.colorbar(image, ax=heatmap_axis, shrink=0.78, label="P(célula ocupada)")

    bars = bar_axis.barh(labels, expected_shots, color=colors)
    bar_axis.invert_yaxis()
    bar_axis.set_title("Custo esperado exato")
    bar_axis.set_xlabel("tiros até vencer (menor é melhor)")
    bar_axis.set_xlim(0, max(expected_shots) + 0.8)
    for bar, value in zip(bars, expected_shots, strict=True):
        bar_axis.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    figure.suptitle("Microtabuleiro 3×3, um navio de tamanho 2, prior uniforme")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
