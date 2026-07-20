"""Deterministic public visual artifacts for placement-policy evaluations.

The functions in this module consume persisted :class:`PlacementResult`
records and immutable topology information only.  They reconstruct a fleet
from the policy's recorded actions, never from an environment's private state,
which makes the CSV, charts, heatmap, and GIF reproducible after evaluation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

import numpy as np

from periodic_table_battleship_rl.evaluation import PlacementResult
from periodic_table_battleship_rl.game import CANONICAL_FLEET, Orientation
from periodic_table_battleship_rl.topology import Topology


PLACEMENT_RESULT_COLUMNS = (
    "episode_id",
    "run_id",
    "policy_id",
    "seed",
    "scenario",
    "attacker_id",
    "attacker_seed",
    "placement_actions",
    "valid_cells",
    "valid_shots_to_sink",
    "hit_segments",
    "truncated",
    "auc_discovery",
    "first_hit_shot",
    "first_sunk_shot",
    "all_sunk_shot",
)

_CELL_SIZE = 24
_MARGIN = 36
_FRAME_SECONDS = 0.55
_GAP_COLOR = "#ffffff"
_GRID_COLOR = "#ffffff"
_EMPTY_COLOR = "#dce6f2"
_FLEET_COLORS = ("#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00")


def placement_result_rows(
    results: Sequence[PlacementResult],
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Return deterministically ordered, public rows for placement artifacts."""

    labels = {} if policy_by_run is None else dict(policy_by_run)
    rows = [
        {
            "episode_id": result.episode_id,
            "run_id": result.run_id,
            "policy_id": labels.get(result.run_id, result.run_id),
            "seed": result.seed,
            "scenario": result.scenario,
            "attacker_id": result.attacker_id,
            "attacker_seed": result.attacker_seed,
            "placement_actions": ";".join(str(action) for action in result.placement_actions),
            "valid_cells": result.valid_cells,
            "valid_shots_to_sink": result.valid_shots_to_sink,
            "hit_segments": result.hit_segments,
            "truncated": result.truncated,
            "auc_discovery": result.auc_discovery,
            "first_hit_shot": result.first_hit_shot,
            "first_sunk_shot": result.first_sunk_shot,
            "all_sunk_shot": result.all_sunk_shot,
        }
        for result in results
    ]
    return sorted(
        rows,
        key=lambda row: (
            str(row["scenario"]),
            str(row["attacker_id"]),
            str(row["policy_id"]),
            str(row["episode_id"]),
        ),
    )


def write_placement_results_csv(
    results: Sequence[PlacementResult],
    path: str | Path,
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> Path:
    """Write one UTF-8, stable record per evaluated placement episode."""

    destination = _prepare_destination(path)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PLACEMENT_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(placement_result_rows(results, policy_by_run=policy_by_run))
    return destination


def placement_summary_markdown(
    results: Sequence[PlacementResult],
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> str:
    """Build a Markdown comparison grouped by scenario, attacker, and policy."""

    rows = placement_result_rows(results, policy_by_run=policy_by_run)
    if not rows:
        raise ValueError("results must contain at least one placement episode")

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = str(row["scenario"]), str(row["attacker_id"]), str(row["policy_id"])
        grouped.setdefault(key, []).append(row)

    lines = [
        "| Scenario | Attacker or mixture | Placement policy | Episodes | Mean shots to sink | Completion |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for (scenario, attacker, policy), group in sorted(grouped.items()):
        lines.append(
            "| {scenario} | {attacker} | {policy} | {episodes} | {shots:.2f} | {completion:.1%} |".format(
                scenario=scenario,
                attacker=attacker,
                policy=policy,
                episodes=len(group),
                shots=fmean(float(row["valid_shots_to_sink"]) for row in group),
                completion=fmean(float(not bool(row["truncated"])) for row in group),
            )
        )
    return "\n".join(lines) + "\n"


def write_placement_summary_markdown(
    results: Sequence[PlacementResult],
    path: str | Path,
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> Path:
    """Write a deterministic placement comparison in Markdown."""

    destination = _prepare_destination(path)
    destination.write_text(
        placement_summary_markdown(results, policy_by_run=policy_by_run), encoding="utf-8"
    )
    return destination


def plot_placement_comparison(
    results: Sequence[PlacementResult],
    path: str | Path,
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> Path:
    """Plot mean survival by scenario, attacker component, and frozen mixture.

    Higher values mean that the placement policy made the attacker's task take
    longer.  Each scenario receives its own panel, so its component and
    mixture results are never aggregated with a different board.  The chart
    deliberately uses compact display labels; the companion CSV and Markdown
    table retain the complete policy and attacker identifiers.
    """

    rows = placement_result_rows(results, policy_by_run=policy_by_run)
    if not rows:
        raise ValueError("results must contain at least one placement episode")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    destination = _prepare_destination(path)
    data = pd.DataFrame(rows).assign(
        attacker_label=lambda frame: frame["attacker_id"].map(_attacker_display_label),
        policy_label=lambda frame: frame["policy_id"].map(_placement_policy_display_label),
    )
    attackers = sorted(
        {str(row["attacker_id"]) for row in rows}, key=_attacker_sort_key
    )
    attacker_labels = [_attacker_display_label(attacker) for attacker in attackers]
    policies = sorted({str(row["policy_id"]) for row in rows})
    policy_labels = [_placement_policy_display_label(policy) for policy in policies]
    scenarios = sorted({str(row["scenario"]) for row in rows})
    with sns.axes_style("whitegrid"):
        figure, axes = plt.subplots(
            nrows=1,
            ncols=len(scenarios),
            figsize=(max(7.0, 5.3 * len(scenarios)), 5.25),
            sharey=True,
            squeeze=False,
        )
        for index, (scenario, axis) in enumerate(zip(scenarios, axes[0], strict=True)):
            scenario_data = data.loc[data["scenario"] == scenario]
            sns.barplot(
                data=scenario_data,
                x="attacker_label",
                y="valid_shots_to_sink",
                hue="policy_label",
                order=attacker_labels,
                hue_order=policy_labels,
                estimator="mean",
                errorbar=None,
                palette="colorblind",
                ax=axis,
            )
            axis.set_title(_scenario_display_label(scenario))
            axis.set_xlabel("Attacker or frozen mixture")
            axis.set_ylabel("Mean valid shots to sink fleet" if index == 0 else "")
            axis.tick_params(axis="x", rotation=0)
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()

        handles, legend_labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            title="Placement policy",
            loc="upper center",
            ncols=min(4, len(policy_labels)),
            bbox_to_anchor=(0.5, 0.99),
        )
        figure.suptitle("Placement survival by scenario and attacker", y=1.08)
        figure.text(
            0.5,
            0.01,
            "Complete attacker and policy IDs are retained in the CSV and Markdown summary.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.91))
        figure.savefig(destination, dpi=160, metadata={"Date": None})
        plt.close(figure)
    return destination


def _attacker_display_label(attacker_id: str) -> str:
    """Return a short visual label while source artifacts retain the full ID."""

    known_labels = {
        "random-masked-v1": "Random",
        "hunt-target-v1": "Hunt-target",
        "frozen-defensive-mixture-v1": "Frozen\nmixture",
    }
    if attacker_id in known_labels:
        return known_labels[attacker_id]
    if attacker_id.startswith("maskable-ppo-v1:"):
        return "Frozen PPO\nattacker"
    if attacker_id.endswith("random-hunt-frozen-ppo"):
        return "Frozen PPO\nmixture"
    return _short_identifier(attacker_id)


def _attacker_sort_key(attacker_id: str) -> tuple[int, str]:
    """Keep baseline components and the frozen mixture in comparison order."""

    order = {
        "random-masked-v1": 0,
        "hunt-target-v1": 1,
        "frozen-defensive-mixture-v1": 2,
    }
    if attacker_id.startswith("maskable-ppo-v1:"):
        return 2, attacker_id
    if attacker_id.endswith("random-hunt-frozen-ppo"):
        return 3, attacker_id
    return order.get(attacker_id, 3), attacker_id


def _placement_policy_display_label(policy_id: str) -> str:
    """Return concise legend text for known placement-policy implementations."""

    known_labels = {
        "random-legal-placement-v1": "Random legal",
        "dispersion-placement-v1": "Dispersion",
        "hunt-target-resistant-placement-v1": "Hunt-target resistant",
        "MaskablePPO placement (multi-seed)": "MaskablePPO (multi-seed)",
    }
    return known_labels.get(policy_id, _short_identifier(policy_id))


def _scenario_display_label(scenario: str) -> str:
    """Return a readable scenario title without altering persisted scenario IDs."""

    known_labels = {
        "battleship": "Classic Battleship",
        "dense-118": "Dense 118-cell board",
        "periodic-table-battleship": "Periodic Table Battleship",
    }
    return known_labels.get(scenario, _short_identifier(scenario))


def _short_identifier(identifier: str, *, maximum_length: int = 28) -> str:
    """Avoid a long unknown identifier overwhelming a static chart axis."""

    if len(identifier) <= maximum_length:
        return identifier.replace("-", "\n")
    return f"{identifier[: maximum_length - 1].replace('-', ' ')}…"


def plot_placement_segment_heatmap(
    results: Sequence[PlacementResult],
    topology: Topology,
    path: str | Path,
    *,
    attacker_id: str | None = None,
) -> Path:
    """Render how often accepted placement actions occupy each valid segment.

    ``attacker_id`` optionally narrows the view to a component or the frozen
    mixture.  Gaps in a sparse topology remain blank, so the periodic layout
    is not visually converted into a dense rectangle.
    """

    selected = [
        result
        for result in results
        if attacker_id is None or result.attacker_id == attacker_id
    ]
    if not selected:
        raise ValueError("results must contain an episode for the requested attacker_id")

    counts = np.full((topology.rows, topology.columns), np.nan, dtype=np.float64)
    for action in topology.valid_cells:
        row, column = topology.coordinate_for(action)
        counts[row, column] = 0.0
    for result in selected:
        for cells in _placement_cells(result, topology):
            for action in cells:
                row, column = topology.coordinate_for(action)
                counts[row, column] += 1.0

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    destination = _prepare_destination(path)
    maximum = float(np.nanmax(counts))
    with sns.axes_style("white"):
        figure, axis = plt.subplots(
            figsize=(max(7.2, topology.columns * 0.5), max(4.2, topology.rows * 0.5))
        )
        sns.heatmap(
            counts,
            mask=np.isnan(counts),
            cmap="mako",
            vmin=0.0,
            vmax=max(1.0, maximum),
            square=True,
            cbar_kws={"label": "Placed fleet segments"},
            xticklabels=range(1, topology.columns + 1),
            yticklabels=range(1, topology.rows + 1),
            linewidths=0.3,
            linecolor=_GRID_COLOR,
            ax=axis,
        )
        suffix = "all attackers and mixture" if attacker_id is None else attacker_id
        axis.set_title(f"Placement segment frequency: {suffix}")
        axis.set_xlabel("Topology column")
        axis.set_ylabel("Topology row")
        figure.tight_layout()
        figure.savefig(destination, dpi=160, metadata={"Date": None})
        plt.close(figure)
    return destination


def write_placement_trace_gif(
    result: PlacementResult,
    topology: Topology,
    path: str | Path,
    *,
    frame_duration_ms: int = int(_FRAME_SECONDS * 1000),
) -> Path:
    """Render the initial board plus one deterministic frame per placement action."""

    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")

    destination = _prepare_destination(path)
    frames = _placement_cells(result, topology)
    images = [_placement_frame_image(topology, frames[:index]) for index in range(6)]
    first, *remaining = images
    first.save(
        destination,
        format="GIF",
        save_all=True,
        append_images=remaining,
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return destination


def _placement_cells(
    result: PlacementResult, topology: Topology
) -> tuple[tuple[int, ...], ...]:
    """Decode and validate the recorded action for each canonical ship."""

    occupied: set[int] = set()
    placements: list[tuple[int, ...]] = []
    for action, spec in zip(result.placement_actions, CANONICAL_FLEET, strict=True):
        anchor = action % topology.action_count
        orientation = Orientation.HORIZONTAL if action < topology.action_count else Orientation.VERTICAL
        cells = topology.segment_from(anchor, orientation, spec.length)
        if cells is None or not occupied.isdisjoint(cells):
            raise ValueError(
                f"result {result.episode_id!r} contains an illegal placement action {action}"
            )
        placements.append(cells)
        occupied.update(cells)
    return tuple(placements)


def _placement_frame_image(topology: Topology, placements: Sequence[tuple[int, ...]]):
    from PIL import Image, ImageDraw

    width = _MARGIN * 2 + topology.columns * _CELL_SIZE
    height = _MARGIN * 2 + topology.rows * _CELL_SIZE
    image = Image.new("RGB", (width, height), _GAP_COLOR)
    draw = ImageDraw.Draw(image)
    colors = {
        action: _FLEET_COLORS[index]
        for index, placement in enumerate(placements)
        for action in placement
    }
    for action in topology.valid_cells:
        row, column = topology.coordinate_for(action)
        left = _MARGIN + column * _CELL_SIZE
        top = _MARGIN + row * _CELL_SIZE
        draw.rectangle(
            (left, top, left + _CELL_SIZE, top + _CELL_SIZE),
            fill=colors.get(action, _EMPTY_COLOR),
            outline=_GRID_COLOR,
        )
    return image


def _prepare_destination(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
