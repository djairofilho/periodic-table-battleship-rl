"""Deterministic public visual artifacts for attack-policy evaluations.

The module deliberately consumes :class:`EpisodeResult` records and public
``AttackEpisodeTrace`` snapshots.  It never reads an environment or a fleet,
so a chart or GIF can be regenerated from persisted public artifacts alone.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

from periodic_table_battleship_rl.evaluation import EpisodeResult
from periodic_table_battleship_rl.rendering import AttackEpisodeTrace, AttackFrame


RESULT_COLUMNS = (
    "episode_id",
    "run_id",
    "policy_id",
    "seed",
    "scenario",
    "valid_cells",
    "valid_shots",
    "invalid_attempts",
    "hit_segments",
    "won",
    "truncated",
    "auc_discovery",
    "first_hit_shot",
    "first_sunk_shot",
)

_CELL_SIZE = 24
_MARGIN = 36
_FRAME_SECONDS = 0.45
_COLORS = {
    "gap": "#ffffff",
    "unknown": "#dce6f2",
    "miss": "#78899b",
    "hit": "#e66101",
    "sunk": "#b2182b",
    "secret": "#fdb863",
    "grid": "#ffffff",
}


def attack_result_rows(
    results: Sequence[EpisodeResult],
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Convert public result schemas into deterministically ordered CSV rows.

    ``policy_by_run`` decouples a display label from a run identifier.  When
    omitted, the run identifier itself is the policy label, preserving useful
    output for one-policy-per-run evaluations.
    """

    labels = {} if policy_by_run is None else dict(policy_by_run)
    rows = [
        {
            "episode_id": result.episode_id,
            "run_id": result.run_id,
            "policy_id": labels.get(result.run_id, result.run_id),
            "seed": result.seed,
            "scenario": result.scenario,
            "valid_cells": result.valid_cells,
            "valid_shots": result.valid_shots,
            "invalid_attempts": result.invalid_attempts,
            "hit_segments": result.hit_segments,
            "won": result.won,
            "truncated": result.truncated,
            "auc_discovery": result.auc_discovery,
            "first_hit_shot": result.first_hit_shot,
            "first_sunk_shot": result.first_sunk_shot,
        }
        for result in results
    ]
    return sorted(rows, key=lambda row: (str(row["scenario"]), str(row["policy_id"]), str(row["episode_id"])))


def write_attack_results_csv(
    results: Sequence[EpisodeResult],
    path: str | Path,
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> Path:
    """Write one stable, UTF-8 public CSV record per attack episode."""

    destination = _prepare_destination(path)
    rows = attack_result_rows(results, policy_by_run=policy_by_run)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def attack_summary_markdown(
    results: Sequence[EpisodeResult],
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> str:
    """Summarize attack episodes in a compact comparison table in Markdown."""

    rows = attack_result_rows(results, policy_by_run=policy_by_run)
    if not rows:
        raise ValueError("results must contain at least one attack episode")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = str(row["scenario"]), str(row["policy_id"])
        grouped.setdefault(key, []).append(row)

    lines = [
        "| Scenario | Policy | Episodes | Mean shots | Win rate | Mean AUC discovery |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (scenario, policy), group in sorted(grouped.items()):
        lines.append(
            "| {scenario} | {policy} | {episodes} | {shots:.2f} | {win_rate:.1%} | {auc:.3f} |".format(
                scenario=scenario,
                policy=policy,
                episodes=len(group),
                shots=fmean(float(row["valid_shots"]) for row in group),
                win_rate=fmean(float(bool(row["won"])) for row in group),
                auc=fmean(float(row["auc_discovery"]) for row in group),
            )
        )
    return "\n".join(lines) + "\n"


def write_attack_summary_markdown(
    results: Sequence[EpisodeResult],
    path: str | Path,
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> Path:
    """Write the deterministic attack comparison table to a Markdown file."""

    destination = _prepare_destination(path)
    destination.write_text(
        attack_summary_markdown(results, policy_by_run=policy_by_run),
        encoding="utf-8",
    )
    return destination


def plot_attack_comparison(
    results: Sequence[EpisodeResult],
    path: str | Path,
    *,
    policy_by_run: Mapping[str, str] | None = None,
) -> Path:
    """Create a static Seaborn comparison of mean valid shots by scenario.

    Lower bars indicate a more efficient attacker.  The aggregation is kept in
    Seaborn rather than pre-aggregating so individual episodes remain available
    to callers in the accompanying CSV.
    """

    rows = attack_result_rows(results, policy_by_run=policy_by_run)
    if not rows:
        raise ValueError("results must contain at least one attack episode")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    destination = _prepare_destination(path)
    scenarios = sorted({str(row["scenario"]) for row in rows})
    policies = sorted({str(row["policy_id"]) for row in rows})
    with sns.axes_style("whitegrid"):
        figure, axis = plt.subplots(figsize=(max(6.5, 2.6 * len(scenarios)), 4.8))
        sns.barplot(
            data=pd.DataFrame(rows),
            x="scenario",
            y="valid_shots",
            hue="policy_id",
            order=scenarios,
            hue_order=policies,
            estimator="mean",
            errorbar=None,
            palette="colorblind",
            ax=axis,
        )
        axis.set_title("Attack policy efficiency")
        axis.set_xlabel("Scenario")
        axis.set_ylabel("Mean valid shots to terminal state")
        axis.legend(title="Policy")
        figure.tight_layout()
        figure.savefig(destination, dpi=160, metadata={"Date": None})
        plt.close(figure)
    return destination


def animation_frames(
    trace: AttackEpisodeTrace,
    *,
    reveal_secret_on_final: bool = False,
) -> tuple[AttackFrame, ...]:
    """Return safe animation frames, revealing a secret only on explicit opt-in.

    Even if a diagnostic trace contains secrets in every snapshot, only the
    final snapshot may retain its secret field and only when the flag is true.
    """

    frames = (trace.initial_frame, *(step.frame for step in trace.steps))
    if not frames:
        raise ValueError("trace must contain an initial frame")
    sanitized = [replace(frame, secret_occupied_cells=None) for frame in frames]
    if reveal_secret_on_final:
        sanitized[-1] = frames[-1]
    return tuple(sanitized)


def write_attack_trace_gif(
    trace: AttackEpisodeTrace,
    path: str | Path,
    *,
    reveal_secret_on_final: bool = False,
    frame_duration_ms: int = int(_FRAME_SECONDS * 1000),
) -> Path:
    """Render a deterministic GIF from public snapshots.

    Diagnostic occupancy is not rendered by default.  Set
    ``reveal_secret_on_final=True`` only for a local post-episode review; it
    displays the secret occupancy exclusively in the final GIF frame.
    """

    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")

    destination = _prepare_destination(path)
    images = [
        _frame_image(frame)
        for frame in animation_frames(
            trace, reveal_secret_on_final=reveal_secret_on_final
        )
    ]
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


def _frame_image(frame: AttackFrame):
    from PIL import Image, ImageDraw

    width = _MARGIN * 2 + frame.columns * _CELL_SIZE
    height = _MARGIN * 2 + frame.rows * _CELL_SIZE
    image = Image.new("RGB", (width, height), _COLORS["gap"])
    draw = ImageDraw.Draw(image)
    valid_actions = set(frame.valid_actions)
    states = {action: "unknown" for action in valid_actions}
    if frame.secret_occupied_cells is not None:
        states.update({action: "secret" for action in frame.secret_occupied_cells})
    states.update({action: "hit" for action in frame.active_hits})
    states.update({action: "sunk" for action in frame.sunk_hits})
    states.update({action: "miss" for action in frame.misses})

    for row in range(frame.rows):
        for column in range(frame.columns):
            action = row * frame.columns + column
            if action not in valid_actions:
                continue
            left = _MARGIN + column * _CELL_SIZE
            top = _MARGIN + row * _CELL_SIZE
            draw.rectangle(
                (left, top, left + _CELL_SIZE, top + _CELL_SIZE),
                fill=_COLORS[states[action]],
                outline=_COLORS["grid"],
            )
    return image


def _prepare_destination(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
