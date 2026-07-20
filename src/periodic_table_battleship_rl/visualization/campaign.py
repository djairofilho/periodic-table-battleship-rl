"""Deterministic learning-curve artifacts for multi-seed training campaigns.

The module is intentionally independent from a trainer.  Callers persist a
small sequence of :class:`LearningCurvePoint` records as checkpoints arrive,
then use these functions to create a static curve and an animated progression
from the same public data.  No training state, environment, or policy weights
are inspected here.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Literal, Sequence


IntervalKind = Literal["ci95", "range"]
"""Supported uncertainty bands for values observed across training seeds."""


@dataclass(frozen=True, slots=True, order=True)
class LearningCurvePoint:
    """One scalar measurement from one seed at a training checkpoint."""

    seed: int
    stage: int
    value: float

    def __post_init__(self) -> None:
        if self.stage < 0:
            raise ValueError("stage must be non-negative")
        if not isfinite(self.value):
            raise ValueError("value must be finite")


@dataclass(frozen=True, slots=True)
class LearningCurveSummary:
    """Aggregate of a single checkpoint, retaining its seed sample count."""

    stage: int
    mean: float
    lower: float
    upper: float
    seed_count: int


def summarize_learning_curve(
    points: Sequence[LearningCurvePoint], *, interval: IntervalKind = "ci95"
) -> tuple[LearningCurveSummary, ...]:
    """Aggregate public checkpoint values into a deterministic uncertainty band.

    ``ci95`` is a normal-approximation 95% confidence interval for the mean
    across seeds.  ``range`` instead draws the observed minimum and maximum.
    Missing seed/stage combinations are allowed, which accommodates interrupted
    runs while preserving the exact number of contributing seeds per stage.
    """

    if not points:
        raise ValueError("points must contain at least one learning-curve point")
    if interval not in {"ci95", "range"}:
        raise ValueError("interval must be either 'ci95' or 'range'")

    grouped: dict[int, list[float]] = {}
    observed: set[tuple[int, int]] = set()
    for point in points:
        key = point.seed, point.stage
        if key in observed:
            raise ValueError("points must contain at most one value per seed and stage")
        observed.add(key)
        grouped.setdefault(point.stage, []).append(float(point.value))

    summaries: list[LearningCurveSummary] = []
    for stage in sorted(grouped):
        values = grouped[stage]
        mean = fmean(values)
        if interval == "range" or len(values) == 1:
            lower, upper = min(values), max(values)
        else:
            half_width = 1.96 * stdev(values) / sqrt(len(values))
            lower, upper = mean - half_width, mean + half_width
        summaries.append(
            LearningCurveSummary(
                stage=stage,
                mean=mean,
                lower=lower,
                upper=upper,
                seed_count=len(values),
            )
        )
    return tuple(summaries)


def plot_learning_curve(
    points: Sequence[LearningCurvePoint],
    path: str | Path,
    *,
    interval: IntervalKind = "ci95",
    title: str = "Training learning curve",
    x_label: str = "Training stage",
    y_label: str = "Mean evaluation value",
) -> Path:
    """Write a PNG of the mean learning curve with a per-seed uncertainty band."""

    summaries = summarize_learning_curve(points, interval=interval)
    destination = _prepare_destination(path)
    figure = _curve_figure(
        summaries,
        visible_count=len(summaries),
        title=title,
        x_label=x_label,
        y_label=y_label,
    )
    try:
        figure.savefig(destination, dpi=160, metadata={"Date": None})
    finally:
        _close_figure(figure)
    return destination


def write_learning_progress_gif(
    points: Sequence[LearningCurvePoint],
    path: str | Path,
    *,
    checkpoints: Sequence[int] | None = None,
    interval: IntervalKind = "ci95",
    title: str = "Training learning curve",
    x_label: str = "Training stage",
    y_label: str = "Mean evaluation value",
    frame_duration_ms: int = 550,
) -> Path:
    """Write a GIF that reveals a multi-seed trace progressively at checkpoints.

    ``checkpoints`` selects stages to render, in chronological order.  When it
    is omitted, every observed stage becomes one frame.  The fixed axes make
    changes across frames directly comparable instead of re-scaling each one.
    """

    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")
    summaries = summarize_learning_curve(points, interval=interval)
    stages = tuple(summary.stage for summary in summaries)
    selected = _selected_checkpoints(stages, checkpoints)
    indices = [stages.index(checkpoint) + 1 for checkpoint in selected]
    destination = _prepare_destination(path)
    images = [
        _figure_image(
            _curve_figure(
                summaries,
                visible_count=index,
                title=title,
                x_label=x_label,
                y_label=y_label,
            )
        )
        for index in indices
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


def write_checkpoint_progress_gif(
    points: Sequence[LearningCurvePoint], path: str | Path, **kwargs: object
) -> Path:
    """Compatibility name for :func:`write_learning_progress_gif`."""

    return write_learning_progress_gif(points, path, **kwargs)  # type: ignore[arg-type]


def _selected_checkpoints(
    stages: tuple[int, ...], checkpoints: Sequence[int] | None
) -> tuple[int, ...]:
    if checkpoints is None:
        return stages
    selected = tuple(checkpoints)
    if not selected:
        raise ValueError("checkpoints must not be empty")
    if tuple(sorted(selected)) != selected or len(set(selected)) != len(selected):
        raise ValueError("checkpoints must be unique and sorted in ascending order")
    missing = sorted(set(selected).difference(stages))
    if missing:
        raise ValueError(f"checkpoints are not present in points: {missing}")
    return selected


def _curve_figure(
    summaries: Sequence[LearningCurveSummary],
    *,
    visible_count: int,
    title: str,
    x_label: str,
    y_label: str,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not 1 <= visible_count <= len(summaries):
        raise ValueError("visible_count must select at least one summary")
    visible = summaries[:visible_count]
    all_stages = [summary.stage for summary in summaries]
    all_lower = [summary.lower for summary in summaries]
    all_upper = [summary.upper for summary in summaries]
    values = all_lower + all_upper
    minimum, maximum = min(values), max(values)
    padding = max(0.05, (maximum - minimum) * 0.08)

    with sns.axes_style("whitegrid"):
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        axis.fill_between(
            [summary.stage for summary in visible],
            [summary.lower for summary in visible],
            [summary.upper for summary in visible],
            color="#0072b2",
            alpha=0.22,
            label="Per-seed interval",
        )
        axis.plot(
            [summary.stage for summary in visible],
            [summary.mean for summary in visible],
            color="#0072b2",
            marker="o",
            linewidth=2.0,
            label="Seed mean",
        )
        current = visible[-1]
        axis.annotate(
            f"n={current.seed_count}",
            (current.stage, current.mean),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
        )
        axis.set_xlim(min(all_stages), max(all_stages) if len(all_stages) > 1 else min(all_stages) + 1)
        axis.set_ylim(minimum - padding, maximum + padding)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.legend(loc="best")
        figure.tight_layout()
    return figure


def _figure_image(figure):
    from PIL import Image

    buffer = BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=120, metadata={"Date": None})
    finally:
        _close_figure(figure)
    buffer.seek(0)
    with Image.open(buffer) as source:
        return source.convert("RGB").copy()


def _close_figure(figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)


def _prepare_destination(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
