from __future__ import annotations

from PIL import Image
import pytest

from periodic_table_battleship_rl.visualization.campaign import (
    LearningCurvePoint,
    plot_learning_curve,
    summarize_learning_curve,
    write_checkpoint_progress_gif,
    write_learning_progress_gif,
)


def _points() -> tuple[LearningCurvePoint, ...]:
    return (
        LearningCurvePoint(seed=11, stage=0, value=0.20),
        LearningCurvePoint(seed=19, stage=0, value=0.30),
        LearningCurvePoint(seed=11, stage=10, value=0.50),
        LearningCurvePoint(seed=19, stage=10, value=0.70),
        LearningCurvePoint(seed=11, stage=20, value=0.80),
        LearningCurvePoint(seed=19, stage=20, value=0.90),
    )


def test_curve_summary_is_ordered_and_exposes_seed_intervals() -> None:
    summaries = summarize_learning_curve(tuple(reversed(_points())), interval="range")

    assert [summary.stage for summary in summaries] == [0, 10, 20]
    assert summaries[0].mean == pytest.approx(0.25)
    assert summaries[0].lower == pytest.approx(0.20)
    assert summaries[0].upper == pytest.approx(0.30)
    assert summaries[-1].seed_count == 2


def test_learning_curve_png_and_checkpoint_gif_are_valid_and_deterministic(tmp_path) -> None:
    points = _points()
    png_path = plot_learning_curve(points, tmp_path / "curve.png")
    gif_path = write_learning_progress_gif(
        points,
        tmp_path / "progress.gif",
        checkpoints=(0, 10, 20),
    )
    second_gif = write_checkpoint_progress_gif(
        points,
        tmp_path / "progress-second.gif",
        checkpoints=(0, 10, 20),
    )

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert gif_path.read_bytes() == second_gif.read_bytes()
    with Image.open(gif_path) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3


def test_curve_rejects_ambiguous_points_and_unknown_checkpoints(tmp_path) -> None:
    duplicate = (*_points(), LearningCurvePoint(seed=11, stage=0, value=0.40))

    with pytest.raises(ValueError, match="at most one"):
        summarize_learning_curve(duplicate)
    with pytest.raises(ValueError, match="not present"):
        write_learning_progress_gif(_points(), tmp_path / "bad.gif", checkpoints=(5,))
