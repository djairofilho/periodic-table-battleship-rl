from __future__ import annotations

import csv

from PIL import Image

from periodic_table_battleship_rl.evaluation import EpisodeResult
from periodic_table_battleship_rl.rendering import (
    AttackEpisodeTrace,
    AttackFrame,
    AttackTraceStep,
)
from periodic_table_battleship_rl.visualization.attack import (
    animation_frames,
    attack_summary_markdown,
    plot_attack_comparison,
    write_attack_results_csv,
    write_attack_summary_markdown,
    write_attack_trace_gif,
)


def _results() -> tuple[EpisodeResult, ...]:
    common = {
        "valid_cells": 100,
        "invalid_attempts": 0,
        "hit_segments": 17,
        "sunk_ship_lengths": (2, 3, 3, 4, 5),
        "won": True,
        "truncated": False,
        "auc_discovery": 0.42,
        "first_hit_shot": 2,
        "first_sunk_shot": 8,
    }
    return (
        EpisodeResult(
            episode_id="random-1",
            run_id="random-run",
            seed=11,
            scenario="battleship",
            valid_shots=82,
            **common,
        ),
        EpisodeResult(
            episode_id="hunt-1",
            run_id="hunt-run",
            seed=11,
            scenario="battleship",
            valid_shots=57,
            **common,
        ),
    )


def _trace() -> AttackEpisodeTrace:
    initial = AttackFrame(
        topology_name="battleship",
        rows=1,
        columns=2,
        valid_actions=(0, 1),
        active_hits=(),
        sunk_hits=(),
        misses=(),
        secret_occupied_cells=(1,),
    )
    final = AttackFrame(
        topology_name="battleship",
        rows=1,
        columns=2,
        valid_actions=(0, 1),
        active_hits=(0,),
        sunk_hits=(),
        misses=(),
        secret_occupied_cells=(1,),
    )
    return AttackEpisodeTrace(
        initial,
        (
            AttackTraceStep(
                index=1,
                action=0,
                reward=1.0,
                terminated=False,
                truncated=False,
                info={"is_hit": True},
                frame=final,
            ),
        ),
    )


def test_attack_result_artifacts_are_ordered_and_readable(tmp_path) -> None:
    results = _results()
    labels = {"random-run": "random", "hunt-run": "hunt-target"}

    csv_path = write_attack_results_csv(results, tmp_path / "episodes.csv", policy_by_run=labels)
    markdown_path = write_attack_summary_markdown(
        results, tmp_path / "summary.md", policy_by_run=labels
    )
    chart_path = plot_attack_comparison(
        results, tmp_path / "comparison.png", policy_by_run=labels
    )

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["policy_id"] for row in rows] == ["hunt-target", "random"]
    assert "| battleship | hunt-target | 1 | 57.00 | 100.0% | 0.420 |" in markdown_path.read_text(encoding="utf-8")
    assert attack_summary_markdown(results, policy_by_run=labels) == markdown_path.read_text(encoding="utf-8")
    assert chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_trace_gif_hides_secret_except_when_final_reveal_is_requested(tmp_path) -> None:
    trace = _trace()

    public_frames = animation_frames(trace)
    revealed_frames = animation_frames(trace, reveal_secret_on_final=True)
    public_path = write_attack_trace_gif(trace, tmp_path / "public.gif")
    revealed_path = write_attack_trace_gif(
        trace, tmp_path / "revealed.gif", reveal_secret_on_final=True
    )

    assert all(frame.secret_occupied_cells is None for frame in public_frames)
    assert revealed_frames[0].secret_occupied_cells is None
    assert revealed_frames[-1].secret_occupied_cells == (1,)
    with Image.open(public_path) as image:
        assert image.n_frames == 2
        assert image.format == "GIF"
    with Image.open(revealed_path) as image:
        assert image.n_frames == 2
        assert image.format == "GIF"
