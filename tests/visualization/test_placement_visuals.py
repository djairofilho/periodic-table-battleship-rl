from __future__ import annotations

import csv

from PIL import Image

from periodic_table_battleship_rl.evaluation import PlacementResult
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP
from periodic_table_battleship_rl.visualization.placement import (
    placement_result_rows,
    placement_summary_markdown,
    plot_placement_comparison,
    plot_placement_segment_heatmap,
    write_placement_results_csv,
    write_placement_summary_markdown,
    write_placement_trace_gif,
)


def _result(
    *,
    episode_id: str,
    run_id: str,
    attacker_id: str,
    shots: int,
    actions: tuple[int, ...] = (0, 18, 36, 54, 72),
) -> PlacementResult:
    return PlacementResult(
        episode_id=episode_id,
        run_id=run_id,
        seed=11,
        scenario="battleship",
        attacker_id=attacker_id,
        attacker_seed=19,
        placement_actions=actions,
        valid_cells=100,
        valid_shots_to_sink=shots,
        hit_segments=17,
        sunk_ship_lengths=(5, 4, 3, 3, 2),
        auc_discovery=0.4,
        first_hit_shot=3,
        first_sunk_shot=9,
        all_sunk_shot=shots,
    )


def _results() -> tuple[PlacementResult, ...]:
    return (
        _result(
            episode_id="random-a",
            run_id="random-placement",
            attacker_id="random-masked-v1",
            shots=61,
        ),
        _result(
            episode_id="hunt-a",
            run_id="random-placement",
            attacker_id="hunt-target-v1",
            shots=49,
        ),
        _result(
            episode_id="mixture-a",
            run_id="ppo-placement",
            attacker_id="frozen-defensive-mixture-v1",
            shots=67,
        ),
    )


def test_placement_artifacts_are_ordered_readable_and_deterministic(tmp_path) -> None:
    results = _results()
    labels = {"random-placement": "random", "ppo-placement": "maskable-ppo"}

    csv_path = write_placement_results_csv(
        results, tmp_path / "episodes.csv", policy_by_run=labels
    )
    markdown_path = write_placement_summary_markdown(
        results, tmp_path / "summary.md", policy_by_run=labels
    )
    chart_path = plot_placement_comparison(
        results, tmp_path / "comparison.png", policy_by_run=labels
    )

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["attacker_id"] for row in rows] == [
        "frozen-defensive-mixture-v1",
        "hunt-target-v1",
        "random-masked-v1",
    ]
    assert rows[0]["placement_actions"] == "0;18;36;54;72"
    assert "| battleship | hunt-target-v1 | random | 1 | 49.00 | 100.0% |" in markdown_path.read_text(
        encoding="utf-8"
    )
    assert placement_summary_markdown(results, policy_by_run=labels) == markdown_path.read_text(
        encoding="utf-8"
    )
    assert placement_result_rows(results, policy_by_run=labels) == placement_result_rows(
        tuple(reversed(results)), policy_by_run=labels
    )
    assert chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_heatmap_and_gif_preserve_topology_gaps_and_sequence(tmp_path) -> None:
    result = _results()[0]
    periodic_result = PlacementResult(
        episode_id="periodic",
        run_id="periodic-placement",
        seed=7,
        scenario="periodic-table-battleship",
        attacker_id="frozen-defensive-mixture-v1",
        attacker_seed=23,
        placement_actions=(54, 72, 93, 111, 147),
        valid_cells=118,
        valid_shots_to_sink=72,
        hit_segments=17,
        sunk_ship_lengths=(5, 4, 3, 3, 2),
        auc_discovery=0.3,
        all_sunk_shot=72,
    )

    heatmap_path = plot_placement_segment_heatmap(
        (periodic_result,), PERIODIC_TABLE_BATTLESHIP, tmp_path / "periodic-heatmap.png"
    )
    gif_path = write_placement_trace_gif(result, BATTLESHIP, tmp_path / "placement.gif")
    periodic_gif = write_placement_trace_gif(
        periodic_result, PERIODIC_TABLE_BATTLESHIP, tmp_path / "periodic-placement.gif"
    )
    second_gif = write_placement_trace_gif(
        result, BATTLESHIP, tmp_path / "placement-second.gif"
    )

    assert heatmap_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert gif_path.read_bytes() == second_gif.read_bytes()
    with Image.open(gif_path) as image:
        assert image.format == "GIF"
        assert image.n_frames == 6
    with Image.open(periodic_gif) as image:
        image.seek(0)
        pixels = image.convert("RGB")
        assert pixels.getpixel((36 + 12, 36 + 12)) != (255, 255, 255)
        assert pixels.getpixel((36 + 24 + 12, 36 + 12)) == (255, 255, 255)


def test_heatmap_rejects_unknown_attacker_and_illegal_action(tmp_path) -> None:
    result = _results()[0]
    illegal = _result(
        episode_id="illegal",
        run_id="random-placement",
        attacker_id="random-masked-v1",
        shots=61,
        actions=(0, 0, 36, 54, 72),
    )

    import pytest

    with pytest.raises(ValueError, match="requested attacker_id"):
        plot_placement_segment_heatmap(
            (result,), BATTLESHIP, tmp_path / "none.png", attacker_id="missing"
        )
    with pytest.raises(ValueError, match="illegal placement"):
        write_placement_trace_gif(illegal, BATTLESHIP, tmp_path / "illegal.gif")
