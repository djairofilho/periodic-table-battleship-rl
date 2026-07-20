from __future__ import annotations

import csv

import pytest

from periodic_table_battleship_rl.analysis.campaign import (
    CampaignObservation,
    compare_policies,
    load_campaign_csv,
    summarize_policies,
    write_campaign_analysis,
)


def _observation(
    episode_id: str,
    policy_id: str,
    seed: int,
    value: float,
    *,
    attacker_id: str | None = None,
) -> CampaignObservation:
    return CampaignObservation(
        episode_id=episode_id,
        policy_id=policy_id,
        seed=seed,
        scenario="battleship",
        metric="valid_shots",
        value=value,
        attacker_id=attacker_id,
    )


def test_summaries_reduce_repeated_ppo_episodes_within_each_seed() -> None:
    observations = (
        _observation("ppo-a", "PPO", 1, 10.0),
        _observation("ppo-b", "PPO", 1, 14.0),
        _observation("ppo-c", "PPO", 2, 20.0),
        _observation("ppo-d", "PPO", 2, 24.0),
        _observation("baseline-a", "Baseline", 1, 30.0),
        _observation("baseline-b", "Baseline", 2, 40.0),
    )

    summaries = summarize_policies(
        observations,
        experiment="attack",
        direction="lower",
        resamples=200,
    )
    ppo = next(summary for summary in summaries if summary.policy_id == "PPO")

    assert ppo.seed_count == 2
    assert ppo.episode_count == 4
    assert ppo.mean == 17.0
    assert ppo.lower_95 <= ppo.mean <= ppo.upper_95


def test_comparisons_are_paired_by_seed_and_reproducible() -> None:
    observations = (
        _observation("ppo-a", "PPO", 1, 10.0),
        _observation("ppo-b", "PPO", 1, 14.0),
        _observation("ppo-c", "PPO", 2, 20.0),
        _observation("ppo-d", "PPO", 2, 24.0),
        _observation("baseline-a", "Baseline", 1, 30.0),
        _observation("baseline-b", "Baseline", 2, 40.0),
    )

    first = compare_policies(
        observations,
        experiment="attack",
        candidate_policy="PPO",
        reference_policies=("Baseline",),
        direction="lower",
        resamples=200,
    )
    second = compare_policies(
        observations,
        experiment="attack",
        candidate_policy="PPO",
        reference_policies=("Baseline",),
        direction="lower",
        resamples=200,
    )

    assert first == second
    assert first[0].candidate_minus_reference_mean == -18.0
    assert first[0].conclusion == "candidate_favored"


def test_comparisons_reject_unpaired_seed_inventories() -> None:
    observations = (
        _observation("ppo", "PPO", 1, 10.0),
        _observation("baseline", "Baseline", 2, 20.0),
    )

    with pytest.raises(ValueError, match="identical seeds"):
        compare_policies(
            observations,
            experiment="attack",
            candidate_policy="PPO",
            reference_policies=("Baseline",),
            direction="lower",
        )


def test_write_campaign_analysis_emits_portable_artifacts(tmp_path) -> None:
    attack_path = tmp_path / "attack.csv"
    placement_path = tmp_path / "placement.csv"
    _write_csv(
        attack_path,
        ("episode_id", "policy_id", "seed", "scenario", "valid_shots"),
        (
            ("ppo-1", "MaskablePPO (multi-seed)", 1, "battleship", 10),
            ("ppo-2", "MaskablePPO (multi-seed)", 2, "battleship", 12),
            ("hunt-1", "Hunt-target", 1, "battleship", 20),
            ("hunt-2", "Hunt-target", 2, "battleship", 24),
            ("random-1", "Random masked", 1, "battleship", 15),
            ("random-2", "Random masked", 2, "battleship", 18),
        ),
    )
    _write_csv(
        placement_path,
        (
            "episode_id",
            "policy_id",
            "seed",
            "scenario",
            "attacker_id",
            "valid_shots_to_sink",
        ),
        (
            ("ppo-1", "MaskablePPO placement (multi-seed)", 1, "battleship", "mix", 50),
            ("ppo-2", "MaskablePPO placement (multi-seed)", 2, "battleship", "mix", 55),
            ("disp-1", "dispersion-placement-v1", 1, "battleship", "mix", 40),
            ("disp-2", "dispersion-placement-v1", 2, "battleship", "mix", 45),
            ("resist-1", "hunt-target-resistant-placement-v1", 1, "battleship", "mix", 42),
            ("resist-2", "hunt-target-resistant-placement-v1", 2, "battleship", "mix", 47),
            ("random-1", "random-legal-placement-v1", 1, "battleship", "mix", 44),
            ("random-2", "random-legal-placement-v1", 2, "battleship", "mix", 49),
        ),
    )

    summaries, comparisons = write_campaign_analysis(
        attack_csv=attack_path,
        placement_csv=placement_path,
        destination=tmp_path / "analysis",
        resamples=200,
    )

    assert len(summaries) == 7
    assert len(comparisons) == 5
    assert (tmp_path / "analysis" / "analysis-report.json").is_file()
    assert (tmp_path / "analysis" / "analysis-summary.md").is_file()
    assert (tmp_path / "analysis" / "policy-comparisons.csv").is_file()
    assert "Análise v0.3 por seed" in (
        tmp_path / "analysis" / "analysis-summary.md"
    ).read_text(encoding="utf-8")


def test_load_campaign_csv_rejects_duplicate_episode_id(tmp_path) -> None:
    source = tmp_path / "bad.csv"
    _write_csv(
        source,
        ("episode_id", "policy_id", "seed", "scenario", "valid_shots"),
        (
            ("same", "PPO", 1, "battleship", 10),
            ("same", "PPO", 2, "battleship", 11),
        ),
    )

    with pytest.raises(ValueError, match="duplicate episode_id"):
        load_campaign_csv(source, metric="valid_shots")


def _write_csv(path, headers, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)
