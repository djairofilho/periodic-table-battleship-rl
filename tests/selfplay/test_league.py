"""Tests for reproducible self-play league and provenance contracts."""

from __future__ import annotations

import json

import pytest

from periodic_table_battleship_rl.selfplay import (
    AlternatingSelfPlaySchedule,
    FrozenEvaluationSuite,
    SelfPlayCampaignConfig,
    SelfPlayCampaignRecord,
    SnapshotLeague,
    SnapshotProvenance,
    persist_self_play_campaign,
)
from periodic_table_battleship_rl.training.attack import ATTACK_POLICY_ID
from periodic_table_battleship_rl.training.placement import PLACEMENT_POLICY_ID


SCENARIO = "periodic-table-battleship"


def _snapshot(
    snapshot_id: str,
    role: str,
    *,
    round_index: int = 0,
    parents: tuple[str, ...] = (),
) -> SnapshotProvenance:
    return SnapshotProvenance(
        snapshot_id=snapshot_id,
        role=role,  # type: ignore[arg-type]
        policy_id=ATTACK_POLICY_ID if role == "attacker" else PLACEMENT_POLICY_ID,
        scenario=SCENARIO,
        source_run_id=f"run-{snapshot_id}",
        checkpoint_sha256=("a" if role == "attacker" else "b") * 64,
        training_round=round_index,
        parent_snapshot_ids=parents,
    )


def _suite() -> FrozenEvaluationSuite:
    return FrozenEvaluationSuite(
        attacker_evaluator_ids=("random-masked-v1", "hunt-target-v1"),
        placement_policy_ids=("random-legal-placement-v1", "dispersion-placement-v1"),
    )


def _league() -> SnapshotLeague:
    return SnapshotLeague(
        scenario=SCENARIO,
        snapshots=(
            _snapshot("attacker-bootstrap", "attacker"),
            _snapshot("placer-bootstrap", "placer"),
        ),
    )


def _config() -> SelfPlayCampaignConfig:
    return SelfPlayCampaignConfig(
        campaign_id="self-play-periodic-v1",
        scenario=SCENARIO,
        seed=20260720,
        round_count=4,
        attacker_timesteps=50_000,
        placer_timesteps=40_000,
        frozen_evaluation=_suite(),
    )


def test_schedule_alternates_roles_with_reproducible_opponents_and_seeds() -> None:
    schedule = AlternatingSelfPlaySchedule(config=_config())
    league = _league()

    first = schedule.plan_round(league, round_index=0)
    second = schedule.plan_round(league, round_index=1)

    assert first.learner_role == "attacker"
    assert first.opponent_role == "placer"
    assert first.opponent_snapshot_id == "placer-bootstrap"
    assert first.timesteps == 50_000
    assert first.frozen_evaluation_target_ids == _suite().placement_policy_ids
    assert second.learner_role == "placer"
    assert second.opponent_role == "attacker"
    assert second.opponent_snapshot_id == "attacker-bootstrap"
    assert second.timesteps == 40_000
    assert second.frozen_evaluation_target_ids == _suite().attacker_evaluator_ids
    assert schedule.plan_round(league, round_index=0) == first


def test_league_sampling_is_deterministic_and_independent_of_input_order() -> None:
    snapshots = (
        _snapshot("attacker-b", "attacker"),
        _snapshot("attacker-a", "attacker"),
        _snapshot("placer-bootstrap", "placer"),
    )
    left = SnapshotLeague(scenario=SCENARIO, snapshots=snapshots)
    right = SnapshotLeague(scenario=SCENARIO, snapshots=tuple(reversed(snapshots)))

    assert left.select_opponent("placer", seed=11) == right.select_opponent(
        "placer", seed=11
    )


def test_campaign_record_requires_selected_parent_and_grows_the_pool(tmp_path) -> None:
    record = SelfPlayCampaignRecord(config=_config(), initial_league=_league())
    first = record.next_round
    assert first is not None

    first_snapshot = _snapshot(
        "attacker-round-0",
        "attacker",
        parents=(first.opponent_snapshot_id,),
    )
    record = record.record_snapshot(first_snapshot)
    second = record.next_round

    assert second is not None
    assert len(record.current_league.snapshots) == 3
    assert record.completed_rounds == (first,)
    assert second.learner_role == "placer"

    output = persist_self_play_campaign(tmp_path / "campaign.json", record)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["config"]["frozen_evaluation"] == _suite().to_dict()
    assert persisted["produced_snapshots"][0]["checkpoint_sha256"] == "a" * 64
    assert "checkpoint_path" not in output.read_text(encoding="utf-8")


def test_campaign_record_rejects_a_snapshot_without_selected_opponent_parent() -> None:
    record = SelfPlayCampaignRecord(config=_config(), initial_league=_league())

    with pytest.raises(ValueError, match="selected opponent"):
        record.record_snapshot(_snapshot("attacker-round-0", "attacker"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"checkpoint_sha256": "not-a-digest"}, "SHA-256"),
        ({"policy_id": PLACEMENT_POLICY_ID}, "attacker snapshots"),
        ({"parent_snapshot_ids": ("duplicate", "duplicate")}, "duplicates"),
    ),
)
def test_snapshot_contract_rejects_nonportable_or_incompatible_provenance(
    kwargs, message
) -> None:
    values = {
        "snapshot_id": "attacker-bootstrap",
        "role": "attacker",
        "policy_id": ATTACK_POLICY_ID,
        "scenario": SCENARIO,
        "source_run_id": "run-attacker-bootstrap",
        "checkpoint_sha256": "a" * 64,
        "training_round": 0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SnapshotProvenance(**values)


def test_league_rejects_missing_parent_and_wrong_scenario() -> None:
    with pytest.raises(ValueError, match="parents"):
        SnapshotLeague(
            scenario=SCENARIO,
            snapshots=(_snapshot("attacker-child", "attacker", parents=("missing",)),),
        )

    with pytest.raises(ValueError, match="scenario"):
        _league().with_snapshot(
            SnapshotProvenance(
                snapshot_id="placer-other",
                role="placer",
                policy_id=PLACEMENT_POLICY_ID,
                scenario="battleship",
                source_run_id="run-placer-other",
                checkpoint_sha256="b" * 64,
                training_round=1,
            )
        )
