"""Tests for the explicit, public-state coupled self-play adapters."""

from __future__ import annotations

import json

import numpy as np
import pytest

from periodic_table_battleship_rl.placement.baselines import RandomLegalPlacementPolicy
from periodic_table_battleship_rl.selfplay import (
    CoupledAttackEnv,
    CoupledSelfPlayRunner,
    CoupledTrainingOutput,
    FrozenEvaluationSuite,
    PlacementPolicyFleetSampler,
    PublicAttackPolicyEvaluator,
    SelfPlayCampaignConfig,
    SelfPlayCampaignRecord,
    SnapshotLeague,
    SnapshotProvenance,
)
from periodic_table_battleship_rl.topology import PERIODIC_TABLE_BATTLESHIP
from periodic_table_battleship_rl.training.attack import ATTACK_POLICY_ID
from periodic_table_battleship_rl.training.placement import PLACEMENT_POLICY_ID


class LowestLegalPolicy:
    """A public-only deterministic policy suitable for adapter tests."""

    policy_id = "test-lowest-legal-v1"

    def select_action(self, observation, action_mask, *, deterministic: bool = True) -> int:
        del observation, deterministic
        return int(np.flatnonzero(action_mask)[0])


def _placer() -> PlacementPolicyFleetSampler:
    return PlacementPolicyFleetSampler(
        policy=RandomLegalPlacementPolicy(PERIODIC_TABLE_BATTLESHIP, seed=9),
        sampler_id="placer-bootstrap-runtime",
    )


def _attacker() -> PublicAttackPolicyEvaluator:
    return PublicAttackPolicyEvaluator(
        policy=LowestLegalPolicy(),
        topology=PERIODIC_TABLE_BATTLESHIP,
        evaluator_id="attacker-bootstrap-runtime",
    )


def _snapshot(snapshot_id: str, role: str) -> SnapshotProvenance:
    return SnapshotProvenance(
        snapshot_id=snapshot_id,
        role=role,  # type: ignore[arg-type]
        policy_id=ATTACK_POLICY_ID if role == "attacker" else PLACEMENT_POLICY_ID,
        scenario=PERIODIC_TABLE_BATTLESHIP.name,
        source_run_id=f"run-{snapshot_id}",
        checkpoint_sha256=("a" if role == "attacker" else "b") * 64,
        training_round=0,
    )


def test_coupled_attack_env_draws_a_legal_hidden_fleet_from_public_placer() -> None:
    environment = CoupledAttackEnv(PERIODIC_TABLE_BATTLESHIP, _placer())
    observation, _ = environment.reset(seed=101)

    assert environment.placer_id == "placer-bootstrap-runtime"
    assert observation.shape == (4, 10, 18)
    assert environment.action_masks().sum() == PERIODIC_TABLE_BATTLESHIP.valid_cell_count

    terminated = truncated = False
    while not (terminated or truncated):
        action = LowestLegalPolicy().select_action(observation, environment.action_masks())
        observation, _, terminated, truncated, info = environment.step(action)

    assert terminated
    assert not truncated
    assert int(info["valid_shots"]) <= PERIODIC_TABLE_BATTLESHIP.valid_cell_count


def test_public_attack_adapter_scores_a_placed_fleet_without_hidden_policy_input() -> None:
    fleet = _placer().sample_fleet(
        PERIODIC_TABLE_BATTLESHIP, rng=np.random.default_rng(701)
    )

    shots = _attacker().evaluate(fleet, rng=np.random.default_rng(702))

    assert 17 <= shots <= PERIODIC_TABLE_BATTLESHIP.valid_cell_count


class _FakeTrainer:
    def __init__(self, directory) -> None:
        self.directory = directory
        self.calls: list[tuple[str, str]] = []

    def train_attacker(self, plan, environment) -> CoupledTrainingOutput:
        self.calls.append((plan.learner_role, environment.placer_id))
        path = self.directory / f"attacker-{plan.round_index}.bin"
        path.write_bytes(b"attacker")
        return CoupledTrainingOutput(
            checkpoint_path=path,
            source_run_id=f"attack-round-{plan.round_index}",
            runtime_opponent=_attacker(),
        )

    def train_placer(self, plan, evaluator) -> CoupledTrainingOutput:
        self.calls.append((plan.learner_role, evaluator.evaluator_id))
        path = self.directory / f"placer-{plan.round_index}.bin"
        path.write_bytes(b"placer")
        return CoupledTrainingOutput(
            checkpoint_path=path,
            source_run_id=f"placer-round-{plan.round_index}",
            runtime_opponent=_placer(),
        )


class _FrozenSuite:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def evaluate(self, *, role, runtime_opponent, target_ids):
        del runtime_opponent
        self.calls.append((role, target_ids))
        return {target_id: float(index + 1) for index, target_id in enumerate(target_ids)}


def test_runner_rejects_incomplete_or_non_numeric_frozen_evidence() -> None:
    plan_targets = ("fixed-a", "fixed-b")

    with pytest.raises(ValueError, match="exactly every target"):
        CoupledSelfPlayRunner._validate_frozen_scores(  # noqa: SLF001
            type("Plan", (), {"frozen_evaluation_target_ids": plan_targets})(),
            {"fixed-a": 1.0},
        )
    with pytest.raises(ValueError, match="finite numeric"):
        CoupledSelfPlayRunner._validate_frozen_scores(  # noqa: SLF001
            type("Plan", (), {"frozen_evaluation_target_ids": plan_targets})(),
            {"fixed-a": 1.0, "fixed-b": "not-a-score"},
        )


def test_runner_alternates_and_persists_frozen_evidence_without_promotion(tmp_path) -> None:
    suite = FrozenEvaluationSuite(
        attacker_evaluator_ids=("random-masked-v1", "hunt-target-v1"),
        placement_policy_ids=("random-legal-placement-v1",),
    )
    config = SelfPlayCampaignConfig(
        campaign_id="coupled-test",
        scenario=PERIODIC_TABLE_BATTLESHIP.name,
        seed=303,
        round_count=2,
        attacker_timesteps=10,
        placer_timesteps=10,
        frozen_evaluation=suite,
    )
    record = SelfPlayCampaignRecord(
        config=config,
        initial_league=SnapshotLeague(
            scenario=PERIODIC_TABLE_BATTLESHIP.name,
            snapshots=(
                _snapshot("attacker-bootstrap", "attacker"),
                _snapshot("placer-bootstrap", "placer"),
            ),
        ),
    )
    trainer = _FakeTrainer(tmp_path)
    frozen_suite = _FrozenSuite()
    runner = CoupledSelfPlayRunner(
        record=record,
        topology=PERIODIC_TABLE_BATTLESHIP,
        trainer=trainer,
        frozen_suite=frozen_suite,
        runtime_opponents={
            "attacker-bootstrap": _attacker(),
            "placer-bootstrap": _placer(),
        },
        output_directory=tmp_path / "audit",
    )

    first = runner.run_next_round()
    second = runner.run_next_round()

    assert first is not None and first.role == "attacker"
    assert second is not None and second.role == "placer"
    assert runner.run_next_round() is None
    assert trainer.calls[0] == ("attacker", "placer-bootstrap-runtime")
    assert frozen_suite.calls == [
        ("attacker", ("random-legal-placement-v1",)),
        ("placer", ("random-masked-v1", "hunt-target-v1")),
    ]
    round_audit = json.loads((tmp_path / "audit" / "round-000.json").read_text())
    ledger = (tmp_path / "audit" / "campaign.json").read_text()
    assert round_audit["promotion"]["status"] == "not-decided"
    assert "checkpoint_path" not in ledger
    assert len(runner.record.produced_snapshots) == 2
