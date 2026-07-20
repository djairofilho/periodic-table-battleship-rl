"""Versioned league and alternating-schedule contracts for self-play.

This module intentionally does *not* claim to train an attacker against a
placer yet.  The current environments train attack against random fleets and
placement against defensive evaluators.  Instead, it gives a future coupled
environment a reproducible public contract: every update has a role, seed,
opponent snapshot, fixed benchmark targets, and a resulting checkpoint hash.

Snapshots never store local checkpoint paths.  A checkpoint is identified by
its portable digest and the source run that created it.  This makes a campaign
ledger safe to persist, move, and compare between machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np

from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.training.attack import ATTACK_POLICY_ID
from periodic_table_battleship_rl.training.placement import PLACEMENT_POLICY_ID


SELF_PLAY_SCHEMA_VERSION = "self-play-v1"
"""Schema for portable self-play campaign provenance."""

SelfPlayRole = Literal["attacker", "placer"]
"""The two alternating learners in a self-play campaign."""

BELIEF_ATTACK_POLICY_IDS = frozenset(
    {
        "belief_probability_mc-v1",
        "belief_information_mc-v1",
        "belief_horizon2_mc-v1",
    }
)
"""Public-history planners that can be represented as frozen attackers.

The league originally accepted only learned MaskablePPO attackers.  A Bayesian
planner is a legitimate, reproducible frozen opponent too, and preserving its
own policy ID is necessary to avoid falsely claiming that a snapshot is PPO.
"""


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_role(value: str, name: str = "role") -> None:
    if value not in ("attacker", "placer"):
        raise ValueError(f"{name} must be 'attacker' or 'placer'")


def _opponent_role(role: SelfPlayRole) -> SelfPlayRole:
    return "placer" if role == "attacker" else "attacker"


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotProvenance:
    """Portable identity of one frozen learner checkpoint in the league."""

    snapshot_id: str
    role: SelfPlayRole
    policy_id: str
    scenario: str
    source_run_id: str
    checkpoint_sha256: str
    training_round: int
    parent_snapshot_ids: tuple[str, ...] = ()
    schema_version: str = SELF_PLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "policy_id", "scenario", "source_run_id"):
            _require_identifier(getattr(self, name), name)
        _require_role(self.role)
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if self.training_round < 0:
            raise ValueError("training_round must be non-negative")
        if self.schema_version != SELF_PLAY_SCHEMA_VERSION:
            raise ValueError("unsupported self-play snapshot schema version")
        if len(set(self.parent_snapshot_ids)) != len(self.parent_snapshot_ids):
            raise ValueError("parent_snapshot_ids must not contain duplicates")
        if any(not parent.strip() for parent in self.parent_snapshot_ids):
            raise ValueError("parent_snapshot_ids must not contain empty identifiers")
        if self.snapshot_id in self.parent_snapshot_ids:
            raise ValueError("a snapshot cannot list itself as a parent")

        permitted_policies = (
            frozenset({ATTACK_POLICY_ID}).union(BELIEF_ATTACK_POLICY_IDS)
            if self.role == "attacker"
            else frozenset({PLACEMENT_POLICY_ID})
        )
        if self.policy_id not in permitted_policies:
            raise ValueError(
                f"unsupported policy_id {self.policy_id!r}; {self.role} snapshots "
                "must use a registered policy identity"
            )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native portable checkpoint provenance."""

        return {
            "snapshot_id": self.snapshot_id,
            "role": self.role,
            "policy_id": self.policy_id,
            "scenario": self.scenario,
            "source_run_id": self.source_run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_round": self.training_round,
            "parent_snapshot_ids": list(self.parent_snapshot_ids),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenEvaluationSuite:
    """Immutable fixed targets used beside the changing league opponent.

    An attacker is evaluated against frozen placement-policy baseline IDs.  A
    placer is evaluated against frozen attacker-evaluator IDs.  The IDs are
    intentionally opaque because concrete evaluation adapters remain owned by
    the existing attack and placement benchmark modules.
    """

    attacker_evaluator_ids: tuple[str, ...]
    placement_policy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("attacker_evaluator_ids", "placement_policy_ids"):
            identifiers = getattr(self, name)
            if not identifiers:
                raise ValueError(f"{name} must not be empty")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{name} must not contain duplicates")
            if any(not isinstance(item, str) or not item.strip() for item in identifiers):
                raise ValueError(f"{name} must contain non-empty strings")

    def targets_for(self, role: SelfPlayRole) -> tuple[str, ...]:
        """Return frozen benchmark targets for the learner in ``role``."""

        _require_role(role)
        return (
            self.placement_policy_ids
            if role == "attacker"
            else self.attacker_evaluator_ids
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return stable, JSON-native benchmark-target provenance."""

        return {
            "attacker_evaluator_ids": list(self.attacker_evaluator_ids),
            "placement_policy_ids": list(self.placement_policy_ids),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotLeague:
    """Immutable per-scenario pool of frozen attacker and placer snapshots."""

    scenario: str
    snapshots: tuple[SnapshotProvenance, ...]
    sampling_policy: Literal["uniform-v1"] = "uniform-v1"

    def __post_init__(self) -> None:
        _require_identifier(self.scenario, "scenario")
        if self.sampling_policy != "uniform-v1":
            raise ValueError("unsupported self-play league sampling policy")
        identifiers = tuple(snapshot.snapshot_id for snapshot in self.snapshots)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("league snapshot_ids must be unique")
        if any(snapshot.scenario != self.scenario for snapshot in self.snapshots):
            raise ValueError("all league snapshots must match the league scenario")
        known_ids = set(identifiers)
        if any(
            parent not in known_ids
            for snapshot in self.snapshots
            for parent in snapshot.parent_snapshot_ids
        ):
            raise ValueError("snapshot parents must already be present in the league")

    def snapshots_for(self, role: SelfPlayRole) -> tuple[SnapshotProvenance, ...]:
        """Return snapshots of one role ordered by their stable identifier."""

        _require_role(role)
        return tuple(
            sorted(
                (snapshot for snapshot in self.snapshots if snapshot.role == role),
                key=lambda snapshot: snapshot.snapshot_id,
            )
        )

    def select_opponent(
        self, learner_role: SelfPlayRole, *, seed: int
    ) -> SnapshotProvenance:
        """Select one opposite-role snapshot with a recorded portable seed."""

        _require_role(learner_role, "learner_role")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        candidates = self.snapshots_for(_opponent_role(learner_role))
        if not candidates:
            raise ValueError(
                f"league has no {_opponent_role(learner_role)} snapshot for {learner_role}"
            )
        rng = np.random.default_rng(seed)
        return candidates[int(rng.integers(len(candidates)))]

    def with_snapshot(self, snapshot: SnapshotProvenance) -> SnapshotLeague:
        """Return a new league after accepting one compatible frozen snapshot."""

        if snapshot.scenario != self.scenario:
            raise ValueError("snapshot scenario must match the league scenario")
        if snapshot.snapshot_id in {item.snapshot_id for item in self.snapshots}:
            raise ValueError("league already contains this snapshot_id")
        missing_parents = set(snapshot.parent_snapshot_ids) - {
            item.snapshot_id for item in self.snapshots
        }
        if missing_parents:
            raise ValueError("snapshot parents must exist before adding the snapshot")
        return replace(self, snapshots=(*self.snapshots, snapshot))

    def to_dict(self) -> dict[str, object]:
        """Return sorted snapshots for portable and diff-friendly provenance."""

        return {
            "scenario": self.scenario,
            "sampling_policy": self.sampling_policy,
            "snapshots": [
                snapshot.to_dict()
                for snapshot in sorted(self.snapshots, key=lambda item: item.snapshot_id)
            ],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SelfPlayCampaignConfig:
    """Fixed update calendar, seeds, and benchmarks for an alternating run."""

    campaign_id: str
    scenario: str
    seed: int
    round_count: int
    attacker_timesteps: int
    placer_timesteps: int
    frozen_evaluation: FrozenEvaluationSuite
    first_learner: SelfPlayRole = "attacker"
    schema_version: str = SELF_PLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("campaign_id", "scenario"):
            _require_identifier(getattr(self, name), name)
        _require_role(self.first_learner, "first_learner")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("round_count", "attacker_timesteps", "placer_timesteps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.schema_version != SELF_PLAY_SCHEMA_VERSION:
            raise ValueError("unsupported self-play campaign schema version")

    def timesteps_for(self, role: SelfPlayRole) -> int:
        """Return the fixed budget for an attacker or placer update."""

        _require_role(role)
        return self.attacker_timesteps if role == "attacker" else self.placer_timesteps

    def to_dict(self) -> dict[str, object]:
        """Return the complete public calendar configuration."""

        return {
            "campaign_id": self.campaign_id,
            "scenario": self.scenario,
            "seed": self.seed,
            "round_count": self.round_count,
            "attacker_timesteps": self.attacker_timesteps,
            "placer_timesteps": self.placer_timesteps,
            "frozen_evaluation": self.frozen_evaluation.to_dict(),
            "first_learner": self.first_learner,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SelfPlayRoundPlan:
    """One executable alternating update planned from public campaign state."""

    round_index: int
    learner_role: SelfPlayRole
    training_seed: int
    opponent_selection_seed: int
    opponent_snapshot_id: str
    opponent_role: SelfPlayRole
    timesteps: int
    frozen_evaluation_target_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        _require_role(self.learner_role, "learner_role")
        _require_role(self.opponent_role, "opponent_role")
        if self.opponent_role != _opponent_role(self.learner_role):
            raise ValueError("opponent_role must be the opposite of learner_role")
        for name in ("training_seed", "opponent_selection_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _require_identifier(self.opponent_snapshot_id, "opponent_snapshot_id")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not self.frozen_evaluation_target_ids:
            raise ValueError("frozen_evaluation_target_ids must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return one public update record suitable for a campaign ledger."""

        return {
            "round_index": self.round_index,
            "learner_role": self.learner_role,
            "training_seed": self.training_seed,
            "opponent_selection_seed": self.opponent_selection_seed,
            "opponent_snapshot_id": self.opponent_snapshot_id,
            "opponent_role": self.opponent_role,
            "timesteps": self.timesteps,
            "frozen_evaluation_target_ids": list(self.frozen_evaluation_target_ids),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AlternatingSelfPlaySchedule:
    """Plan reproducible updates while snapshots are appended to a league."""

    config: SelfPlayCampaignConfig

    def plan_round(self, league: SnapshotLeague, *, round_index: int) -> SelfPlayRoundPlan:
        """Plan one round from the current pool without mutating it.

        A caller must add the newly trained snapshot to ``league`` before
        planning the next round.  That explicit transition prevents a trainer
        from silently changing historical opponent choices.
        """

        if league.scenario != self.config.scenario:
            raise ValueError("league scenario must match the self-play campaign")
        if not 0 <= round_index < self.config.round_count:
            raise ValueError("round_index must be within the configured campaign")
        learner_role = self._learner_role(round_index)
        training_seed, selection_seed = self._round_seeds(round_index)
        opponent = league.select_opponent(learner_role, seed=selection_seed)
        return SelfPlayRoundPlan(
            round_index=round_index,
            learner_role=learner_role,
            training_seed=training_seed,
            opponent_selection_seed=selection_seed,
            opponent_snapshot_id=opponent.snapshot_id,
            opponent_role=opponent.role,
            timesteps=self.config.timesteps_for(learner_role),
            frozen_evaluation_target_ids=self.config.frozen_evaluation.targets_for(
                learner_role
            ),
        )

    def _learner_role(self, round_index: int) -> SelfPlayRole:
        if round_index % 2 == 0:
            return self.config.first_learner
        return _opponent_role(self.config.first_learner)

    def _round_seeds(self, round_index: int) -> tuple[int, int]:
        generated = np.random.SeedSequence((self.config.seed, round_index)).generate_state(2)
        return int(generated[0]), int(generated[1])


@dataclass(frozen=True, slots=True, kw_only=True)
class SelfPlayCampaignRecord:
    """Append-only public ledger for completed or partially completed rounds."""

    config: SelfPlayCampaignConfig
    initial_league: SnapshotLeague
    completed_rounds: tuple[SelfPlayRoundPlan, ...] = ()
    produced_snapshots: tuple[SnapshotProvenance, ...] = ()
    schema_version: str = SELF_PLAY_SCHEMA_VERSION
    _schedule: AlternatingSelfPlaySchedule = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != SELF_PLAY_SCHEMA_VERSION:
            raise ValueError("unsupported self-play record schema version")
        if self.initial_league.scenario != self.config.scenario:
            raise ValueError("initial league scenario must match campaign scenario")
        if len(self.completed_rounds) != len(self.produced_snapshots):
            raise ValueError("every completed round requires one produced snapshot")
        if len(self.completed_rounds) > self.config.round_count:
            raise ValueError("completed rounds cannot exceed round_count")
        object.__setattr__(
            self, "_schedule", AlternatingSelfPlaySchedule(config=self.config)
        )

        evolving_league = self.initial_league
        for expected_index, (round_plan, snapshot) in enumerate(
            zip(self.completed_rounds, self.produced_snapshots, strict=True)
        ):
            expected = self._schedule.plan_round(evolving_league, round_index=expected_index)
            if round_plan != expected:
                raise ValueError("completed round does not match the deterministic schedule")
            if snapshot.role != round_plan.learner_role:
                raise ValueError("produced snapshot role must match the round learner")
            if snapshot.training_round != round_plan.round_index:
                raise ValueError("produced snapshot training_round must match round_index")
            if snapshot.scenario != self.config.scenario:
                raise ValueError("produced snapshot scenario must match campaign scenario")
            if round_plan.opponent_snapshot_id not in snapshot.parent_snapshot_ids:
                raise ValueError("produced snapshot must record its selected opponent parent")
            evolving_league = evolving_league.with_snapshot(snapshot)

    @property
    def current_league(self) -> SnapshotLeague:
        """Return the initial league plus every accepted produced snapshot."""

        league = self.initial_league
        for snapshot in self.produced_snapshots:
            league = league.with_snapshot(snapshot)
        return league

    @property
    def next_round(self) -> SelfPlayRoundPlan | None:
        """Return the next deterministic update, or ``None`` at completion."""

        if len(self.completed_rounds) == self.config.round_count:
            return None
        return self._schedule.plan_round(
            self.current_league, round_index=len(self.completed_rounds)
        )

    def record_snapshot(self, snapshot: SnapshotProvenance) -> SelfPlayCampaignRecord:
        """Append the snapshot resulting from ``next_round`` immutably."""

        next_round = self.next_round
        if next_round is None:
            raise ValueError("self-play campaign has no remaining rounds")
        if snapshot.role != next_round.learner_role:
            raise ValueError("snapshot role must match the next round learner")
        if snapshot.training_round != next_round.round_index:
            raise ValueError("snapshot training_round must match the next round")
        if next_round.opponent_snapshot_id not in snapshot.parent_snapshot_ids:
            raise ValueError("snapshot must record the selected opponent as a parent")
        return replace(
            self,
            completed_rounds=(*self.completed_rounds, next_round),
            produced_snapshots=(*self.produced_snapshots, snapshot),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the full portable campaign ledger without local paths."""

        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "initial_league": self.initial_league.to_dict(),
            "completed_rounds": [round_plan.to_dict() for round_plan in self.completed_rounds],
            "produced_snapshots": [
                snapshot.to_dict() for snapshot in self.produced_snapshots
            ],
            "current_league": self.current_league.to_dict(),
            "next_round": None if self.next_round is None else self.next_round.to_dict(),
        }


def persist_self_play_campaign(
    path: str | Path, record: SelfPlayCampaignRecord
) -> Path:
    """Atomically persist a portable, append-only self-play ledger."""

    return write_json_atomic(path, record.to_dict())
