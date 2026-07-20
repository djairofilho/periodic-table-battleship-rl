"""Minimal auditable attacker-versus-placer self-play wiring.

The benchmark environments remain unchanged by default: :class:`AttackEnv`
continues to sample random fleets.  This module makes the alternative explicit
by adapting a frozen placement policy into a private fleet sampler and a
frozen attack policy into a placement evaluator.  In both directions, learner
policies receive only observations and legal action masks.

``CoupledSelfPlayRunner`` intentionally records frozen-suite scores beside
every league result but never promotes a snapshot.  Promotion remains an
experiment-level decision after validation and a separate blinded test.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from periodic_table_battleship_rl.envs.attack import (
    AttackEnvironmentConfig,
    AttackEnv,
    FleetFactory,
)
from periodic_table_battleship_rl.envs.placement import PlacementEnv
from periodic_table_battleship_rl.evaluation.schemas import sha256_file
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.game import Fleet
from periodic_table_battleship_rl.placement.defensive import DefensiveEvaluator
from periodic_table_battleship_rl.selfplay.league import (
    SelfPlayCampaignRecord,
    SelfPlayRoundPlan,
    SnapshotProvenance,
    persist_self_play_campaign,
)
from periodic_table_battleship_rl.topology import Topology
from periodic_table_battleship_rl.training.attack import ATTACK_POLICY_ID
from periodic_table_battleship_rl.training.placement import PLACEMENT_POLICY_ID


@runtime_checkable
class PublicActionPolicy(Protocol):
    """Choose an action from public observation and legal-mask inputs only."""

    policy_id: str

    def select_action(
        self, observation: Any, action_mask: Any, *, deterministic: bool = True
    ) -> int:
        """Return one currently legal action."""


@runtime_checkable
class FleetSampler(Protocol):
    """Sample a legal hidden fleet for an environment-owned episode."""

    sampler_id: str

    def sample_fleet(self, topology: Topology, *, rng: np.random.Generator) -> Fleet:
        """Return a legal fleet on ``topology``."""


@dataclass(frozen=True, slots=True)
class PlacementPolicyFleetSampler:
    """Adapt a frozen public placement policy into a hidden fleet sampler."""

    policy: PublicActionPolicy
    sampler_id: str

    def sample_fleet(self, topology: Topology, *, rng: np.random.Generator) -> Fleet:
        """Roll out a placement policy without exposing its fleet to an attacker."""

        episode_seed = int(rng.integers(2**32, dtype=np.uint32))
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(seed=episode_seed)
        environment = PlacementEnv(topology, evaluator=_TerminalFleetEvaluator())
        observation, _ = environment.reset(seed=episode_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = self.policy.select_action(
                observation, environment.action_masks(), deterministic=True
            )
            observation, _, terminated, truncated, _ = environment.step(action)
        if truncated or environment.fleet is None:
            raise RuntimeError("placement policy did not produce a complete legal fleet")
        return environment.fleet


@dataclass(frozen=True, slots=True)
class PublicAttackPolicyEvaluator:
    """Adapt a frozen attacker to the evaluator protocol using public state."""

    policy: PublicActionPolicy
    topology: Topology
    evaluator_id: str
    environment_config: AttackEnvironmentConfig | None = None

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        """Sink ``fleet`` while the policy sees only state and legal actions."""

        if not fleet.occupied_cells.issubset(self.topology.valid_cells):
            raise ValueError("fleet must stay within the evaluator topology")

        def fleet_factory(topology: Topology, _: np.random.Generator) -> Fleet:
            if topology != self.topology:
                raise ValueError("fleet factory topology does not match evaluator")
            return fleet

        environment = AttackEnv(
            self.topology,
            config=self.environment_config,
            fleet_factory=fleet_factory,
        )
        observation, _ = environment.reset(
            seed=int(rng.integers(2**32, dtype=np.uint32))
        )
        terminated = truncated = False
        info: Mapping[str, int | bool] = {}
        while not (terminated or truncated):
            action = self.policy.select_action(
                observation, environment.action_masks(), deterministic=True
            )
            observation, _, terminated, truncated, info = environment.step(action)
        if truncated:
            raise RuntimeError("attack policy did not finish the fleet")
        return int(info["valid_shots"])


class CoupledAttackEnv(AttackEnv):
    """Attack environment that draws its hidden fleet from a frozen placer."""

    def __init__(
        self,
        topology: Topology,
        placer: FleetSampler,
        *,
        config: AttackEnvironmentConfig | None = None,
    ) -> None:
        self.placer_id = placer.sampler_id

        def fleet_factory(factory_topology: Topology, rng: np.random.Generator) -> Fleet:
            return placer.sample_fleet(factory_topology, rng=rng)

        super().__init__(topology, config=config, fleet_factory=fleet_factory)


@dataclass(frozen=True, slots=True)
class CoupledTrainingOutput:
    """Local result returned by a role-specific trainer before ledger persistence."""

    checkpoint_path: Path
    source_run_id: str
    runtime_opponent: FleetSampler | DefensiveEvaluator
    policy_id: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("self-play trainer checkpoint does not exist")
        if not self.source_run_id.strip():
            raise ValueError("source_run_id must not be empty")


@runtime_checkable
class CoupledTrainer(Protocol):
    """Role-specific training implementation supplied by an experiment runner."""

    def train_attacker(
        self, plan: SelfPlayRoundPlan, environment: CoupledAttackEnv
    ) -> CoupledTrainingOutput:
        """Train one attacker against the selected frozen placement sampler."""

    def train_placer(
        self, plan: SelfPlayRoundPlan, evaluator: DefensiveEvaluator
    ) -> CoupledTrainingOutput:
        """Train one placer against the selected frozen public attacker."""


@runtime_checkable
class FrozenSuiteEvaluator(Protocol):
    """Evaluate a candidate against every fixed target after each round."""

    def evaluate(
        self,
        *,
        role: str,
        runtime_opponent: FleetSampler | DefensiveEvaluator,
        target_ids: tuple[str, ...],
    ) -> Mapping[str, float]:
        """Return one numeric metric for every requested immutable target."""


@dataclass(slots=True)
class CoupledSelfPlayRunner:
    """Execute a pre-registered alternating campaign with an append-only ledger.

    ``runtime_opponents`` is deliberately local and never written into the
    campaign JSON.  Snapshot hashes and run identifiers provide the portable
    identity, while this mapping is the explicit machine-local bridge needed
    to resume a training job.
    """

    record: SelfPlayCampaignRecord
    topology: Topology
    trainer: CoupledTrainer
    frozen_suite: FrozenSuiteEvaluator
    runtime_opponents: dict[str, FleetSampler | DefensiveEvaluator]
    output_directory: Path

    def __post_init__(self) -> None:
        if self.record.config.scenario != self.topology.name:
            raise ValueError("campaign scenario must match runner topology")
        self.output_directory = Path(self.output_directory)

    def run_next_round(self) -> SnapshotProvenance | None:
        """Execute, audit and append one scheduled round, or return ``None``.

        The method evaluates every candidate on its frozen targets before the
        ledger is persisted.  It records those observations only; deciding
        whether a candidate is promotable is intentionally outside this API.
        """

        plan = self.record.next_round
        if plan is None:
            return None
        opponent = self.runtime_opponents.get(plan.opponent_snapshot_id)
        if opponent is None:
            raise KeyError(f"missing local runtime for {plan.opponent_snapshot_id!r}")
        output = self._train(plan, opponent)
        snapshot = SnapshotProvenance(
            snapshot_id=(
                f"{self.record.config.campaign_id}-{plan.learner_role}"
                f"-round-{plan.round_index:03d}"
            ),
            role=plan.learner_role,
            policy_id=(
                output.policy_id
                or (
                    ATTACK_POLICY_ID
                    if plan.learner_role == "attacker"
                    else PLACEMENT_POLICY_ID
                )
            ),
            scenario=self.topology.name,
            source_run_id=output.source_run_id,
            checkpoint_sha256=sha256_file(output.checkpoint_path),
            training_round=plan.round_index,
            parent_snapshot_ids=(plan.opponent_snapshot_id,),
        )
        frozen_scores = self.frozen_suite.evaluate(
            role=plan.learner_role,
            runtime_opponent=output.runtime_opponent,
            target_ids=plan.frozen_evaluation_target_ids,
        )
        self._validate_frozen_scores(plan, frozen_scores)
        self.record = self.record.record_snapshot(snapshot)
        self.runtime_opponents[snapshot.snapshot_id] = output.runtime_opponent
        self.output_directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.output_directory / f"round-{plan.round_index:03d}.json",
            {
                "plan": plan.to_dict(),
                "snapshot": snapshot.to_dict(),
                "frozen_evaluation": dict(sorted(frozen_scores.items())),
                "promotion": {
                    "status": "not-decided",
                    "reason": "frozen-suite metrics are evidence, not promotion",
                },
            },
        )
        persist_self_play_campaign(self.output_directory / "campaign.json", self.record)
        return snapshot

    def _train(
        self,
        plan: SelfPlayRoundPlan,
        opponent: FleetSampler | DefensiveEvaluator,
    ) -> CoupledTrainingOutput:
        if plan.learner_role == "attacker":
            if not isinstance(opponent, FleetSampler):
                raise TypeError("attacker updates require a FleetSampler opponent")
            return self.trainer.train_attacker(
                plan, CoupledAttackEnv(self.topology, opponent)
            )
        if not isinstance(opponent, DefensiveEvaluator):
            raise TypeError("placer updates require a DefensiveEvaluator opponent")
        return self.trainer.train_placer(plan, opponent)

    @staticmethod
    def _validate_frozen_scores(
        plan: SelfPlayRoundPlan, scores: Mapping[str, float]
    ) -> None:
        if set(scores) != set(plan.frozen_evaluation_target_ids):
            raise ValueError("frozen evaluation must report exactly every target")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            for value in scores.values()
        ):
            raise ValueError("frozen evaluation scores must be finite numeric values")


@dataclass(frozen=True, slots=True)
class _TerminalFleetEvaluator:
    """Allow a placement rollout to terminate after emitting its legal fleet."""

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        del fleet, rng
        return 1


def fleet_factory_from_sampler(sampler: FleetSampler) -> FleetFactory:
    """Expose a sampler as the explicit factory accepted by :class:`AttackEnv`."""

    return lambda topology, rng: sampler.sample_fleet(topology, rng=rng)
