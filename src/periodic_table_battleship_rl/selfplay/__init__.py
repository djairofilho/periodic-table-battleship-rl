"""Reproducible contracts for alternating self-play experiments.

The package deliberately contains no training loop.  It defines the durable
league, schedule, and provenance contracts that a future attacker-versus-
placer environment can consume without weakening the fixed-baseline
evaluation protocol.
"""

from .league import (
    SELF_PLAY_SCHEMA_VERSION,
    AlternatingSelfPlaySchedule,
    FrozenEvaluationSuite,
    SelfPlayCampaignConfig,
    SelfPlayCampaignRecord,
    SelfPlayRoundPlan,
    SnapshotLeague,
    SnapshotProvenance,
    persist_self_play_campaign,
)
from .coupled import (
    CoupledAttackEnv,
    CoupledSelfPlayRunner,
    CoupledTrainingOutput,
    FleetSampler,
    FrozenSuiteEvaluator,
    PlacementPolicyFleetSampler,
    PublicActionPolicy,
    PublicAttackPolicyEvaluator,
)

__all__ = [
    "SELF_PLAY_SCHEMA_VERSION",
    "AlternatingSelfPlaySchedule",
    "FrozenEvaluationSuite",
    "SelfPlayCampaignConfig",
    "SelfPlayCampaignRecord",
    "SelfPlayRoundPlan",
    "SnapshotLeague",
    "SnapshotProvenance",
    "persist_self_play_campaign",
    "CoupledAttackEnv",
    "CoupledSelfPlayRunner",
    "CoupledTrainingOutput",
    "FleetSampler",
    "FrozenSuiteEvaluator",
    "PlacementPolicyFleetSampler",
    "PublicActionPolicy",
    "PublicAttackPolicyEvaluator",
]
