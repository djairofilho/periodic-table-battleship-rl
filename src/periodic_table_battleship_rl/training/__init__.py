"""Optional training pipelines for benchmark policies.

Importing this package never imports PyTorch or Stable-Baselines3.  Install the
``train`` extra only when a training or checkpoint-loading function is used.
"""

from .attack import (
    ATTACK_POLICY_ID,
    AttackTrainingArtifact,
    AttackTrainingConfig,
    MaskableAttackPolicy,
    TrainingDependencyError,
    load_attack_policy,
    load_training_metadata,
    train_attack_policy,
)
from .placement import (
    PLACEMENT_POLICY_ID,
    PLACEMENT_TRAINING_SCHEMA_VERSION,
    MaskablePlacementPolicy,
    PlacementTrainingArtifact,
    PlacementTrainingConfig,
    PlacementTrainingDependencyError,
    load_placement_policy,
    load_placement_training_metadata,
    train_placement_policy,
)

__all__ = [
    "ATTACK_POLICY_ID",
    "AttackTrainingArtifact",
    "AttackTrainingConfig",
    "MaskableAttackPolicy",
    "MaskablePlacementPolicy",
    "PLACEMENT_POLICY_ID",
    "PLACEMENT_TRAINING_SCHEMA_VERSION",
    "PlacementTrainingArtifact",
    "PlacementTrainingConfig",
    "PlacementTrainingDependencyError",
    "TrainingDependencyError",
    "load_attack_policy",
    "load_placement_policy",
    "load_placement_training_metadata",
    "load_training_metadata",
    "train_attack_policy",
    "train_placement_policy",
]
