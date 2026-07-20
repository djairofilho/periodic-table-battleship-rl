"""Optional training pipelines for benchmark policies.

Importing this package never imports PyTorch or Stable-Baselines3.  Install the
``train`` extra only when a training or checkpoint-loading function is used.
"""

from .attack import (
    ATTACK_POLICY_ID,
    VALIDATION_CURVE_SCHEMA_VERSION,
    AttackCheckpointArtifact,
    AttackTrainingArtifact,
    AttackTrainingConfig,
    AttackValidationConfig,
    AttackValidationResult,
    MaskableAttackPolicy,
    TrainingDependencyError,
    load_attack_policy,
    load_training_metadata,
    evaluate_attack_validation,
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
    "VALIDATION_CURVE_SCHEMA_VERSION",
    "AttackCheckpointArtifact",
    "AttackTrainingArtifact",
    "AttackTrainingConfig",
    "AttackValidationConfig",
    "AttackValidationResult",
    "MaskableAttackPolicy",
    "MaskablePlacementPolicy",
    "PLACEMENT_POLICY_ID",
    "PLACEMENT_TRAINING_SCHEMA_VERSION",
    "PlacementTrainingArtifact",
    "PlacementTrainingConfig",
    "PlacementTrainingDependencyError",
    "TrainingDependencyError",
    "load_attack_policy",
    "evaluate_attack_validation",
    "load_placement_policy",
    "load_placement_training_metadata",
    "load_training_metadata",
    "train_attack_policy",
    "train_placement_policy",
]
