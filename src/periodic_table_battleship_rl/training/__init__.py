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

__all__ = [
    "ATTACK_POLICY_ID",
    "AttackTrainingArtifact",
    "AttackTrainingConfig",
    "MaskableAttackPolicy",
    "TrainingDependencyError",
    "load_attack_policy",
    "load_training_metadata",
    "train_attack_policy",
]
