"""Pre-registered attack-PPO ablation definitions and seed-level analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from periodic_table_battleship_rl.analysis.campaign import (
    CampaignObservation,
    PolicyComparison,
    compare_policies,
)
from periodic_table_battleship_rl.envs import AttackEnvironmentConfig
from periodic_table_battleship_rl.evaluation import EpisodeResult


@dataclass(frozen=True, slots=True)
class AttackAblationArm:
    """One intervention with an explicit causal hypothesis."""

    arm_id: str
    hypothesis: str
    environment_config: AttackEnvironmentConfig

    def __post_init__(self) -> None:
        if not self.arm_id.strip():
            raise ValueError("arm_id must not be empty")
        if not self.hypothesis.strip():
            raise ValueError("hypothesis must not be empty")

    def public_dict(self) -> dict[str, object]:
        """Return auditable, JSON-native arm provenance."""
        return {
            "arm_id": self.arm_id,
            "hypothesis": self.hypothesis,
            "environment_config": self.environment_config.public_dict(),
        }


@dataclass(frozen=True, slots=True)
class AttackAblationSchedule:
    """Disjoint train/validation/test seed inventories for one ablation."""

    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    total_timesteps: int
    checkpoint_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        inventories = {
            "training_seeds": self.training_seeds,
            "validation_seeds": self.validation_seeds,
            "test_seeds": self.test_seeds,
        }
        for name, seeds in inventories.items():
            if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
                raise ValueError(f"{name} must contain unique non-negative seeds")
        if any(
            set(left) & set(right)
            for left, right in (
                (self.training_seeds, self.validation_seeds),
                (self.training_seeds, self.test_seeds),
                (self.validation_seeds, self.test_seeds),
            )
        ):
            raise ValueError("train, validation, and test seed inventories must be disjoint")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if (
            not self.checkpoint_steps
            or tuple(sorted(self.checkpoint_steps)) != self.checkpoint_steps
            or self.checkpoint_steps[-1] > self.total_timesteps
        ):
            raise ValueError("checkpoint_steps must be sorted and within total_timesteps")

    def public_dict(self) -> dict[str, object]:
        """Return a portable schedule ledger."""
        return {
            "training_seeds": list(self.training_seeds),
            "validation_seeds": list(self.validation_seeds),
            "test_seeds": list(self.test_seeds),
            "total_timesteps": self.total_timesteps,
            "checkpoint_steps": list(self.checkpoint_steps),
        }


def default_periodic_ablation_arms() -> tuple[AttackAblationArm, ...]:
    """Return the fixed one-factor-at-a-time arms for issue A9.

    The control is the v0.3 environment.  ``exploration-reward`` only changes
    the miss penalty; ``available-channel`` only adds the fifth public plane.
    """

    return (
        AttackAblationArm(
            arm_id="control-v03",
            hypothesis="Controle: reproduz recompensa e observação da v0.3.",
            environment_config=AttackEnvironmentConfig(),
        ),
        AttackAblationArm(
            arm_id="exploration-reward",
            hypothesis=(
                "Reduzir a penalidade de erro de -1,0 para -0,2 permite ao PPO "
                "explorar e converter informação local de acertos em menos tiros."
            ),
            environment_config=AttackEnvironmentConfig(reward_profile="exploration-v1"),
        ),
        AttackAblationArm(
            arm_id="available-channel",
            hypothesis=(
                "Expor o plano binário de ações ainda disponíveis melhora a estimativa "
                "de valor sem revelar a frota e reduz tiros válidos."
            ),
            environment_config=AttackEnvironmentConfig(
                observation_profile="outcomes-plus-available-v1"
            ),
        ),
    )


def compare_ablation_arms(
    *,
    scenario: str,
    results_by_arm: dict[str, Sequence[EpisodeResult]],
    reference_arm_id: str = "control-v03",
    resamples: int = 10_000,
) -> tuple[PolicyComparison, ...]:
    """Return blind-seed paired comparisons against the frozen control arm."""

    if reference_arm_id not in results_by_arm:
        raise ValueError("reference_arm_id must have results")
    observations = tuple(
        CampaignObservation(
            episode_id=result.episode_id,
            policy_id=arm_id,
            seed=result.seed,
            scenario=scenario,
            metric="valid_shots",
            value=float(result.valid_shots),
        )
        for arm_id, results in sorted(results_by_arm.items())
        for result in results
    )
    comparisons: list[PolicyComparison] = []
    for arm_id in sorted(results_by_arm):
        if arm_id == reference_arm_id:
            continue
        comparisons.extend(
            compare_policies(
                observations,
                experiment="attack-ablation",
                candidate_policy=arm_id,
                reference_policies=(reference_arm_id,),
                direction="lower",
                resamples=resamples,
            )
        )
    return tuple(comparisons)
