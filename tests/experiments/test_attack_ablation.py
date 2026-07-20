from __future__ import annotations

import pytest

from periodic_table_battleship_rl.evaluation import EpisodeResult
from periodic_table_battleship_rl.experiments.attack_ablation import (
    AttackAblationSchedule,
    compare_ablation_arms,
    default_periodic_ablation_arms,
)


def _result(arm: str, seed: int, shots: int) -> EpisodeResult:
    return EpisodeResult(
        episode_id=f"{arm}-{seed}",
        run_id=arm,
        seed=seed,
        scenario="periodic-table-battleship",
        valid_cells=118,
        valid_shots=shots,
        invalid_attempts=0,
        hit_segments=17,
        sunk_ship_lengths=(2, 3, 3, 4, 5),
        won=True,
        truncated=False,
        auc_discovery=0.5,
    )


def test_default_arms_change_one_factor_at_a_time() -> None:
    control, reward, observation = default_periodic_ablation_arms()

    assert control.environment_config.public_dict() == {
        "observation_profile": "outcomes-v1",
        "reward_profile": "hit-miss-terminal-v1",
    }
    assert reward.environment_config.observation_profile == control.environment_config.observation_profile
    assert observation.environment_config.reward_profile == control.environment_config.reward_profile


def test_schedule_requires_disjoint_seed_sets() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        AttackAblationSchedule(
            training_seeds=(1,),
            validation_seeds=(1,),
            test_seeds=(2,),
            total_timesteps=10,
            checkpoint_steps=(10,),
        )


def test_comparison_is_paired_by_held_out_seed() -> None:
    comparisons = compare_ablation_arms(
        scenario="periodic-table-battleship",
        results_by_arm={
            "control-v03": (_result("control", 1, 100), _result("control", 2, 102)),
            "exploration-reward": (_result("reward", 1, 90), _result("reward", 2, 92)),
        },
        resamples=100,
    )

    assert len(comparisons) == 1
    assert comparisons[0].candidate_minus_reference_mean == -10.0
    assert comparisons[0].conclusion == "candidate_favored"
