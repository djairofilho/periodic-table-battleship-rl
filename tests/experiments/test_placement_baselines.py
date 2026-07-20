from __future__ import annotations

from periodic_table_battleship_rl.evaluation.schemas import RunConfig
from periodic_table_battleship_rl.experiments.placement_baselines import (
    run_placement_baseline_evaluation,
)
from periodic_table_battleship_rl.experiments.placement_evaluation import (
    PLACEMENT_ENVIRONMENT_VERSION,
)
from periodic_table_battleship_rl.placement import (
    RandomLegalPlacementPolicy,
    default_defensive_mixture,
)
from periodic_table_battleship_rl.topology import BATTLESHIP


def _config() -> RunConfig:
    return RunConfig(
        run_id="random-placement-test",
        experiment="placement",
        scenario="battleship",
        environment_version=PLACEMENT_ENVIRONMENT_VERSION,
        policy_id=RandomLegalPlacementPolicy.policy_id,
        split="test",
        seeds=(101, 103),
        episodes_per_seed=1,
    )


def test_baseline_evaluation_is_paired_between_attacker_components(tmp_path) -> None:
    mixture = default_defensive_mixture(BATTLESHIP)
    first = run_placement_baseline_evaluation(
        _config(),
        BATTLESHIP,
        RandomLegalPlacementPolicy(BATTLESHIP),
        mixture,
        tmp_path / "first",
        git_commit="a" * 40,
        uv_lock_path="uv.lock",
    )
    second = run_placement_baseline_evaluation(
        _config(),
        BATTLESHIP,
        RandomLegalPlacementPolicy(BATTLESHIP, seed=999),
        mixture,
        tmp_path / "second",
        git_commit="a" * 40,
        uv_lock_path="uv.lock",
    )

    first_actions = {
        (result.attacker_id, result.seed): result.placement_actions for result in first.results
    }
    second_actions = {
        (result.attacker_id, result.seed): result.placement_actions for result in second.results
    }
    assert first_actions == second_actions
    for seed in _config().seeds:
        paired = {
            result.placement_actions for result in first.results if result.seed == seed
        }
        assert len(paired) == 1
    assert first.manifest.config.parameters["placement_policy_kind"] == "independent-baseline-v1"
