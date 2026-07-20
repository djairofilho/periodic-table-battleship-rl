from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from periodic_table_battleship_rl.envs.placement import (
    PLACEMENT_ACTION_COUNT,
    PlacementEnv,
    PlacementEvaluator,
)
from periodic_table_battleship_rl.game import CANONICAL_FLEET, Fleet, is_legal_fleet


@dataclass
class FakeEvaluator:
    """Deterministic evaluator which also proves the final fleet is passed in."""

    shots: int = 41
    fleets: list[Fleet] = field(default_factory=list)

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        del rng
        self.fleets.append(fleet)
        return self.shots


class SeededFakeEvaluator:
    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        del fleet
        return int(rng.integers(20, 51))


def _first_legal_action(env: PlacementEnv) -> int:
    return int(np.flatnonzero(env.action_masks())[0])


def _complete_episode(env: PlacementEnv) -> tuple[list[int], float, dict[str, object]]:
    actions: list[int] = []
    info: dict[str, object] = {}
    reward = 0.0
    for _ in range(5):
        action = _first_legal_action(env)
        actions.append(action)
        _, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    return actions, reward, info


def test_contract_has_fixed_action_space_and_float_observation() -> None:
    env = PlacementEnv("battleship", FakeEvaluator())

    observation, info = env.reset(seed=4)

    assert env.action_space.n == PLACEMENT_ACTION_COUNT == 360
    assert observation.dtype == np.float32
    assert observation.shape == (3, 10, 18)
    assert env.observation_space.contains(observation)
    assert observation[0].sum() == 100
    assert observation[1].sum() == 0
    assert np.all(observation[2][observation[0] == 1] == 1.0)
    assert info["placements_completed"] == 0


def test_mask_contains_only_complete_non_overlapping_placements() -> None:
    env = PlacementEnv("periodic-table-battleship", FakeEvaluator())
    env.reset(seed=2)

    for _ in range(5):
        mask = env.action_masks()
        assert mask.dtype == np.bool_
        assert mask.shape == (360,)
        next_length = CANONICAL_FLEET[len(env.placement_actions)].length
        expected = np.array(
            [
                env._cells_for_action(action, next_length) is not None
                for action in range(PLACEMENT_ACTION_COUNT)
            ],
            dtype=np.bool_,
        )
        np.testing.assert_array_equal(mask, expected)
        action = _first_legal_action(env)
        _, _, terminated, _, _ = env.step(action)
        assert not terminated or not env.action_masks().any()


def test_valid_placements_build_a_legal_complete_fleet_and_terminal_reward() -> None:
    evaluator = FakeEvaluator(shots=73)
    env = PlacementEnv("periodic-table-battleship", evaluator)
    env.reset(seed=8)

    actions, reward, info = _complete_episode(env)

    assert len(actions) == 5
    assert len(evaluator.fleets) == 1
    assert env.fleet == evaluator.fleets[0]
    assert is_legal_fleet(env.topology, env.fleet)
    assert reward == pytest.approx(73 / 118)
    assert info["valid_shots_to_sink"] == 73
    assert env.action_masks().sum() == 0


def test_invalid_action_is_penalized_without_changing_the_partial_fleet() -> None:
    env = PlacementEnv("battleship", FakeEvaluator())
    initial, _ = env.reset(seed=11)

    observation, reward, terminated, truncated, info = env.step(100)

    assert reward == -1.0
    assert not terminated and not truncated
    assert info["invalid_action"] is True
    assert info["placements_completed"] == 0
    np.testing.assert_array_equal(observation, initial)


def test_same_seed_and_deterministic_actions_produce_same_result() -> None:
    first = PlacementEnv("battleship", FakeEvaluator(shots=29))
    second = PlacementEnv("battleship", FakeEvaluator(shots=29))
    first.reset(seed=839)
    second.reset(seed=839)

    first_actions, first_reward, first_info = _complete_episode(first)
    second_actions, second_reward, second_info = _complete_episode(second)

    assert first_actions == second_actions
    assert first.fleet == second.fleet
    assert first_reward == second_reward
    assert first_info == second_info


def test_reset_seed_is_forwarded_to_a_randomized_evaluator() -> None:
    first = PlacementEnv("battleship", SeededFakeEvaluator())
    second = PlacementEnv("battleship", SeededFakeEvaluator())
    first.reset(seed=91)
    second.reset(seed=91)

    _, first_reward, first_info = _complete_episode(first)
    _, second_reward, second_info = _complete_episode(second)

    assert first_reward == second_reward
    assert first_info["valid_shots_to_sink"] == second_info["valid_shots_to_sink"]


def test_evaluator_contract_is_public_and_invalid_results_are_rejected() -> None:
    evaluator = FakeEvaluator(shots=0)
    assert isinstance(evaluator, PlacementEvaluator)
    env = PlacementEnv("battleship", evaluator)
    env.reset(seed=5)

    with pytest.raises(ValueError, match="within"):
        _complete_episode(env)


def test_gymnasium_checker_accepts_environment_contract() -> None:
    check_env(PlacementEnv("battleship", FakeEvaluator()), skip_render_check=True)
