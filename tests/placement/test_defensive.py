"""Tests for the fixed defensive attacker suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from periodic_table_battleship_rl.envs.placement import PlacementEnv
from periodic_table_battleship_rl.game import Fleet, sample_random_legal_fleet
from periodic_table_battleship_rl.placement import (
    DEFAULT_DEFENSIVE_WEIGHTS,
    FrozenDefensiveMixture,
    HuntTargetEvaluator,
    RandomMaskedEvaluator,
    default_defensive_mixture,
)
from periodic_table_battleship_rl.topology import (
    BATTLESHIP,
    PERIODIC_TABLE_BATTLESHIP,
    Topology,
)


TOPOLOGIES = (BATTLESHIP, PERIODIC_TABLE_BATTLESHIP)


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=lambda topology: topology.name)
@pytest.mark.parametrize("evaluator_type", (RandomMaskedEvaluator, HuntTargetEvaluator))
def test_defensive_evaluators_are_seed_reproducible_and_legal(
    topology: Topology, evaluator_type: type[RandomMaskedEvaluator | HuntTargetEvaluator]
) -> None:
    """Each policy sinks a legal fleet with valid shots only on both boards."""

    fleet = sample_random_legal_fleet(topology, np.random.default_rng(2026))
    evaluator = evaluator_type(topology)

    first = evaluator.evaluate(fleet, rng=np.random.default_rng(99))
    second = evaluator.evaluate(fleet, rng=np.random.default_rng(99))

    assert first == second
    assert fleet.segment_count <= first <= topology.valid_cell_count


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=lambda topology: topology.name)
def test_default_mixture_has_stable_components_and_seeded_result(topology: Topology) -> None:
    """The frozen benchmark mixture has stable IDs and deterministic draws."""

    fleet = sample_random_legal_fleet(topology, np.random.default_rng(27))
    evaluator = default_defensive_mixture(topology)

    assert evaluator.evaluator_id == "frozen-defensive-mixture-v1"
    assert evaluator.component_ids == ("random-masked-v1", "hunt-target-v1")
    assert evaluator.weights == DEFAULT_DEFENSIVE_WEIGHTS
    assert evaluator.evaluate(fleet, rng=np.random.default_rng(5)) == evaluator.evaluate(
        fleet, rng=np.random.default_rng(5)
    )


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=lambda topology: topology.name)
def test_default_mixture_is_compatible_with_the_placement_environment(topology: Topology) -> None:
    """The suite satisfies ``PlacementEvaluator`` without leaking fleet state."""

    environment = PlacementEnv(topology, default_defensive_mixture(topology))
    environment.reset(seed=42)
    terminated = False
    info: dict[str, object] = {}
    while not terminated:
        action = int(np.flatnonzero(environment.action_masks())[0])
        _, _, terminated, _, info = environment.step(action)

    assert isinstance(info["valid_shots_to_sink"], int)
    assert 17 <= info["valid_shots_to_sink"] <= topology.valid_cell_count


@dataclass(frozen=True)
class _FixedEvaluator:
    evaluator_id: str
    value: int

    def evaluate(self, fleet: Fleet, *, rng: np.random.Generator) -> int:
        del fleet, rng
        return self.value


def test_mixture_normalizes_its_immutable_weights() -> None:
    """Equivalent positive weights create the same frozen distribution."""

    mixture = FrozenDefensiveMixture(
        evaluators=(_FixedEvaluator("first-v1", 17), _FixedEvaluator("second-v1", 25)),
        weights=(2.0, 6.0),
    )

    assert mixture.weights == (0.25, 0.75)
    assert mixture.component_ids == ("first-v1", "second-v1")


@pytest.mark.parametrize(
    ("evaluators", "weights", "message"),
    (
        ((), (), "at least one"),
        ((_FixedEvaluator("one-v1", 17),), (1.0, 2.0), "same length"),
        (
            (_FixedEvaluator("same-v1", 17), _FixedEvaluator("same-v1", 18)),
            (1.0, 1.0),
            "unique",
        ),
        ((_FixedEvaluator("one-v1", 17),), (0.0,), "positive"),
    ),
)
def test_mixture_rejects_an_ambiguous_or_invalid_specification(
    evaluators: tuple[_FixedEvaluator, ...], weights: tuple[float, ...], message: str
) -> None:
    """A frozen suite cannot silently change its sampling definition."""

    with pytest.raises(ValueError, match=message):
        FrozenDefensiveMixture(evaluators=evaluators, weights=weights)
