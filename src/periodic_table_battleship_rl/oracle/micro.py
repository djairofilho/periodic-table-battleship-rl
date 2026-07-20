"""Exact public-belief Battleship oracle for small rectangular boards.

This module intentionally targets a microboard, not the 10 by 10 benchmark.
It enumerates every legal hidden fleet once, then plans only from public
outcomes.  A :class:`BeliefState` holds the *set of compatible fleet indices*,
not the actual fleet selected for an episode.  That is the exact information
state of a finite POMDP with a uniform prior over enumerated fleets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isclose

import numpy as np


class ShotOutcome(StrEnum):
    """The public result of a shot in the microgame."""

    MISS = "miss"
    HIT = "hit"
    WIN = "win"


@dataclass(frozen=True, slots=True)
class MicroBoardConfig:
    """A tiny rectangular board and its hidden fleet lengths.

    The default 3 by 3 board with one length-two ship has twelve equally likely
    legal fleets.  It is purposely small enough for an exact dynamic-program
    solution while retaining the central Battleship inference problem.
    """

    rows: int = 3
    columns: int = 3
    ship_lengths: tuple[int, ...] = (2,)

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("rows and columns must be positive")
        if not self.ship_lengths:
            raise ValueError("ship_lengths must not be empty")
        if any(length <= 0 for length in self.ship_lengths):
            raise ValueError("ship lengths must be positive")
        if sum(self.ship_lengths) > self.cell_count:
            raise ValueError("fleet contains more segments than board cells")

    @property
    def cell_count(self) -> int:
        """Return the number of valid cells in the rectangular microboard."""

        return self.rows * self.columns

    def coordinate_for(self, action: int) -> tuple[int, int]:
        """Return the row-major coordinate for one valid action."""

        if action not in range(self.cell_count):
            raise ValueError(f"action must be in [0, {self.cell_count})")
        return divmod(action, self.columns)

    def neighbors(self, action: int) -> tuple[int, ...]:
        """Return orthogonal neighbours in deterministic row-major order."""

        row, column = self.coordinate_for(action)
        candidates = (
            (row - 1, column),
            (row, column - 1),
            (row, column + 1),
            (row + 1, column),
        )
        return tuple(
            candidate_row * self.columns + candidate_column
            for candidate_row, candidate_column in candidates
            if 0 <= candidate_row < self.rows and 0 <= candidate_column < self.columns
        )


@dataclass(frozen=True, slots=True)
class MicroFleet:
    """One physical hidden fleet represented exclusively by bit masks."""

    ship_masks: tuple[int, ...]

    @property
    def occupied_mask(self) -> int:
        """Return the union of all ship segments."""

        occupied = 0
        for ship_mask in self.ship_masks:
            occupied |= ship_mask
        return occupied


@dataclass(frozen=True, slots=True)
class BeliefState:
    """Sufficient public information state for the exact solver.

    ``candidate_ids`` are the fleets consistent with every public result so
    far.  They are posterior support, not hidden-state access.  ``tried_mask``
    and ``hit_mask`` are public row-major cell masks.
    """

    candidate_ids: tuple[int, ...]
    tried_mask: int = 0
    hit_mask: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_ids:
            raise ValueError("a nonterminal belief needs at least one candidate")
        if self.tried_mask < 0 or self.hit_mask < 0:
            raise ValueError("public masks must be non-negative")
        if self.hit_mask & ~self.tried_mask:
            raise ValueError("every hit must also be tried")


@dataclass(frozen=True, slots=True)
class OracleSolution:
    """The exact expected cost and optimal first actions of the initial state."""

    expected_shots: float
    optimal_actions: tuple[int, ...]
    action_values: Mapping[int, float]
    solved_states: int


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Exact expected cost of one public-state baseline policy."""

    name: str
    expected_shots: float
    regret_vs_oracle: float


@dataclass(frozen=True, slots=True)
class OracleComparison:
    """The oracle and exact expectation of all reference baselines."""

    config: MicroBoardConfig
    fleet_count: int
    oracle: OracleSolution
    baselines: tuple[BaselineResult, ...]


PolicyDistribution = Callable[[BeliefState, "ExactBattleshipOracle"], Mapping[int, float]]


def enumerate_fleets(config: MicroBoardConfig) -> tuple[MicroFleet, ...]:
    """Enumerate each unique non-overlapping physical fleet exactly once.

    Equal-length ships are de-duplicated by their sorted masks because their
    labels are not public and must not change the uniform physical-layout prior.
    Ships may touch, matching the project's benchmark contract.
    """

    placements_by_length = {
        length: _ship_placements(config, length) for length in set(config.ship_lengths)
    }
    fleets: dict[tuple[int, ...], MicroFleet] = {}

    def visit(index: int, occupied: int, ship_masks: tuple[int, ...]) -> None:
        if index == len(config.ship_lengths):
            canonical = tuple(sorted(ship_masks))
            fleets.setdefault(canonical, MicroFleet(canonical))
            return
        length = config.ship_lengths[index]
        for placement in placements_by_length[length]:
            if occupied & placement:
                continue
            visit(index + 1, occupied | placement, ship_masks + (placement,))

    visit(0, 0, ())
    if not fleets:
        raise ValueError("microboard configuration has no legal complete fleet")
    return tuple(fleets[key] for key in sorted(fleets))


class ExactBattleshipOracle:
    """Memoized Bellman solver that never exposes a selected hidden fleet."""

    def __init__(self, config: MicroBoardConfig = MicroBoardConfig()) -> None:
        self.config = config
        self.fleets = enumerate_fleets(config)
        self._values: dict[BeliefState, float] = {}
        self._action_values: dict[BeliefState, dict[int, float]] = {}

    @property
    def initial_state(self) -> BeliefState:
        """Return the public state before the first shot."""

        return BeliefState(tuple(range(len(self.fleets))))

    def valid_actions(self, state: BeliefState) -> tuple[int, ...]:
        """Return all untried cells in deterministic row-major order."""

        return tuple(
            action
            for action in range(self.config.cell_count)
            if not state.tried_mask & _bit(action)
        )

    def transitions(
        self, state: BeliefState, action: int
    ) -> tuple[tuple[ShotOutcome, float, BeliefState | None], ...]:
        """Return exact public-outcome branches of firing one valid action."""

        if action not in self.valid_actions(state):
            raise ValueError("action must be an untried microboard cell")
        partitions: dict[ShotOutcome, list[int]] = defaultdict(list)
        for candidate_id in state.candidate_ids:
            outcome = self._outcome(self.fleets[candidate_id], state.tried_mask, action)
            partitions[outcome].append(candidate_id)

        next_tried = state.tried_mask | _bit(action)
        branches: list[tuple[ShotOutcome, float, BeliefState | None]] = []
        candidate_count = len(state.candidate_ids)
        for outcome in ShotOutcome:
            candidate_ids = partitions.get(outcome)
            if not candidate_ids:
                continue
            probability = len(candidate_ids) / candidate_count
            if outcome is ShotOutcome.WIN:
                branches.append((outcome, probability, None))
                continue
            next_hits = state.hit_mask | (_bit(action) if outcome is ShotOutcome.HIT else 0)
            branches.append(
                (
                    outcome,
                    probability,
                    BeliefState(tuple(candidate_ids), next_tried, next_hits),
                )
            )
        return tuple(branches)

    def action_value(self, state: BeliefState, action: int) -> float:
        """Return the exact Bellman cost of taking ``action`` in ``state``."""

        expected_follow_up = sum(
            probability * (0.0 if next_state is None else self.value(next_state))
            for _, probability, next_state in self.transitions(state, action)
        )
        return 1.0 + expected_follow_up

    def value(self, state: BeliefState) -> float:
        """Return the optimal expected number of additional valid shots."""

        cached = self._values.get(state)
        if cached is not None:
            return cached
        values = {action: self.action_value(state, action) for action in self.valid_actions(state)}
        value = min(values.values())
        self._values[state] = value
        self._action_values[state] = values
        return value

    def solve(self) -> OracleSolution:
        """Solve the initial public belief state exactly with memoization."""

        initial = self.initial_state
        expected_shots = self.value(initial)
        action_values = self._action_values[initial]
        optimal_actions = tuple(
            action
            for action, value in action_values.items()
            if isclose(value, expected_shots, rel_tol=0.0, abs_tol=1e-12)
        )
        return OracleSolution(
            expected_shots=expected_shots,
            optimal_actions=optimal_actions,
            action_values=dict(action_values),
            solved_states=len(self._values),
        )

    def occupancy_probabilities(self, state: BeliefState) -> np.ndarray:
        """Return posterior occupancy probability for every board cell."""

        probabilities = np.zeros(self.config.cell_count, dtype=float)
        for candidate_id in state.candidate_ids:
            occupied = self.fleets[candidate_id].occupied_mask
            for action in range(self.config.cell_count):
                probabilities[action] += bool(occupied & _bit(action))
        return probabilities / len(state.candidate_ids)

    def evaluate_policy(self, policy: PolicyDistribution) -> float:
        """Evaluate a stochastic public-state policy exactly, without sampling."""

        cache: dict[BeliefState, float] = {}

        def evaluate(state: BeliefState) -> float:
            cached = cache.get(state)
            if cached is not None:
                return cached
            distribution = _validate_distribution(policy(state, self), self.valid_actions(state))
            result = 0.0
            for action, action_probability in distribution.items():
                if action_probability == 0:
                    continue
                follow_up = sum(
                    probability * (0.0 if next_state is None else evaluate(next_state))
                    for _, probability, next_state in self.transitions(state, action)
                )
                result += action_probability * (1.0 + follow_up)
            cache[state] = result
            return result

        return evaluate(self.initial_state)

    def _outcome(self, fleet: MicroFleet, tried_mask: int, action: int) -> ShotOutcome:
        action_bit = _bit(action)
        if not fleet.occupied_mask & action_bit:
            return ShotOutcome.MISS
        after_shot = tried_mask | action_bit
        if fleet.occupied_mask & ~after_shot == 0:
            return ShotOutcome.WIN
        return ShotOutcome.HIT


def evaluate_baselines(
    config: MicroBoardConfig = MicroBoardConfig(),
) -> OracleComparison:
    """Compare random, hunt-target and posterior-greedy with the exact oracle."""

    oracle = ExactBattleshipOracle(config)
    solution = oracle.solve()
    policies: tuple[tuple[str, PolicyDistribution], ...] = (
        ("random-masked", _random_distribution),
        ("hunt-target", _hunt_target_distribution),
        ("posterior-greedy", _posterior_greedy_distribution),
    )
    baselines = tuple(
        BaselineResult(
            name=name,
            expected_shots=expected_shots,
            regret_vs_oracle=expected_shots - solution.expected_shots,
        )
        for name, policy in policies
        for expected_shots in (oracle.evaluate_policy(policy),)
    )
    return OracleComparison(config, len(oracle.fleets), solution, baselines)


def _ship_placements(config: MicroBoardConfig, length: int) -> tuple[int, ...]:
    placements: set[int] = set()
    for row in range(config.rows):
        for column in range(config.columns):
            if column + length <= config.columns:
                placements.add(_segment_mask(config, row, column, 0, 1, length))
            if length > 1 and row + length <= config.rows:
                placements.add(_segment_mask(config, row, column, 1, 0, length))
    return tuple(sorted(placements))


def _segment_mask(
    config: MicroBoardConfig,
    row: int,
    column: int,
    row_step: int,
    column_step: int,
    length: int,
) -> int:
    return sum(
        _bit((row + offset * row_step) * config.columns + column + offset * column_step)
        for offset in range(length)
    )


def _random_distribution(
    state: BeliefState, oracle: ExactBattleshipOracle
) -> Mapping[int, float]:
    valid_actions = oracle.valid_actions(state)
    probability = 1.0 / len(valid_actions)
    return {action: probability for action in valid_actions}


def _hunt_target_distribution(
    state: BeliefState, oracle: ExactBattleshipOracle
) -> Mapping[int, float]:
    valid_actions = oracle.valid_actions(state)
    valid_set = frozenset(valid_actions)
    targets = tuple(
        sorted(
            {
                neighbour
                for hit in range(oracle.config.cell_count)
                if state.hit_mask & _bit(hit)
                for neighbour in oracle.config.neighbors(hit)
                if neighbour in valid_set
            }
        )
    )
    choices = targets or valid_actions
    probability = 1.0 / len(choices)
    return {action: probability for action in choices}


def _posterior_greedy_distribution(
    state: BeliefState, oracle: ExactBattleshipOracle
) -> Mapping[int, float]:
    probabilities = oracle.occupancy_probabilities(state)
    valid_actions = oracle.valid_actions(state)
    best_probability = max(probabilities[action] for action in valid_actions)
    choices = tuple(
        action
        for action in valid_actions
        if isclose(probabilities[action], best_probability, rel_tol=0.0, abs_tol=1e-12)
    )
    probability = 1.0 / len(choices)
    return {action: probability for action in choices}


def _validate_distribution(
    distribution: Mapping[int, float], valid_actions: tuple[int, ...]
) -> Mapping[int, float]:
    if set(distribution) - set(valid_actions):
        raise ValueError("policy assigned mass to an invalid action")
    if not distribution:
        raise ValueError("policy must assign mass to at least one valid action")
    if any(probability < 0 for probability in distribution.values()):
        raise ValueError("policy probabilities must be non-negative")
    total = float(sum(distribution.values()))
    if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("policy probabilities must sum to one")
    return distribution


def _bit(action: int) -> int:
    return 1 << action
