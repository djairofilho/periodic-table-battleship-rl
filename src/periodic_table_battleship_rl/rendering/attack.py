"""Text and JSON-ready renderings for attack episodes.

The public renderer is deliberately built from the public observation alone.
It cannot include a fleet unless a caller explicitly asks for a secret frame,
which is intended only for local diagnostics after an episode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from periodic_table_battleship_rl.topology import Topology

if TYPE_CHECKING:
    from periodic_table_battleship_rl.envs.attack import AttackEnv


_PUBLIC_INFO_KEYS = (
    "episode_id",
    "invalid_attempts",
    "is_hit",
    "sunk_ship_length",
    "valid_shots",
)


@dataclass(frozen=True, slots=True)
class AttackFrame:
    """One attack-board snapshot suitable for text, JSON, or GIF pipelines.

    ``secret_occupied_cells`` is ``None`` for public frames.  Its absence is
    intentional: serialising or rendering a public frame must never disclose
    where an unhit ship segment is located.
    """

    topology_name: str
    rows: int
    columns: int
    valid_actions: tuple[int, ...]
    active_hits: tuple[int, ...]
    sunk_hits: tuple[int, ...]
    misses: tuple[int, ...]
    secret_occupied_cells: tuple[int, ...] | None = None

    @classmethod
    def from_observation(
        cls,
        topology: Topology,
        observation: np.ndarray,
        *,
        secret_occupied_cells: Sequence[int] | None = None,
    ) -> AttackFrame:
        """Create a frame from the four public AttackEnv observation channels.

        A non-``None`` ``secret_occupied_cells`` opt-in creates a diagnostic
        frame.  Callers should not persist or publish such a frame.
        """

        expected_shape = (4, topology.rows, topology.columns)
        if observation.shape != expected_shape:
            raise ValueError(
                f"attack observation must have shape {expected_shape}, got {observation.shape}"
            )

        valid_actions = tuple(sorted(topology.valid_actions))
        valid_set = set(valid_actions)
        channels = tuple(
            tuple(
                sorted(
                    topology.action_for(int(row), int(column))
                    for row, column in np.argwhere(observation[channel] != 0)
                )
            )
            for channel in range(1, 4)
        )
        active_hits, sunk_hits, misses = channels
        _validate_public_cells(valid_set, active_hits, sunk_hits, misses)

        secret_cells = None
        if secret_occupied_cells is not None:
            secret_cells = tuple(sorted(set(secret_occupied_cells)))
            if not set(secret_cells).issubset(valid_set):
                raise ValueError("secret occupied cells must be valid topology actions")

        return cls(
            topology_name=topology.name,
            rows=topology.rows,
            columns=topology.columns,
            valid_actions=valid_actions,
            active_hits=active_hits,
            sunk_hits=sunk_hits,
            misses=misses,
            secret_occupied_cells=secret_cells,
        )

    @property
    def is_secret(self) -> bool:
        """Whether this frame includes hidden fleet occupancy."""

        return self.secret_occupied_cells is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation, omitting secrets by default."""

        payload: dict[str, Any] = {
            "topology": self.topology_name,
            "rows": self.rows,
            "columns": self.columns,
            "valid_actions": list(self.valid_actions),
            "active_hits": list(self.active_hits),
            "sunk_hits": list(self.sunk_hits),
            "misses": list(self.misses),
        }
        if self.secret_occupied_cells is not None:
            payload["secret_occupied_cells"] = list(self.secret_occupied_cells)
        return payload


@dataclass(frozen=True, slots=True)
class AttackTraceStep:
    """One public attack transition and its board snapshot."""

    index: int
    action: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, int | bool]
    frame: AttackFrame

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready, public-only transition representation."""

        return {
            "index": self.index,
            "action": self.action,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": dict(self.info),
            "frame": self.frame.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AttackEpisodeTrace:
    """A public, frame-by-frame trace of a single attack episode."""

    initial_frame: AttackFrame
    steps: tuple[AttackTraceStep, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return an ordered JSON-ready trace for debugging or animation."""

        return {
            "initial_frame": self.initial_frame.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


class AttackTraceRecorder:
    """Record public attack observations without retaining the hidden fleet."""

    def __init__(self, topology: Topology, initial_observation: np.ndarray) -> None:
        self._topology = topology
        self._initial_frame = AttackFrame.from_observation(topology, initial_observation)
        self._steps: list[AttackTraceStep] = []

    def record(
        self,
        *,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, object],
        observation: np.ndarray,
    ) -> AttackTraceStep:
        """Append one transition, retaining only the documented public info."""

        public_info = _public_info(info)
        step = AttackTraceStep(
            index=len(self._steps) + 1,
            action=int(action),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=public_info,
            frame=AttackFrame.from_observation(self._topology, observation),
        )
        self._steps.append(step)
        return step

    def build(self) -> AttackEpisodeTrace:
        """Freeze the accumulated snapshots into an immutable episode trace."""

        return AttackEpisodeTrace(self._initial_frame, tuple(self._steps))


def capture_attack_frame(env: AttackEnv, *, reveal_fleet: bool = False) -> AttackFrame:
    """Capture an environment board, exposing ships only with explicit opt-in.

    The environment has no public observation property between transitions, so
    this adapter intentionally reads its current observation.  The secret fleet
    is read only when ``reveal_fleet`` is true.
    """

    observation = env._observation()
    secret_cells: Sequence[int] | None = None
    if reveal_fleet:
        if env._fleet is None:
            raise RuntimeError("reset() must be called before revealing a fleet")
        secret_cells = tuple(env._fleet.occupied_cells)
    return AttackFrame.from_observation(
        env.topology,
        observation,
        secret_occupied_cells=secret_cells,
    )


def render_topology(topology: Topology) -> str:
    """Render playable geometry only, using a blank token for gaps."""

    return _render_grid(
        topology.rows,
        topology.columns,
        topology.valid_actions,
        {},
        legend="· playable cell; blank gap",
    )


def render_attack_frame(frame: AttackFrame) -> str:
    """Render a deterministic attack board from a public or secret frame."""

    states: dict[int, str] = {action: "·" for action in frame.valid_actions}
    if frame.secret_occupied_cells is not None:
        states.update({action: "#" for action in frame.secret_occupied_cells})
    states.update({action: "H" for action in frame.active_hits})
    states.update({action: "S" for action in frame.sunk_hits})
    states.update({action: "o" for action in frame.misses})
    legend = "· unknown; H hit; S sunk; o miss"
    if frame.secret_occupied_cells is not None:
        legend += "; # unhit ship"
    return _render_grid(
        frame.rows,
        frame.columns,
        frame.valid_actions,
        states,
        legend=legend,
    )


def render_episode_trace(trace: AttackEpisodeTrace) -> str:
    """Render a concise public transition log for reproducible diagnostics."""

    lines = [
        f"topology={trace.initial_frame.topology_name}",
        f"steps={len(trace.steps)}",
    ]
    for step in trace.steps:
        outcome = "hit" if step.info.get("is_hit") else "miss"
        lines.append(
            "step={index:03d} action={action:03d} outcome={outcome} "
            "reward={reward:+.1f} terminated={terminated} truncated={truncated}".format(
                index=step.index,
                action=step.action,
                outcome=outcome,
                reward=step.reward,
                terminated=step.terminated,
                truncated=step.truncated,
            )
        )
    return "\n".join(lines)


def _render_grid(
    rows: int,
    columns: int,
    valid_actions: Sequence[int],
    states: Mapping[int, str],
    *,
    legend: str,
) -> str:
    valid_set = set(valid_actions)
    header = "    " + " ".join(f"{column:02d}" for column in range(columns))
    lines = [header]
    for row in range(rows):
        tokens = []
        for column in range(columns):
            action = row * columns + column
            tokens.append(states.get(action, "·") if action in valid_set else " ")
        lines.append(f"{row:02d} | " + "  ".join(tokens))
    lines.append(f"legend: {legend}")
    return "\n".join(lines)


def _public_info(info: Mapping[str, object]) -> dict[str, int | bool]:
    public_info: dict[str, int | bool] = {}
    for key in _PUBLIC_INFO_KEYS:
        value = info.get(key)
        if isinstance(value, bool):
            public_info[key] = value
        elif isinstance(value, int):
            public_info[key] = value
    return public_info


def _validate_public_cells(
    valid_actions: set[int],
    active_hits: Sequence[int],
    sunk_hits: Sequence[int],
    misses: Sequence[int],
) -> None:
    channels = (active_hits, sunk_hits, misses)
    if any(not set(cells).issubset(valid_actions) for cells in channels):
        raise ValueError("attack outcomes must be valid topology actions")
    union = set().union(*channels)
    if sum(len(cells) for cells in channels) != len(union):
        raise ValueError("attack outcome channels must not overlap")
