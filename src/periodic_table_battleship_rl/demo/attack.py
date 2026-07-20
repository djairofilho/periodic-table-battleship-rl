"""Playable terminal attack demo and public, reproducible replay files.

The replay format intentionally persists only an episode seed, policy identity,
actions, and public transition outcomes.  It never serialises a fleet, ship
identifier, occupied-cell collection, or a secret rendering frame.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import numpy as np

from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.policies import hunt_target_action, random_masked_action
from periodic_table_battleship_rl.rendering import AttackFrame, render_attack_frame
from periodic_table_battleship_rl.topology import Topology, get_topology

REPLAY_SCHEMA_VERSION = "attack-demo-replay-v1"
BaselinePolicyId = Literal["random_masked-v1", "hunt_target-v1"]
_PUBLIC_REPLAY_KEYS = frozenset(
    {"schema_version", "topology", "seed", "policy_id", "steps"}
)
_PUBLIC_STEP_KEYS = frozenset(
    {
        "action",
        "is_hit",
        "sunk_ship_length",
        "valid_shots",
        "invalid_attempts",
        "terminated",
        "truncated",
    }
)


class ReplayMismatchError(ValueError):
    """Raised when a purported replay differs from its seeded environment."""


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One public transition of an attack demo."""

    action: int
    is_hit: bool
    sunk_ship_length: int
    valid_shots: int
    invalid_attempts: int
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, int | bool]:
        """Return the deliberately small, JSON-ready public representation."""

        return {
            "action": self.action,
            "is_hit": self.is_hit,
            "sunk_ship_length": self.sunk_ship_length,
            "valid_shots": self.valid_shots,
            "invalid_attempts": self.invalid_attempts,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class AttackDemoReplay:
    """A safe, seed-reproducible action trace for one attack episode."""

    topology: str
    seed: int
    policy_id: str
    steps: tuple[ReplayStep, ...]
    schema_version: str = REPLAY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a public replay payload with no hidden fleet state."""

        return {
            "schema_version": self.schema_version,
            "topology": self.topology,
            "seed": self.seed,
            "policy_id": self.policy_id,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: object) -> AttackDemoReplay:
        """Parse and validate one public replay payload.

        The exact key allow-list makes accidental secret-state additions fail at
        load time instead of silently becoming part of a reusable replay.
        """

        if not isinstance(payload, dict):
            raise ValueError("replay must be a JSON object")
        if set(payload) != _PUBLIC_REPLAY_KEYS:
            raise ValueError("replay contains unsupported or missing fields")
        if payload["schema_version"] != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported replay schema version")
        if not isinstance(payload["topology"], str):
            raise ValueError("replay topology must be a string")
        if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
            raise ValueError("replay seed must be an integer")
        if not isinstance(payload["policy_id"], str) or not payload["policy_id"]:
            raise ValueError("replay policy_id must be a non-empty string")
        if not isinstance(payload["steps"], list):
            raise ValueError("replay steps must be a list")

        steps = tuple(_replay_step_from_dict(item) for item in payload["steps"])
        _validate_step_sequence(steps)
        return cls(
            topology=payload["topology"],
            seed=payload["seed"],
            policy_id=payload["policy_id"],
            steps=steps,
        )


def run_baseline_demo(
    topology: Topology,
    *,
    seed: int,
    policy_id: BaselinePolicyId = "hunt_target-v1",
) -> AttackDemoReplay:
    """Run one deterministic baseline episode using only public state."""

    env = AttackEnv(topology)
    observation, _ = env.reset(seed=seed)
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 71]))
    steps: list[ReplayStep] = []
    terminated = truncated = False

    while not (terminated or truncated):
        action = _baseline_action(policy_id, topology, env, observation, policy_rng)
        observation, _, terminated, truncated, info = env.step(action)
        steps.append(_public_step(action, terminated, truncated, info))

    return AttackDemoReplay(topology.name, seed, policy_id, tuple(steps))


def play_interactive_demo(
    topology: Topology,
    *,
    seed: int,
    input_fn: Callable[[str], str] = input,
    output: TextIO,
) -> AttackDemoReplay:
    """Play a masked episode in a terminal without rendering secret cells.

    Enter either an action index (``0`` through ``179``) or ``row,column``.
    ``quit`` writes an intentionally incomplete but valid public replay.
    """

    env = AttackEnv(topology)
    observation, _ = env.reset(seed=seed)
    steps: list[ReplayStep] = []
    output.write(
        f"topology={topology.name} seed={seed} policy=human-v1\n"
        "Ação: índice 0..179 ou linha,coluna. Digite quit para encerrar.\n"
    )
    _write_public_board(output, topology, observation)

    terminated = truncated = False
    while not (terminated or truncated):
        raw_action = input_fn("ação> ").strip()
        if raw_action.lower() in {"quit", "exit", "sair"}:
            break
        try:
            action = parse_public_action(raw_action, topology)
        except ValueError as error:
            output.write(f"Entrada inválida: {error}\n")
            continue
        if not env.action_masks()[action]:
            output.write("Ação indisponível: escolha uma célula jogável ainda não chamada.\n")
            continue

        observation, _, terminated, truncated, info = env.step(action)
        step = _public_step(action, terminated, truncated, info)
        steps.append(step)
        outcome = "acerto" if step.is_hit else "água"
        suffix = f"; navio de {step.sunk_ship_length} afundado" if step.sunk_ship_length else ""
        output.write(f"{outcome}{suffix}\n")
        _write_public_board(output, topology, observation)

    if terminated:
        output.write(f"Frota encontrada em {steps[-1].valid_shots} tiros válidos.\n")
    elif truncated:
        output.write("Partida truncada pelo limite de tentativas.\n")
    else:
        output.write("Partida interrompida. O replay contém somente as jogadas públicas.\n")
    return AttackDemoReplay(topology.name, seed, "human-v1", tuple(steps))


def parse_public_action(value: str, topology: Topology) -> int:
    """Parse an action index or a zero-based ``row,column`` public coordinate."""

    if not value:
        raise ValueError("informe uma ação")
    if "," in value:
        pieces = [piece.strip() for piece in value.split(",")]
        if len(pieces) != 2:
            raise ValueError("use linha,coluna")
        try:
            row, column = (int(piece) for piece in pieces)
            action = topology.action_for(row, column)
        except ValueError as error:
            raise ValueError("coordenada fora do tabuleiro") from error
    else:
        try:
            action = int(value)
        except ValueError as error:
            raise ValueError("use um índice inteiro ou linha,coluna") from error

    if not topology.is_valid_action(action):
        raise ValueError("a ação não aponta para uma célula jogável")
    return action


def save_public_replay(replay: AttackDemoReplay, path: str | Path) -> Path:
    """Write a UTF-8 public replay after re-validating its serialised form."""

    destination = Path(path)
    payload = replay.to_dict()
    AttackDemoReplay.from_dict(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def load_public_replay(path: str | Path) -> AttackDemoReplay:
    """Load a strict public replay file without accepting hidden payloads."""

    try:
        payload: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("replay não contém JSON válido") from error
    return AttackDemoReplay.from_dict(payload)


def verify_public_replay(replay: AttackDemoReplay) -> None:
    """Re-execute every action and require its public results to match exactly."""

    topology = get_topology(replay.topology)
    env = AttackEnv(topology)
    env.reset(seed=replay.seed)
    ended = False
    for index, recorded in enumerate(replay.steps, start=1):
        if ended:
            raise ReplayMismatchError("replay contains a step after episode completion")
        _, _, terminated, truncated, info = env.step(recorded.action)
        actual = _public_step(recorded.action, terminated, truncated, info)
        if actual != recorded:
            raise ReplayMismatchError(f"step {index} differs from the seeded environment")
        ended = terminated or truncated


def _baseline_action(
    policy_id: BaselinePolicyId,
    topology: Topology,
    env: AttackEnv,
    observation: np.ndarray,
    rng: np.random.Generator,
) -> int:
    mask = env.action_masks()
    if policy_id == "random_masked-v1":
        return random_masked_action(mask, rng)
    if policy_id == "hunt_target-v1":
        active_hits = _actions_for_channel(topology, observation, channel=1)
        return hunt_target_action(topology, mask, active_hits, rng)
    raise ValueError(f"unsupported baseline policy: {policy_id!r}")


def _actions_for_channel(
    topology: Topology, observation: np.ndarray, *, channel: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            topology.action_for(int(row), int(column))
            for row, column in np.argwhere(observation[channel] != 0)
        )
    )


def _public_step(
    action: int,
    terminated: bool,
    truncated: bool,
    info: dict[str, int | bool],
) -> ReplayStep:
    return ReplayStep(
        action=action,
        is_hit=bool(info["is_hit"]),
        sunk_ship_length=int(info["sunk_ship_length"]),
        valid_shots=int(info["valid_shots"]),
        invalid_attempts=int(info["invalid_attempts"]),
        terminated=terminated,
        truncated=truncated,
    )


def _replay_step_from_dict(value: object) -> ReplayStep:
    if not isinstance(value, dict) or set(value) != _PUBLIC_STEP_KEYS:
        raise ValueError("replay step contains unsupported or missing fields")
    bool_keys = ("is_hit", "terminated", "truncated")
    int_keys = ("action", "sunk_ship_length", "valid_shots", "invalid_attempts")
    if any(not isinstance(value[key], bool) for key in bool_keys):
        raise ValueError("replay boolean fields must be booleans")
    if any(
        isinstance(value[key], bool) or not isinstance(value[key], int) for key in int_keys
    ):
        raise ValueError("replay numeric fields must be integers")
    return ReplayStep(
        action=value["action"],
        is_hit=value["is_hit"],
        sunk_ship_length=value["sunk_ship_length"],
        valid_shots=value["valid_shots"],
        invalid_attempts=value["invalid_attempts"],
        terminated=value["terminated"],
        truncated=value["truncated"],
    )


def _validate_step_sequence(steps: Sequence[ReplayStep]) -> None:
    for index, step in enumerate(steps, start=1):
        if step.action < 0 or step.sunk_ship_length < 0:
            raise ValueError("replay actions and sunk lengths must be non-negative")
        if step.valid_shots < 0 or step.invalid_attempts < 0:
            raise ValueError("replay counters must be non-negative")
        if step.valid_shots + step.invalid_attempts != index:
            raise ValueError("replay counters must equal the recorded step count")
        if index < len(steps) and (step.terminated or step.truncated):
            raise ValueError("replay cannot continue after completion")


def _write_public_board(output: TextIO, topology: Topology, observation: np.ndarray) -> None:
    frame = AttackFrame.from_observation(topology, observation)
    output.write(render_attack_frame(frame) + "\n")
