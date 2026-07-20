from __future__ import annotations

import numpy as np
import pytest

from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.rendering import (
    AttackFrame,
    AttackTraceRecorder,
    capture_attack_frame,
    render_attack_frame,
    render_episode_trace,
    render_topology,
)
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP


@pytest.mark.parametrize("topology", [BATTLESHIP, PERIODIC_TABLE_BATTLESHIP])
def test_topology_render_is_deterministic_and_preserves_gaps(topology) -> None:
    first = render_topology(topology)
    second = render_topology(topology)

    assert first == second
    assert first.splitlines()[0].startswith("    00 01")
    assert len(first.splitlines()) == topology.rows + 2
    assert "legend: · playable cell; blank gap" in first

    periodic_gap = PERIODIC_TABLE_BATTLESHIP.action_for(0, 1)
    assert "00 | ·     " in render_topology(PERIODIC_TABLE_BATTLESHIP)
    assert periodic_gap not in PERIODIC_TABLE_BATTLESHIP.valid_actions


def test_public_attack_frame_and_trace_never_include_secret_fleet() -> None:
    env = AttackEnv(BATTLESHIP)
    initial_observation, _ = env.reset(seed=44)
    assert env._fleet is not None

    recorder = AttackTraceRecorder(env.topology, initial_observation)
    action = next(iter(env._fleet.occupied_cells))
    observation, reward, terminated, truncated, info = env.step(action)
    recorder.record(
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info={**info, "fleet": env._fleet, "occupied_cells": env._fleet.occupied_cells},
        observation=observation,
    )

    public_frame = capture_attack_frame(env)
    public_payload = recorder.build().to_dict()
    public_text = render_attack_frame(public_frame)

    assert public_frame.secret_occupied_cells is None
    assert "secret_occupied_cells" not in public_payload["initial_frame"]
    assert "secret_occupied_cells" not in public_payload["steps"][0]["frame"]
    assert "fleet" not in str(public_payload)
    assert "# unhit ship" not in public_text
    assert "#" not in public_text


def test_secret_attack_frame_requires_explicit_opt_in_and_marks_only_unhit_segments() -> None:
    env = AttackEnv(BATTLESHIP)
    env.reset(seed=7)
    assert env._fleet is not None
    hit = env._fleet.placements[0].cells[0]
    env.step(hit)

    public_text = render_attack_frame(capture_attack_frame(env))
    secret_frame = capture_attack_frame(env, reveal_fleet=True)
    secret_text = render_attack_frame(secret_frame)

    assert secret_frame.secret_occupied_cells == tuple(sorted(env._fleet.occupied_cells))
    assert "# unhit ship" not in public_text
    assert "# unhit ship" in secret_text
    row, column = env.topology.coordinate_for(hit)
    row_tokens = secret_text.splitlines()[row + 1].split("| ", maxsplit=1)[1].split()
    assert row_tokens[column] == "H"


def test_trace_is_json_ready_ordered_and_uses_only_public_transition_fields() -> None:
    env = AttackEnv(PERIODIC_TABLE_BATTLESHIP)
    initial_observation, _ = env.reset(seed=8)
    recorder = AttackTraceRecorder(env.topology, initial_observation)
    action = int(np.flatnonzero(env.action_masks())[0])
    observation, reward, terminated, truncated, info = env.step(action)
    recorder.record(
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        observation=observation,
    )

    trace = recorder.build()
    payload = trace.to_dict()
    text = render_episode_trace(trace)

    assert payload["steps"][0]["index"] == 1
    assert payload["steps"][0]["action"] == action
    assert set(payload["steps"][0]["info"]) <= {
        "episode_id",
        "invalid_attempts",
        "is_hit",
        "sunk_ship_length",
        "valid_shots",
    }
    assert "topology=periodic-table-battleship" in text
    assert "step=001" in text


def test_frame_rejects_wrong_observation_shape_and_overlapping_outcomes() -> None:
    with pytest.raises(ValueError, match="shape"):
        AttackFrame.from_observation(BATTLESHIP, np.zeros((3, 10, 18), dtype=np.uint8))

    observation = np.zeros((4, 10, 18), dtype=np.uint8)
    observation[1, 0, 0] = 1
    observation[3, 0, 0] = 1
    with pytest.raises(ValueError, match="must not overlap"):
        AttackFrame.from_observation(BATTLESHIP, observation)
