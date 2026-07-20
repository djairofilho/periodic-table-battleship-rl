from __future__ import annotations

import io
import json

import pytest

from periodic_table_battleship_rl.demo.attack import (
    AttackDemoReplay,
    ReplayMismatchError,
    load_public_replay,
    parse_public_action,
    play_interactive_demo,
    run_baseline_demo,
    save_public_replay,
    verify_public_replay,
)
from periodic_table_battleship_rl.demo.__main__ import main
from periodic_table_battleship_rl.topology import BATTLESHIP, PERIODIC_TABLE_BATTLESHIP


def test_baseline_demo_is_seed_reproducible_and_contains_public_data_only() -> None:
    first = run_baseline_demo(BATTLESHIP, seed=71, policy_id="hunt_target-v1")
    second = run_baseline_demo(BATTLESHIP, seed=71, policy_id="hunt_target-v1")

    assert first == second
    assert first.policy_id == "hunt_target-v1"
    assert first.steps[-1].terminated
    payload = json.dumps(first.to_dict())
    assert "fleet" not in payload
    assert "occupied" not in payload
    assert "ship_id" not in payload
    verify_public_replay(first)


def test_saved_replay_is_strict_and_reexecutes(tmp_path) -> None:
    replay = run_baseline_demo(PERIODIC_TABLE_BATTLESHIP, seed=72)
    destination = save_public_replay(replay, tmp_path / "public-replay.json")

    loaded = load_public_replay(destination)

    assert loaded == replay
    verify_public_replay(loaded)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["steps"][0]["is_hit"] = not payload["steps"][0]["is_hit"]
    tampered = AttackDemoReplay.from_dict(payload)
    with pytest.raises(ReplayMismatchError, match="step 1"):
        verify_public_replay(tampered)


def test_loader_rejects_secret_or_unknown_fields(tmp_path) -> None:
    replay = run_baseline_demo(BATTLESHIP, seed=73)
    payload = replay.to_dict()
    payload["fleet"] = {"occupied_cells": [0]}
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_public_replay(path)


def test_interactive_demo_uses_public_board_and_can_stop_early() -> None:
    output = io.StringIO()
    inputs = iter(["0", "quit"])

    replay = play_interactive_demo(
        BATTLESHIP,
        seed=74,
        input_fn=lambda _: next(inputs),
        output=output,
    )

    text = output.getvalue()
    assert replay.policy_id == "human-v1"
    assert len(replay.steps) == 1
    assert "# unhit ship" not in text
    assert "#" not in text
    assert "seed=74" in text
    verify_public_replay(replay)


def test_parse_public_action_accepts_indices_and_coordinates() -> None:
    assert parse_public_action("0", BATTLESHIP) == 0
    assert parse_public_action("1, 2", BATTLESHIP) == 20
    with pytest.raises(ValueError, match="jogável"):
        parse_public_action("0,1", PERIODIC_TABLE_BATTLESHIP)


def test_cli_saves_and_verifies_public_replay(tmp_path, capsys) -> None:
    destination = tmp_path / "cli-replay.json"

    assert main(["--topology", "battleship", "--seed", "75", "--replay-out", str(destination)]) == 0
    assert main(["--replay", str(destination)]) == 0

    output = capsys.readouterr().out
    assert "policy=hunt_target-v1" in output
    assert "Replay público verificado" in output
