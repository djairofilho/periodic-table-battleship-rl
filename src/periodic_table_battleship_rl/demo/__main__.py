"""Command-line entry point for the local attack demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from periodic_table_battleship_rl.demo.attack import (
    BaselinePolicyId,
    load_public_replay,
    play_interactive_demo,
    run_baseline_demo,
    save_public_replay,
    verify_public_replay,
)
from periodic_table_battleship_rl.topology import get_topology


def main(argv: Sequence[str] | None = None) -> int:
    """Run, save, or verify a public terminal replay."""

    parser = argparse.ArgumentParser(description="Demonstração local de ataque")
    parser.add_argument(
        "--topology",
        default="periodic-table-battleship",
        choices=("battleship", "dense-118", "periodic-table-battleship"),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--policy",
        default="hunt_target-v1",
        choices=("random_masked-v1", "hunt_target-v1"),
        help="política usada fora do modo --interactive",
    )
    parser.add_argument("--interactive", action="store_true", help="jogar manualmente")
    parser.add_argument("--replay-out", type=Path, help="arquivo JSON público a criar")
    parser.add_argument("--replay", type=Path, help="arquivo JSON público a verificar")
    arguments = parser.parse_args(argv)

    if arguments.replay is not None:
        if arguments.interactive or arguments.replay_out is not None:
            parser.error("--replay não pode ser combinado com execução ou gravação")
        replay = load_public_replay(arguments.replay)
        verify_public_replay(replay)
        print(
            "Replay público verificado: "
            f"topology={replay.topology} seed={replay.seed} policy={replay.policy_id} "
            f"steps={len(replay.steps)}"
        )
        return 0

    topology = get_topology(arguments.topology)
    if arguments.interactive:
        replay = play_interactive_demo(topology, seed=arguments.seed, output=sys.stdout)
    else:
        policy_id: BaselinePolicyId = arguments.policy
        replay = run_baseline_demo(topology, seed=arguments.seed, policy_id=policy_id)
        print(
            "Demonstração concluída: "
            f"topology={replay.topology} seed={replay.seed} policy={replay.policy_id} "
            f"steps={len(replay.steps)}"
        )

    if arguments.replay_out is not None:
        destination = save_public_replay(replay, arguments.replay_out)
        print(f"Replay público salvo em {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
