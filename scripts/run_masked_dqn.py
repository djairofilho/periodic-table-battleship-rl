"""Run the reproducible masked-DQN pilot on periodic-table Battleship.

The default is a small pilot, deliberately labelled non-promotional.  A full
candidate campaign must use a frozen validation/test protocol before any
README comparison is updated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from periodic_table_battleship_rl.evaluation import RunConfig
from periodic_table_battleship_rl.experiments import run_dqn_attack_evaluation
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.topology import PERIODIC_TABLE_BATTLESHIP
from periodic_table_battleship_rl.training import (
    DQN_ATTACK_POLICY_ID,
    DqnAttackTrainingConfig,
    load_masked_dqn_attack_policy,
    train_masked_dqn_attack_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--test-seeds", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.steps <= 0 or args.seed < 0 or args.test_seeds <= 0:
        raise ValueError("steps and test-seeds must be positive; seed must be non-negative")
    run_id = f"dqn-periodic-s{args.seed}"
    artifact = train_masked_dqn_attack_policy(
        PERIODIC_TABLE_BATTLESHIP,
        DqnAttackTrainingConfig(
            run_id=run_id,
            seed=args.seed,
            total_steps=args.steps,
            checkpoint_directory=ROOT / ".local-runs" / "dqn-pilot",
            warmup_steps=min(512, max(64, args.steps // 4)),
        ),
    )
    evaluation = run_dqn_attack_evaluation(
        RunConfig(
            run_id=f"{run_id}-pilot",
            experiment="attack",
            scenario=PERIODIC_TABLE_BATTLESHIP.name,
            environment_version=ENVIRONMENT_VERSION,
            policy_id=DQN_ATTACK_POLICY_ID,
            split="validation",
            seeds=tuple(range(7201, 7201 + args.test_seeds)),
            episodes_per_seed=1,
            parameters={"campaign": "dqn-pilot", "promotion_eligible": False},
        ),
        PERIODIC_TABLE_BATTLESHIP,
        load_masked_dqn_attack_policy(artifact.checkpoint_path),
        ROOT / "runs" / "dqn-pilot" / run_id,
        checkpoint_path=artifact.checkpoint_path,
        training_metadata_path=artifact.metadata_path,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip(),
        uv_lock_path=ROOT / "uv.lock",
    )
    print(evaluation.summary["aggregate"]["valid_shots"])


if __name__ == "__main__":
    main()
