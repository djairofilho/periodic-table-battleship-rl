"""Run a validation-only MaskablePPO CNN pilot for issue #49.

This is deliberately not a blind-test runner.  It emits a checkpoint and a
fixed validation curve; promotion and blind evaluation stay in the release
protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from periodic_table_battleship_rl.topology import get_topology
from periodic_table_battleship_rl.training.cnn import (
    CnnAttackTrainingConfig,
    train_cnn_attack_policy,
)
from periodic_table_battleship_rl.training.attack import AttackValidationConfig


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SEEDS = (5101, 5102, 5103, 5104, 5105)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="battleship")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    topology = get_topology(arguments.scenario)
    timesteps = 256 if arguments.smoke else arguments.timesteps
    n_steps = min(256, timesteps)
    checkpoint_steps = tuple(
        sorted({max(1, timesteps // 2), timesteps})
    )
    run_id = f"v05-cnn-{topology.name}-s{arguments.seed}"
    artifact = train_cnn_attack_policy(
        topology,
        CnnAttackTrainingConfig(
            run_id=run_id,
            seed=arguments.seed,
            total_timesteps=timesteps,
            checkpoint_directory=ROOT / ".local-runs" / "v0.5-cnn",
            n_steps=n_steps,
            batch_size=min(64, n_steps),
            features_dim=arguments.features_dim,
            device="cpu",
        ),
        validation=AttackValidationConfig(
            seeds=VALIDATION_SEEDS,
            checkpoint_steps=checkpoint_steps,
        ),
    )
    report = {
        "run_id": run_id,
        "split": "validation",
        "scenario": topology.name,
        "checkpoint": str(artifact.checkpoint_path),
        "metadata": str(artifact.metadata_path),
        "checkpoints": [
            {
                "training_step": checkpoint.training_step,
                "mean_valid_shots": fmean(
                    result.valid_shots for result in checkpoint.validation_results
                ),
            }
            for checkpoint in artifact.checkpoints
        ],
        "note": "No blind-test seeds were used for this pilot.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
