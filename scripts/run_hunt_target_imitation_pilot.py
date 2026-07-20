"""Generate public hunt-target demonstrations and run a validation-only pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from periodic_table_battleship_rl.topology import get_topology
from periodic_table_battleship_rl.training.attack import AttackValidationConfig
from periodic_table_battleship_rl.training.imitation import (
    HuntTargetDatasetConfig,
    ImitationTrainingConfig,
    generate_hunt_target_dataset,
    train_hunt_target_imitation,
)


ROOT = Path(__file__).resolve().parents[1]
DEMONSTRATION_SEEDS = tuple(range(7101, 7121))
VALIDATION_SEEDS = (8101, 8102, 8103, 8104, 8105)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="battleship")
    parser.add_argument("--fine-tune-timesteps", type=int, default=50_000)
    parser.add_argument("--cloning-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.fine_tune_timesteps < 0:
        raise ValueError("--fine-tune-timesteps must not be negative")
    topology = get_topology(arguments.scenario)
    fine_tune = 256 if arguments.smoke else arguments.fine_tune_timesteps
    epochs = 1 if arguments.smoke else arguments.cloning_epochs
    root = ROOT / ".local-runs" / "v0.5-hunt-target-imitation"
    dataset = generate_hunt_target_dataset(
        topology,
        HuntTargetDatasetConfig(
            dataset_id=f"hunt-target-{topology.name}-public-v1",
            seeds=DEMONSTRATION_SEEDS,
            output_directory=root / "datasets",
        ),
    )
    checkpoints = () if fine_tune == 0 else tuple(sorted({fine_tune // 2, fine_tune}))
    artifact = train_hunt_target_imitation(
        topology,
        ImitationTrainingConfig(
            run_id=f"v05-imitation-{topology.name}-s{arguments.seed}",
            seed=arguments.seed,
            dataset_path=dataset.data_path,
            checkpoint_directory=root / "models",
            cloning_epochs=epochs,
            fine_tune_timesteps=fine_tune,
            fine_tune_checkpoint_steps=checkpoints,
            ppo_n_steps=256,
            ppo_batch_size=64,
            device="cpu",
        ),
        validation=AttackValidationConfig(
            seeds=VALIDATION_SEEDS,
            checkpoint_steps=checkpoints or (1,),
        ),
    )
    report = {
        "split": "validation",
        "scenario": topology.name,
        "dataset": str(dataset.data_path),
        "dataset_metadata": str(dataset.metadata_path),
        "public_samples": dataset.sample_count,
        "behavior_clone": str(artifact.behavior_clone_path),
        "final_checkpoint": str(artifact.final_checkpoint_path),
        "selected_checkpoint": str(artifact.selected_checkpoint_path),
        "cloning_loss": list(artifact.cloning_loss),
        "behavior_clone_mean_valid_shots": fmean(
            result.valid_shots
            for result in artifact.behavior_clone_validation.validation_results
        ),
        "fine_tune_checkpoints": [
            {
                "training_step": checkpoint.training_step,
                "mean_valid_shots": fmean(
                    result.valid_shots for result in checkpoint.validation_results
                ),
            }
            for checkpoint in artifact.fine_tune_checkpoints
        ],
        "note": "No blind-test seeds were used for dataset creation or selection.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
