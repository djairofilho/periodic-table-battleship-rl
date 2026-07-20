"""Generate a small, reproducible public Bayesian teacher dataset for v0.7.

The default is intentionally an audit-sized training artifact.  It contains
only training seeds, never the fixed validation or blind-test inventories.
Use a larger explicit seed schedule after selecting it through validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training.bayesian_distillation import (
    BayesianDemonstrationConfig,
    generate_bayesian_demonstrations,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (9701,)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="v0.7-bayesian-public-sample")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    artifact = generate_bayesian_demonstrations(
        BATTLESHIP,
        BayesianDemonstrationConfig(
            dataset_id=args.dataset_id,
            seeds=tuple(args.seeds),
            output_directory=args.output_directory,
            sample_count=args.sample_count,
            sampler_seed=args.sampler_seed,
        ),
    )
    print(
        json.dumps(
            {
                "data_path": str(artifact.data_path),
                "metadata_path": str(artifact.metadata_path),
                "sample_count": artifact.sample_count,
                "scenario": artifact.scenario,
                "data_sha256": artifact.data_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
