"""Rebuild seed-level v0.3 analysis from its public held-out CSV artifacts.

Example:

    uv run --extra visual python scripts/analyze_v0_3_results.py --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

from periodic_table_battleship_rl.analysis.campaign import (
    plot_primary_comparisons,
    write_campaign_analysis,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "v0.3-fixed-suite"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack-csv",
        type=Path,
        default=DEFAULT_ARTIFACTS / "tables" / "attack-test-episodes.csv",
    )
    parser.add_argument(
        "--placement-csv",
        type=Path,
        default=DEFAULT_ARTIFACTS / "tables" / "placement-test-episodes.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACTS / "analysis",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_720)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="also render the primary paired-comparison forest plot (requires --extra visual)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    summaries, comparisons = write_campaign_analysis(
        attack_csv=arguments.attack_csv,
        placement_csv=arguments.placement_csv,
        destination=arguments.output,
        resamples=arguments.resamples,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    if arguments.plot:
        plot_primary_comparisons(comparisons, arguments.output / "primary-comparisons.png")
    print(
        f"wrote {len(summaries)} policy summaries and {len(comparisons)} "
        f"paired comparisons to {arguments.output}"
    )


if __name__ == "__main__":
    main()
