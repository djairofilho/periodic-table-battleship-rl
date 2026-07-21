"""Generate a larger public Bayesian demonstration dataset for v0.9.

The produced assets only contain public observations, public masks, selected
actions and teacher occupancy scores.  Two independent inventories are generated
per topology: `train` and `validation`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from periodic_table_battleship_rl.topology import BATTLESHIP, DENSE_118, PERIODIC_TABLE_BATTLESHIP
from periodic_table_battleship_rl.training.bayesian_distillation import (
    BayesianDemonstrationConfig,
    generate_bayesian_demonstrations,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "v0.9-demonstrations"
SCENARIOS = (
    BATTLESHIP,
    DENSE_118,
    PERIODIC_TABLE_BATTLESHIP,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--train-seed-start", type=int, default=11_001)
    parser.add_argument("--validation-seed-start", type=int, default=12_001)
    parser.add_argument("--train-seed-count", type=int, default=3)
    parser.add_argument("--validation-seed-count", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sampler-seed", type=int, default=19_001)
    parser.add_argument("--dataset-version", default="v0.9")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    train_seed_count = 1 if arguments.smoke else arguments.train_seed_count
    validation_seed_count = 1 if arguments.smoke else arguments.validation_seed_count
    sample_count = 2 if arguments.smoke else arguments.sample_count

    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    dataset_root = output / f"bayesian-teacher-{arguments.dataset_version}"
    report: dict[str, object] = {
        "campaign": "v0.9-bayesian-demonstrations",
        "dataset_version": arguments.dataset_version,
        "smoke": arguments.smoke,
        "train_seed_count": train_seed_count,
        "validation_seed_count": validation_seed_count,
        "sample_count": sample_count,
        "sampler_seed": arguments.sampler_seed,
        "topologies": {},
    }

    for topology in SCENARIOS:
        train_seeds = tuple(
            range(arguments.train_seed_start, arguments.train_seed_start + train_seed_count)
        )
        validation_seeds = tuple(
            range(
                arguments.validation_seed_start,
                arguments.validation_seed_start + validation_seed_count,
            )
        )
        train_dataset = generate_bayesian_demonstrations(
            topology,
            BayesianDemonstrationConfig(
                dataset_id=f"{topology.name}-{arguments.dataset_version}-train",
                seeds=train_seeds,
                output_directory=dataset_root,
                sample_count=sample_count,
                sampler_seed=arguments.sampler_seed,
            ),
        )
        validation_dataset = generate_bayesian_demonstrations(
            topology,
            BayesianDemonstrationConfig(
                dataset_id=f"{topology.name}-{arguments.dataset_version}-validation",
                seeds=validation_seeds,
                output_directory=dataset_root,
                sample_count=sample_count,
                sampler_seed=arguments.sampler_seed + 1,
            ),
        )
        report["topologies"][topology.name] = {
            "valid_cells": topology.valid_cell_count,
            "train_dataset": {
                "path": str(train_dataset.data_path.relative_to(ROOT)),
                "metadata": str(train_dataset.metadata_path.relative_to(ROOT)),
                "decisions": train_dataset.sample_count,
                "sha256": train_dataset.data_sha256,
            },
            "validation_dataset": {
                "path": str(validation_dataset.data_path.relative_to(ROOT)),
                "metadata": str(validation_dataset.metadata_path.relative_to(ROOT)),
                "decisions": validation_dataset.sample_count,
                "sha256": validation_dataset.data_sha256,
            },
        }

    (output / "dataset-manifest-v0.9.json").write_text(
        json.dumps(_with_hashes(report, dataset_root), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _write_summary(report, output / "dataset-manifest-v0.9.md")


def _with_hashes(report: dict[str, object], dataset_root: Path) -> dict[str, object]:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    total_decisions = 0
    for topology_report in topologies.values():
        assert isinstance(topology_report, dict)
        for split_name in ("train_dataset", "validation_dataset"):
            dataset = topology_report[split_name]
            assert isinstance(dataset, dict)
            npz = dataset["path"]
            assert isinstance(npz, str)
            absolute = ROOT / npz
            dataset["file_sha256"] = _sha256(absolute)
            total_decisions += int(dataset["decisions"])
            dataset["exists"] = absolute.exists()
    report["total_decisions"] = total_decisions
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_summary(report: dict[str, object], destination: Path) -> None:
    topologies = report["topologies"]
    assert isinstance(topologies, dict)
    rows: list[str] = []
    for name, topology_report in topologies.items():
        assert isinstance(topology_report, dict)
        train = topology_report["train_dataset"]
        validation = topology_report["validation_dataset"]
        rows.append(
            f"| `{name}` | {train['decisions']} | {validation['decisions']} | {train['sha256'][:8]} | {validation['sha256'][:8]} |"
        )

    lines = [
        "# Conjunto de demonstrações Bayesiano (v0.9)",
        "",
        f"- Versão: `{report['dataset_version']}`",
        f"- Total de decisões: `{report['total_decisions']}`",
        "| Cenário | Treino | Validação | SHA train | SHA validação |",
        "| --- | ---: | ---: | --- | --- |",
        *rows,
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
