"""Evaluate selected v0.3 PPO attackers across all board topologies.

This is an explicit transfer experiment, not a replacement for the normal
same-topology benchmark guardrail.  It chooses one pre-selected validation
checkpoint (the first fixed training seed) for each v0.3 topology, then runs
the complete ordered train-by-test matrix on the held-out v0.3 test seeds.
The public report never includes local checkpoint paths, only their hashes,
source seed, and selected validation step.

Run the complete matrix with::

    uv run --extra train python scripts/run_cross_topology_v0_3.py

Use ``--seed-count 3`` for a quick wiring check.  The default is the complete
100-seed held-out schedule used by the v0.3 attack comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

from periodic_table_battleship_rl.evaluation import RunConfig
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.experiments import (
    CROSS_TOPOLOGY_PROTOCOL,
    CrossTopologyPpoSource,
    run_cross_topology_matrix,
)
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.topology import (
    BATTLESHIP,
    DENSE_118,
    PERIODIC_TABLE_BATTLESHIP,
    Topology,
)
from periodic_table_battleship_rl.training import ATTACK_POLICY_ID, load_attack_policy


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_REPORT = ROOT / "artifacts" / "v0.3-fixed-suite" / "campaign-report.json"
CHECKPOINT_ROOT = ROOT / ".local-runs" / "v0.3-fixed-suite" / "attack-final"
RUN_DIRECTORY = ROOT / "runs" / "v0.4-cross-topology"
ARTIFACT_DIRECTORY = ROOT / "artifacts" / "v0.4-cross-topology"
TEST_SEEDS = tuple(range(5101, 5201))
ATTACK_TOPOLOGIES = (BATTLESHIP, DENSE_118, PERIODIC_TABLE_BATTLESHIP)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=len(TEST_SEEDS),
        help="prefix length of the fixed v0.3 held-out test schedule",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device passed to MaskablePPO loading (default: cpu)",
    )
    return parser.parse_args()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _representative_sources(device: str) -> tuple[CrossTopologyPpoSource, ...]:
    """Load one validation-selected fixed seed per source topology.

    v0.3 selected a checkpoint using validation data independently for every
    final training seed.  The first fixed seed (3101) is used here as a
    pre-declared representative; its choice is independent of cross-topology
    test results.  The report records the seed and checkpoint step so a later
    multi-seed transfer analysis can expand this protocol without ambiguity.
    """

    report = json.loads(CAMPAIGN_REPORT.read_text(encoding="utf-8"))
    sources: list[CrossTopologyPpoSource] = []
    for topology in ATTACK_TOPOLOGIES:
        selected = report["attack"][topology.name]["selected_final_checkpoints"]
        choice = next(item for item in selected if item["seed"] == 3101)
        run_id = _attack_run_id(topology, int(choice["seed"]))
        checkpoint = (
            CHECKPOINT_ROOT
            / topology.name
            / run_id
            / "checkpoints"
            / f"step-{int(choice['checkpoint_step']):09d}"
            / "model.zip"
        )
        metadata = CHECKPOINT_ROOT / topology.name / run_id / "training.json"
        sources.append(
            CrossTopologyPpoSource(
                topology=topology,
                policy=load_attack_policy(checkpoint, device=device),
                checkpoint_path=checkpoint,
                training_metadata_path=metadata,
            )
        )
    return tuple(sources)


def _attack_run_id(topology: Topology, seed: int) -> str:
    labels = {
        "battleship": "classic",
        "dense-118": "dense",
        "periodic-table-battleship": "periodic",
    }
    return f"v03-attack-{labels[topology.name]}-s{seed}"


def _config(source: Topology, target: Topology, seeds: tuple[int, ...]) -> RunConfig:
    return RunConfig(
        run_id=f"v04-cross-{source.name}-to-{target.name}",
        experiment="attack",
        scenario=target.name,
        environment_version=ENVIRONMENT_VERSION,
        policy_id=ATTACK_POLICY_ID,
        split="test",
        seeds=seeds,
        episodes_per_seed=1,
        parameters={
            "campaign": "v0.4-cross-topology",
            "source_selection": "v0.3-validation-selected-seed-3101",
        },
    )


def _write_report(
    sources: tuple[CrossTopologyPpoSource, ...],
    matrix: Any,
    seeds: tuple[int, ...],
) -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_records = []
    for source in sources:
        parameters = matrix.by_pair()[(source.topology.name, source.topology.name)].manifest.config.parameters
        source_records.append(
            {
                "scenario": source.topology.name,
                "selection": "first-fixed-final-training-seed",
                "training_seed": 3101,
                "checkpoint_step": _selected_step(source.topology.name),
                "checkpoint_sha256": parameters["checkpoint_sha256"],
                "training_metadata_sha256": parameters["training_metadata_sha256"],
            }
        )
    cells = [
        {
            "source_scenario": evaluation.source_topology,
            "target_scenario": evaluation.target_topology,
            "episode_count": len(evaluation.results),
            "valid_shots": evaluation.summary["aggregate"]["valid_shots"],
            "win_rate": evaluation.summary["aggregate"]["win_rate"],
            "run_manifest": str(evaluation.persisted.manifest_path.relative_to(ROOT)),
        }
        for evaluation in matrix.evaluations
    ]
    report = {
        "schema_version": "cross-topology-report-v1",
        "protocol": CROSS_TOPOLOGY_PROTOCOL,
        "source_selection": "v0.3 validation-selected checkpoint for fixed training seed 3101",
        "test_seeds": list(seeds),
        "sources": source_records,
        "matrix": cells,
    }
    write_json_atomic(ARTIFACT_DIRECTORY / "cross-topology-report.json", report)
    _write_csv(cells)
    _write_markdown(cells, len(seeds))


def _selected_step(topology_name: str) -> int:
    report = json.loads(CAMPAIGN_REPORT.read_text(encoding="utf-8"))
    selected = report["attack"][topology_name]["selected_final_checkpoints"]
    return int(next(item for item in selected if item["seed"] == 3101)["checkpoint_step"])


def _write_csv(cells: list[dict[str, Any]]) -> None:
    path = ARTIFACT_DIRECTORY / "cross-topology-matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)


def _write_markdown(cells: list[dict[str, Any]], seed_count: int) -> None:
    rows = [
        "# Transferência PPO entre topologias",
        "",
        f"Protocolo: `{CROSS_TOPOLOGY_PROTOCOL}`. Cada célula usa {seed_count} seeds de teste fixos.",
        "A diagonal é o controle same-topology; as demais células são transferência explícita.",
        "",
        "| Treino | Teste | Média de tiros válidos | Taxa de vitória |",
        "| --- | --- | ---: | ---: |",
    ]
    for cell in cells:
        shots = cell["valid_shots"]
        wins = cell["win_rate"]
        rows.append(
            f"| {cell['source_scenario']} | {cell['target_scenario']} | "
            f"{shots['mean']:.2f} | {wins['mean']:.3f} |"
        )
    (ARTIFACT_DIRECTORY / "cross-topology-summary.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def main() -> None:
    arguments = _arguments()
    if not 1 <= arguments.seed_count <= len(TEST_SEEDS):
        raise ValueError(f"--seed-count must be in [1, {len(TEST_SEEDS)}]")
    seeds = TEST_SEEDS[: arguments.seed_count]
    sources = _representative_sources(arguments.device)
    matrix = run_cross_topology_matrix(
        sources,
        ATTACK_TOPOLOGIES,
        RUN_DIRECTORY,
        lambda source, target: _config(source, target, seeds),
        git_commit=_git_commit(),
        uv_lock_path=ROOT / "uv.lock",
    )
    _write_report(sources, matrix, seeds)


if __name__ == "__main__":
    main()
