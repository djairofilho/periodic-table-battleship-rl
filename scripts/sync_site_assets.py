"""Copy curated artifacts into docs/assets with stable names and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = {
    "artifacts/v0.6-bayes-planner-validation/belief-policy-comparison.png": "belief-policy-comparison-v0.6.png",
    "artifacts/v0.6-bayes-planner-validation/belief-demo-heatmaps.png": "belief-demo-heatmaps-v0.6.png",
    "artifacts/v0.7-bayes-cross-topology-validation/full/paired-valid-shots.png": "paired-valid-shots-v0.7.png",
    "artifacts/v0.7-bayesian-students/student-valid-shots.png": "student-valid-shots-v0.7.png",
    "artifacts/v0.7-bayes-sampler-calibration/belief-sampler-calibration.png": "belief-sampler-calibration-v0.7.png",
    "artifacts/v0.3-fixed-suite/figures/attack-periodic-table-battleship-learning-curve.gif": "attack-periodic-learning-curve-v0.3.gif",
    "artifacts/v0.3-fixed-suite/figures/periodic-ppo-attack.gif": "periodic-ppo-attack-v0.3.gif",
    "artifacts/v0.3-fixed-suite/figures/periodic-ppo-placement.gif": "periodic-ppo-placement-v0.3.gif",
    "artifacts/v0.3-fixed-suite/figures/attack-test-comparison.png": "attack-test-comparison-v0.3.png",
}


@dataclass(frozen=True, slots=True)
class SyncEntry:
    source: Path
    destination: Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "docs" / "assets" / "site-asset-manifest-v0.8.json",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / "docs" / "assets",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falha se algum ativo esperado não existir no estado atual.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    destination_root = arguments.destination
    manifest = arguments.manifest
    destination_root.mkdir(parents=True, exist_ok=True)

    entries = [
        SyncEntry(source=ROOT / key, destination=destination_root / value)
        for key, value in DEFAULT_MANIFEST.items()
    ]
    copied: list[dict[str, str]] = []

    for entry in entries:
        if not entry.source.exists():
            if arguments.strict:
                raise FileNotFoundError(f"Missing source asset: {entry.source}")
            print(f"Skipped (missing): {entry.source}")
            continue
        shutil.copy2(entry.source, entry.destination)
        copied.append(
            {
                "source": str(entry.source.relative_to(ROOT)),
                "destination": str(entry.destination.relative_to(ROOT)),
                "sha256": _sha256(entry.destination),
            }
        )

    manifest.write_text(
        json.dumps(
            {
                "generated_by": "scripts/sync_site_assets.py",
                "files": copied,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(copied)} assets to {destination_root}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
